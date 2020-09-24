from models.heads import Scorer
from utils.utils_sample import predict
import numpy as np
import torch
from transformers import BertTokenizer
import torch.nn.functional as F
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Scorer(hidden_dim=256,
        output_dim=5,
        n_layers=2,
        bidirectional=True,
        dropout=0.25).to(device)
model.load_state_dict(torch.load('models/scorers/AmazonReviewFull.pt'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = F.softmax(model(tensor),1)
    pred_t = prediction.argmax(dim=1, keepdim=True)

    return pred_t.item()

def get_loss(model, tokenizer, sentence, label):
    model.eval()
    ce_loss_logging = torch.nn.CrossEntropyLoss()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    label = torch.tensor([label], device='cuda', dtype=torch.long)
    output_t = model(tensor)
    loss = ce_loss_logging(output_t, label).item()
    return loss

# def predict_sentiment(model, tokenizer, sentence):
#     model.eval()
#     tokens = tokenizer.tokenize(sentence)
#     tokens = tokens[:max_input_length-2]
#     indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
#     tensor = torch.LongTensor(indexed).to(device)
#     tensor = tensor.unsqueeze(0)
#     prediction = torch.round(torch.sigmoid(model(tensor)))
    
#     return prediction.item()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    if(score["compound"] >= 0.05): return 2
    elif(score["compound"] > -0.05 and (score["compound"] < 0.05)): return 1
    elif(score["compound"] <= -0.05): return 0

def get_vater_score(sentences, l):
    lable = {"very negative":0,"very positive":2}
    acc = []
    for s in sentences:
        prediciton = sentiment_analyzer_scores(s)
        if(int(prediciton)==int(lable[l])):acc.append(1)
        else: acc.append(0) 

    return np.mean(acc)


def get_sentiment_score(sentences, l):
    lable = {"very negative":0,"very positive":4}
    acc = []
    for s in sentences:
        prediciton = predict_sentiment(model, tokenizer, s)
        if(int(prediciton)==int(lable[l])):acc.append(1)
        else: acc.append(0) 

    return np.mean(acc)

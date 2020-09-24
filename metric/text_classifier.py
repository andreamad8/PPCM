from models.heads import Scorer
from utils.utils_sample import predict
import numpy as np
import torch
from transformers import BertTokenizer
import torch.nn.functional as F
from collections import Counter 

device = "cuda" if torch.cuda.is_available() else "cpu"

model_AGNEWS = Scorer(hidden_dim=256,
        output_dim=4,
        n_layers=2,
        bidirectional=True,
        dropout=0.25).to(device)
model_AGNEWS.load_state_dict(torch.load('models/scorers/TC_AG_NEWS.pt'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

def predict_class(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = F.softmax(model(tensor),1)
    pred_t = prediction.argmax(dim=1, keepdim=True)
    return pred_t.item()


def get_text_score_AGNEWS(sentences,l):
    idx2class = ["World","Sports","Business","SciTech"]
    class2idx = {c:i for i, c in enumerate(idx2class)}
    idx2class = {i: c for i, c in enumerate(idx2class)}
    pred = []
    acc = []
    for s in sentences:
        prediciton = predict_class(model_AGNEWS, tokenizer, s)
        pred.append(idx2class[prediciton])
        if(int(prediciton)==int(class2idx[l])):
            acc.append(1)
        else: acc.append(0) 

    counter = Counter(pred)
    return np.mean(acc) #"".join([ f"({freq/len(sentences)}){w}" for w, freq in counter.most_common(1)])

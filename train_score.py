import argparse
import time
import torch
import torch.optim
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score

from torchtext import data as torchtext_data
from models.heads import Scorer
from transformers import BertTokenizer, AdamW


def train(model, iterator, optimizer, criterion, binary):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    predicted_list = []
    target_list = []
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        
        if(binary):
            predicted_list.append(torch.round(torch.sigmoid(predictions)).tolist())
            acc = binary_accuracy(predictions, batch.label)
        else:
            predicted_list.append(predictions.argmax(dim=1).tolist())
            acc = accuracy(predictions, batch.label)
        target_list.append(batch.label.tolist())
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    if(binary):
        F1 = f1_score(sum(target_list,[]), sum(predicted_list,[]), average='binary')
    else:
        F1 = f1_score(sum(target_list,[]), sum(predicted_list,[]), average='micro')
    return epoch_loss / len(iterator), epoch_acc / len(iterator), F1



def evaluate(model, iterator, criterion, binary):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    predicted_list = []
    target_list = []
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            if(binary):
                acc = binary_accuracy(predictions, batch.label)
                predicted_list.append(torch.round(torch.sigmoid(predictions)).tolist())
            else:
                acc = accuracy(predictions, batch.label)
                predicted_list.append(predictions.argmax(dim=1).tolist())

            target_list.append(batch.label.tolist())
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    if(binary):
        F1 = f1_score(sum(target_list,[]), sum(predicted_list,[]), average='binary')
    else:
        F1 = f1_score(sum(target_list,[]), sum(predicted_list,[]), average='micro')
    return epoch_loss / len(iterator), epoch_acc / len(iterator), F1

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    pred_t = preds.argmax(dim=1, keepdim=True)
    return  pred_t.eq(y.view_as(pred_t)).float().mean()



def train_scorer(
        dataset, dataset_fp=None, pretrained_model="medium",
        epochs=10, batch_size=64, log_interval=10,
        save_model=False, cached=False, no_cuda=False):
    global device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id

    def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence) 
        tokens = tokens[:max_input_length-2]
        return tokens
    TEXT = torchtext_data.Field(batch_first = True,
            use_vocab = False,
            tokenize = tokenize_and_cut,
            preprocessing = tokenizer.convert_tokens_to_ids,
            init_token = init_token_idx,
            eos_token = eos_token_idx,
            pad_token = pad_token_idx,
            unk_token = unk_token_idx)

    if "TC_" in dataset:

        if(dataset == "TC_AG_NEWS"):
            fil = '.data/ag_news_csv'
            idx2class = ["World","Sports","Business","Sci/Tech"]
        elif(dataset == "TC_SogouNews"):
            fil = '.data/sogou_news_csv'
            idx2class = ["Sports","Finance","Entertainment","Automobile","Technology"]
        elif(dataset == "TC_DBpedia"):
            fil = '.data/dbpedia_csv'
            idx2class = ["Company","EducationalInstitution","Artist","Athlete",
                         "OfficeHolder","MeanOfTransportation","Building",
                         "NaturalPlace","Village","Animal","Plant",
                         "Album","Film","WrittenWork"]
        elif(dataset == "TC_YahooAnswers"):
            fil = '.data/yahoo_answers_csv'
            idx2class = ["Society & Culture","Science & Mathematics",
                        "Health","Education & Reference","Computers & Internet",
                        "Sports","Business & Finance","Entertainment & Music",
                        "Family & Relationships","Politics & Government"]


        LABEL = torchtext_data.LabelField(dtype = torch.long)
        train_val_fields = [
            ('label', LABEL), # process it as label
            ('none', None), # process it as label
            ('text', TEXT) # process it as text
        ]

        test_data, train_data = torchtext_data.TabularDataset.splits(path=fil, 
                                                    format='csv', 
                                                    train='train.csv', 
                                                    validation='test.csv', 
                                                    fields=train_val_fields, 
                                                    skip_header=False)
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of testing examples: {len(test_data)}")

        BATCH_SIZE = args.batch_size
        LABEL.build_vocab(train_data)
        train_iterator, test_iterator = torchtext_data.BucketIterator.splits(
                                                        (train_data, test_data), 
                                                        batch_size = BATCH_SIZE, 
                                                        device = device,
                                                        sort_key=lambda x: len(x.text))
        output_dim = len(idx2class)
    elif dataset == "sentiment":
        idx2class = ["neg","pos"]
        class2idx = {c: i for i, c in enumerate(idx2class)}


        LABEL = torchtext_data.LabelField(dtype = torch.float)
        train_data, test_data = torchtext_data.IMDB.splits(TEXT, LABEL)
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of testing examples: {len(test_data)}")


        BATCH_SIZE = args.batch_size
        LABEL.build_vocab(train_data)
        train_iterator, test_iterator = torchtext_data.BucketIterator.splits(
                                                        (train_data, test_data), 
                                                        batch_size = BATCH_SIZE, 
                                                        device = device)
        idx2class = ["neg","pos"]
        output_dim = len(idx2class)
        
    elif dataset == "AmazonReviewFull":
        idx2class = ["1","2","3","4","5"]
        class2idx = {c: i for i, c in enumerate(idx2class)}
        fil = ".data/amazon_review_full_csv"
        LABEL = torchtext_data.LabelField(dtype = torch.long)
        train_val_fields = [
            ('label', LABEL), # process it as label
            ('none', None), # process it as label
            ('text', TEXT) # process it as text
        ]

        test_data, train_data = torchtext_data.TabularDataset.splits(path=fil, 
                                                    format='csv', 
                                                    train='train.csv', 
                                                    validation='test.csv', 
                                                    fields=train_val_fields, 
                                                    skip_header=False)
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of testing examples: {len(test_data)}")


        BATCH_SIZE = args.batch_size
        LABEL.build_vocab(train_data)
        train_iterator, test_iterator = torchtext_data.BucketIterator.splits(
                                                        (train_data, test_data), 
                                                        batch_size = BATCH_SIZE, 
                                                        device = device,
                                                        sort_key=lambda x: len(x.text))
        output_dim = len(idx2class)
    
    elif dataset == "daily_dialogue_emotion":
        LABEL = torchtext_data.LabelField(dtype = torch.long)
        train_val_fields = [
            ('text', TEXT), # process it as text
            ('label', LABEL) # process it as label
        ]

        train_data, test_data = torchtext_data.TabularDataset.splits(path='data/dailydialog', 
                                                    format='tsv', 
                                                    train='train.tsv', 
                                                    validation='test.tsv', 
                                                    fields=train_val_fields, 
                                                    skip_header=True)
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of testing examples: {len(test_data)}")

        BATCH_SIZE = args.batch_size
        LABEL.build_vocab(train_data)
        train_iterator, test_iterator = torchtext_data.BucketIterator.splits(
                                                        (train_data, test_data), 
                                                        batch_size = BATCH_SIZE, 
                                                        device = device,
                                                        sort_key=lambda x: len(x.text))
        idx2class = ["no_emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
        output_dim = len(idx2class)
    elif dataset == "hate_speech":
        # https://github.com/t-davidson
        LABEL = torchtext_data.LabelField(dtype = torch.long)
        train_val_fields = [
            ('text', TEXT), # process it as text
            ('label', LABEL) # process it as label
        ]

        train_data, test_data = torchtext_data.TabularDataset.splits(path='data/hate_speech', 
                                                    format='tsv', 
                                                    train='train.tsv', 
                                                    validation='test.tsv', 
                                                    fields=train_val_fields, 
                                                    skip_header=True)
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of testing examples: {len(test_data)}")

        BATCH_SIZE = args.batch_size
        LABEL.build_vocab(train_data)
        train_iterator, test_iterator = torchtext_data.BucketIterator.splits(
                                                        (train_data, test_data), 
                                                        batch_size = BATCH_SIZE, 
                                                        device = device,
                                                        sort_key=lambda x: len(x.text))
        
        # 0 - hate speech
        # 1 - offensive  language
        # 2 - neither
        idx2class = ["hate", "offensive","neither"]
        output_dim = len(idx2class)

    elif dataset == "wiki_detox":
        # https://github.com/t-davidson
        LABEL = torchtext_data.LabelField(dtype = torch.float)
        train_val_fields = [
            ('text', TEXT), # process it as text
            ('label', LABEL) # process it as label
        ]

        train_data, test_data = torchtext_data.TabularDataset.splits(path='data/wiki_detox', 
                                                    format='tsv', 
                                                    train='train.tsv', 
                                                    validation='test.tsv', 
                                                    fields=train_val_fields, 
                                                    skip_header=True)
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of testing examples: {len(test_data)}")

        BATCH_SIZE = args.batch_size
        LABEL.build_vocab(train_data)
        train_iterator, test_iterator = torchtext_data.BucketIterator.splits(
                                                        (train_data, test_data), 
                                                        batch_size = BATCH_SIZE, 
                                                        device = device,
                                                        sort_key=lambda x: len(x.text))
        
        # 0 - non attack
        # 1 - attack
        idx2class = ["non_attack","attack"]
        output_dim = len(idx2class)


    end = time.time()

    model = Scorer(hidden_dim=256,
            output_dim=1 if output_dim==2 else output_dim,
            n_layers=2,
            bidirectional=True,
            dropout=0.25).to(device)
    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters())
    if output_dim==2:
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    best_valid_loss = float('inf')
    print("Starting Training")
    for epoch in range(epochs):
        
        start_time = time.time()
        
        train_loss, train_acc, train_F1 = train(model, train_iterator, optimizer, criterion,True if output_dim==2 else False)
        valid_loss, valid_acc, valid_F1 = evaluate(model, test_iterator, criterion,True if output_dim==2 else False)
            
        end_time = time.time()
            
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'models/scorers/{args.dataset}.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% |Train F1: {train_F1*100:.2f}% ')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Val. F1: {valid_F1*100:.2f}% ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--dataset", type=str, default="sentiment",
                        choices=("sentiment", "clickbait", "toxic", "hate_speech","wiki_detox",
                                 "daily_dialogue_topics","daily_dialogue_act",
                                 "daily_dialogue_emotion","generic","emocap","NLI","MNLI","DNLI",
                                 "empathetic_dialogue","TC_AG_NEWS","TC_SogouNews","TC_DBpedia","TC_YahooAnswers",
                                 "AmazonReviewFull"),
                        help="dataset to train the discriminator on."
                             "In case of generic, the dataset is expected"
                             "to be a TSBV file with structure: class \\t text")
    parser.add_argument("--dataset_fp", type=str, default="",
                        help="File path of the dataset to use. "
                             "Needed only in case of generic datadset")
    parser.add_argument("--pretrained_model", type=str, default="medium",
                        help="Pretrained model to use as encoder")
    parser.add_argument("--epochs", type=int, default=5, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save_model", action="store_true",
                        help="whether to save the model")
    parser.add_argument("--cached", action="store_true",
                        help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true",
                        help="use to turn off cuda")
    args = parser.parse_args()

    train_scorer(**(vars(args)))




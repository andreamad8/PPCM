import argparse
import time
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from models.heads import Discriminator
from sklearn.metrics import f1_score
from utils.torchtext_text_classification import AG_NEWS 
torch.manual_seed(0)
np.random.seed(0)

device = "cuda"
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 128

class Dataset(data.Dataset):
    def __init__(self, X, y, entailment=False):
        """Reads source and target sequences from txt files."""
        self.entailment = entailment
        self.X = X
        self.y = y

    def __len__(self):
        if(self.entailment): return len(self.X[0])
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        if(self.entailment):
            data["X_p"] = self.X[0][index]
            data["X_h"] = self.X[1][index]
        else:
            data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    if("X_p" in item_info):
        x_p_batch, _ = pad_sequences(item_info["X_p"])
        x_h_batch, _ = pad_sequences(item_info["X_h"])
        x_batch = (x_p_batch,x_h_batch)
    else:
        x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def train_epoch(args,data_loader, discriminator, optimizer,
                epoch=0, log_interval=10,cached=False,
                entailment=False, loss_type=False):
    samples_so_far = 0
    discriminator.train_custom()
    
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    for batch_idx, (input_t, target_t) in tqdm(enumerate(data_loader)):
        if(entailment and not cached):
            input_p, input_h, target_t = input_t[0].to(device),input_t[1].to(device), target_t.to(device)
            input_t = (input_p, input_h)
        else:
            input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        if(loss_type):
            t = torch.zeros(target_t.size(0)).fill_(args.label).to(device).int()
            target = target_t.eq(t.view_as(target_t)).float()
            loss = bce_loss(output_t, target.unsqueeze(-1))
        else:
            loss = ce_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    # acc = correct.sum() / len(correct)
    return correct.sum()

def evaluate_performance(args,data_loader, discriminator, cached=False, entailment=False, loss_type=False):
    discriminator.eval()
    test_loss = 0
    correct = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    predicted_list = []
    target_list = []
    with torch.no_grad():
        for input_t, target_t in tqdm(data_loader):
            if(entailment):
                input_p, input_h, target_t = input_t[0].to(device),input_t[1].to(device), target_t.to(device)
                input_t = (input_p, input_h)
            else:
                input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            if(loss_type):
                t = torch.zeros(target_t.size(0)).fill_(args.label).to(device).int()
                target = target_t.eq(t.view_as(target_t)).float()

                test_loss += bce_loss(output_t, target.unsqueeze(-1))
                predicted_list.append(torch.round(torch.sigmoid(output_t)).tolist())
                target_list.append(target.tolist())
                correct += binary_accuracy(output_t, target.unsqueeze(-1))
            else:
                test_loss += ce_loss(output_t, target_t).item()
                # get the index of the max log-probability
                pred_t = output_t.argmax(dim=1, keepdim=True)
                correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()
                predicted_list.append(pred_t.squeeze().tolist())
                target_list.append(target_t.tolist())

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    if(loss_type):
        F1 = f1_score(sum(target_list,[]), sum(predicted_list,[]))
    else:
        F1 = f1_score(sum(target_list,[]), sum(predicted_list,[]), average='micro')


    return test_loss, accuracy, F1




def get_cached_data_loader(dataset, batch_size, discriminator, entailment=False, shuffle=False):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader

def train_discriminator(
        dataset, dataset_fp=None, pretrained_model="medium",
        epochs=10, batch_size=64, log_interval=10,
        save_model=False, cached=False, no_cuda=False,
        bce_loss=False, label=3):
    global device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if "TC_" in dataset:
        if(dataset == "TC_AG_NEWS"):
            idx2class = ["World","Sports","Business","Sci/Tech"]


        class2idx = {c: i for i, c in enumerate(idx2class)}
        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached
        ).to(device)

        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        if(dataset == "TC_AG_NEWS"):
            train_data_iter,test_data_iter = AG_NEWS()

        x = []
        y = []
        # i = 0
        for label, text in train_data_iter:
            seq = discriminator.tokenizer.encode(text)[:128]
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            x.append(seq)
            y.append(label)
            # i+=1
            # if(i==10):break
        train_dataset = Dataset(x, y)

        test_x = []
        test_y = []
        # i = 0
        for label, text in test_data_iter:
            seq = discriminator.tokenizer.encode(text)[:128]
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            test_x.append(seq)
            test_y.append(label)
            # i+=1
            # if(i==1000):break
        test_dataset = Dataset(test_x, test_y)

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 2,
        }

    elif dataset == "sentiment":
        idx2class = ["positive", "negative", "very positive", "very negative",
                     "neutral"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=1 if bce_loss else len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached
        ).to(device)

        text = torchtext_data.Field()
        label = torchtext_data.Field(sequential=False)
        train_data, val_data, test_data = datasets.SST.splits(
            text,
            label,
            fine_grained=True,
            train_subtrees=True,
        )

        x = []
        y = []
        for i in trange(len(train_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(
                vars(train_data[i])["text"]
            )
            seq = discriminator.tokenizer.encode(seq)
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            x.append(seq)
            y.append(class2idx[vars(train_data[i])["label"]])
            # if(i==10): break
        train_dataset = Dataset(x, y)

        test_x = []
        test_y = []
        for i in trange(len(test_data), ascii=True):
            seq = TreebankWordDetokenizer().detokenize(
                vars(test_data[i])["text"]
            )
            seq = discriminator.tokenizer.encode(seq)
            seq = torch.tensor(seq, device=device, dtype=torch.long)
            test_x.append(seq)
            test_y.append(class2idx[vars(test_data[i])["label"]])
            # if(i==10): break
        test_dataset = Dataset(test_x, test_y)

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 2,
        }


    elif("daily_dialogue" in dataset):
        in_dial = open("data/dailydialog/dialogues_text.txt", 'r')
        if("act" in dataset):
            idx2class = ["inform", "question", "directive", "commissive"]
            in_lable = open("data/dailydialog/dialogues_act.txt", 'r')


        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached).to(device)
            
        max_length_seq = 128
        
        x = []
        y = []
        for i, (line_dial, line_lable) in enumerate(tqdm(zip(in_dial,in_lable), ascii=True)):
            history = line_dial.split('__eou__')
            history = history[:-1]
            history = [h.strip().replace(" , ",", ")
                            .replace(" . ",". ").replace(" .",".")
                            .replace(" ? ","? ").replace(" ?","?")
                            .replace(" ’ ","’").replace(" : ",": ")
                            for h in history]
            if ("emotion" in dataset or "act" in dataset):
                lables = line_lable.split(" ")
                lables = lables[:-1]
                if len(lables) != len(history):
                    continue
            for id_turn, h in enumerate(history):
                seq = discriminator.tokenizer.encode(h) 
                if len(seq) < max_length_seq:
                    seq = torch.tensor(
                        seq, device=device, dtype=torch.long
                    )

                    x.append(seq)
                    if("act" in dataset):
                        y.append(int(lables[id_turn])-1)
                    else:
                        y.append(int(lables[id_turn]))

                else:
                    print("Line {} is longer than maximum length {}".format(
                        i, max_length_seq
                    ))
                    
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1, stratify=y)
        train_dataset = Dataset(X_train, y_train) 
        test_dataset = Dataset(X_val, y_val)

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }


    end = time.time()
    print(f"Train:{len(train_dataset)}")
    print(f"Test:{len(test_dataset)}")
    # print("Preprocessed {} data points".format(
    #     len(train_dataset) + len(test_dataset))
    # )
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached and ("NLI" not in dataset):
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(
            train_dataset, batch_size, discriminator,entailment=False,shuffle=True
        )

        test_loader = get_cached_data_loader(
            test_dataset, batch_size, discriminator,entailment=False
        )

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  collate_fn=collate_fn)
                                                  
    optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

    loss_per_epoch = []
    accuracy_per_epoch = []
    F1_per_epoch = []
    
    loss_per_epoch_train = []
    accuracy_per_epoch_train = []
    F1_per_epoch_train = []
    
    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            args=args,
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,cached=cached,
            entailment=True if "NLI" in args.dataset else False,
            loss_type=bce_loss
        )

        loss_train, accuracy_train, f1_train = evaluate_performance(
            args=args, 
            data_loader=train_loader,
            discriminator=discriminator,cached=cached,
            entailment=True if "NLI" in args.dataset else False,
            loss_type=bce_loss
        )
        loss_per_epoch_train.append(loss_train)
        accuracy_per_epoch_train.append(accuracy_train)
        F1_per_epoch_train.append(f1_train)

        loss, accuracy, f1 = evaluate_performance(
            args=args,
            data_loader=test_loader,
            discriminator=discriminator,cached=cached,
            entailment=True if "NLI" in args.dataset else False,
            loss_type=bce_loss
        )
        loss_per_epoch.append(loss)
        accuracy_per_epoch.append(accuracy)
        F1_per_epoch.append(f1)

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))
        print(f"TRAIN: Acc {accuracy_train} F1 {f1_train}")
        print(f"TEST: Acc {accuracy} F1 {f1}")
        print()
        # print("\nExample prediction")
        # predict(example_sentence, discriminator, idx2class, cached)

        if save_model:
            torch.save(discriminator.get_classifier().state_dict(),
                       "models/discriminators/DIALOGPT_{}_classifier_head_epoch_{}.pt".format(dataset,
                                                               epoch + 1))
            if bce_loss:
                torch.save(discriminator.get_classifier().state_dict(),
                       f"models/discriminators/TEST/BCE_DIALOGPT_{dataset}_classifier_{args.label}_lab_head_epoch_{epoch+1}.pt")
    print()
    epoch_min_loss = loss_per_epoch.index(min(loss_per_epoch))
    print(f"TRAIN Minimum loss {epoch_min_loss + 1} ACC:{accuracy_per_epoch_train[epoch_min_loss]} F1:{F1_per_epoch_train[epoch_min_loss]}" )
    print(f"TEST Minimum loss {epoch_min_loss + 1} ACC:{accuracy_per_epoch[epoch_min_loss]} F1:{F1_per_epoch[epoch_min_loss]}" )

    epoch_max_accuracy = accuracy_per_epoch.index(max(accuracy_per_epoch))
    print("Maximum accuracy on test set obtained at epoch", epoch_max_accuracy + 1)
    print(f"TRAIN Minimum loss {epoch_max_accuracy + 1} ACC:{accuracy_per_epoch_train[epoch_max_accuracy]} F1:{F1_per_epoch_train[epoch_max_accuracy]}" )
    print(f"TEST Minimum loss {epoch_max_accuracy + 1} ACC:{accuracy_per_epoch[epoch_max_accuracy]} F1:{F1_per_epoch[epoch_max_accuracy]}" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--dataset", type=str, default="sentiment",
                        choices=("sentiment", "clickbait", "toxic", 
                                 "daily_dialogue_topics","daily_dialogue_act",
                                 "daily_dialogue_emotion","generic","emocap","NLI","MNLI","DNLI",
                                 "TC_AG_NEWS","TC_SogouNews","TC_DBpedia","TC_YahooAnswers","empathetic_dialogue",
                                 "emotion","pun"),
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
    parser.add_argument("--bce_loss", action="store_true",help="binary cross entropy")
    parser.add_argument("--label", type=int, default=3, help="binary cross entropy")
    args = parser.parse_args()

    train_discriminator(**(vars(args)))


# def test_epoch(data_loader, discriminator, device='cuda', args=None):
#     discriminator.eval()
#     test_loss = 0
#     correct = 0
#     pred = []
#     gold = []
#     with torch.no_grad():
#         for data, target in data_loader:
#             data, target = data.to(device), target.to(device)
#             output = discriminator(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred_out = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred_out.eq(target.view_as(pred_out)).sum().item()
#             pred.append(output.detach().cpu().numpy())
#             gold.append(target.cpu().numpy())
#     test_loss /= len(data_loader.dataset)
#     pred = np.concatenate(pred)
#     gold = np.concatenate(gold)
#     accuracy, microPrecision, microRecall, microF1 = getMetrics(pred,gold,verbose=False)
#     print(accuracy, microPrecision, microRecall, microF1)
#     print('\nRelu Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), MicroF1 {}\n'.format(
#         test_loss, correct, len(data_loader.dataset),
#         100. * correct / len(data_loader.dataset), microF1))

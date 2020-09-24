from utils.helper import load_classifier, load_model, cut_seq_to_eos, parse_prefixes
from utils.helper import EOS_ID, find_ngrams, dist_score, truncate, pad_sequences, print_loss_matplotlib
from metric.bleu import moses_multi_bleu
from collections import Counter
import torch
import numpy as np


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def scorer(args,turn,classifier,enc,class2idx,knowledge,plot=False,gold=None):
    hypotesis = []
    plots_array = []
    if(plot):
        loss = np.transpose(np.array(turn['loss']), (2, 0, 1)) # batch * sequence_len * iteration
    for i,t in enumerate(turn['text']):
        ind_eos = len(cut_seq_to_eos(t))-1
        
        text = enc.decode(cut_seq_to_eos(t))
        dist = dist_score(text,enc)
        bleu = None
        f1 = None
        if(gold): 
            bleu = truncate(moses_multi_bleu(np.array([text]),np.array([gold]))*100,2)
            f1 = truncate(_prec_recall_f1_score(enc.encode(text), enc.encode(gold))*100,2)
        hypotesis.append([i,1-dist,text, f"{bleu}/{f1}"])
        ## plotting 
        if(plot): plots_array.append(loss[i][:ind_eos,-1])

    x = [h[2] for h in hypotesis]
    if(knowledge):
        sent_p = [knowledge for i in range(args.num_samples)]
        x = (sent_p,x)

    for j, (loss,correct,predition) in enumerate(zip(*predict(args,classifier,x,class2idx))):
        hypotesis[j] = [hypotesis[j][0],loss,hypotesis[j][1],correct,predition,hypotesis[j][3],hypotesis[j][2]]

    hypotesis = sorted(hypotesis, key = lambda x: x[1]) ## sort by loss
    acc = hypotesis[0][3] ## if it is correctly classifed the sample with the lowest loss
    hypotesis = [[h[0],truncate(h[1],4),truncate(h[2],4),h[4],h[5],h[6]] for h in hypotesis]
    return hypotesis, acc, plots_array

def predict(args, classifier, X, class2idx):
    if(type(X) is tuple):
        input_p = pad_sequences([torch.tensor(classifier.tokenizer.encode(s)) for s in X[0]]) 
        input_h = pad_sequences([torch.tensor(classifier.tokenizer.encode(s)) for s in X[1]]) 
        X = [input_p,input_h]
    else:
        X = pad_sequences([torch.tensor(classifier.tokenizer.encode(s)) for s in X]) 

    output_t = classifier(X)

    target_t = torch.tensor([args.label_class], device='cuda', dtype=torch.long).repeat(args.num_samples)
    ce_loss_logging = torch.nn.CrossEntropyLoss(reduction='none')
    loss = ce_loss_logging(output_t, target_t).detach().tolist()
    pred_t = output_t.argmax(dim=1, keepdim=True)
    correct = pred_t.eq(target_t.view_as(pred_t)).detach().tolist()
    return loss, sum(correct, []), [class2idx[int(pred[0])] for pred in pred_t.detach().tolist()]

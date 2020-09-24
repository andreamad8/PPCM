from transformers import GPT2Tokenizer,GPT2LMHeadModel
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
import torch
import numpy as np
import math

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to('cuda')
model.eval()

def get_ppl(text, starter):
    lls = []
    for idx, t in enumerate(text):
        input_ids = torch.tensor(tokenizer.encode(" ".join(starter[idx]['conversation'])+ " "+t)).unsqueeze(0)  # Batch size 1
        input_ids = input_ids.to('cuda')
        if input_ids.size(1)>1:
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            loss, _ = outputs[:2]
            lls.append(loss.item())
    
    return math.exp(np.mean(lls))

def get_ppl_simplified(text, starter):
    lls = []
    for idx, t in enumerate(text):
        input_ids = torch.tensor(tokenizer.encode(" ".join(starter[idx])+ " "+t)).unsqueeze(0)  # Batch size 1
        input_ids = input_ids.to('cuda')
        if input_ids.size(1)>1:
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            loss, _ = outputs[:2]
            lls.append(loss.item())
    
    return math.exp(np.mean(lls))
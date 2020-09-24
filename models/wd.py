#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import os
import sys
import argparse
from tqdm import trange

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from IPython import embed
import pdb
from operator import add
import pickle
import csv
import colorama
from collections import defaultdict
from models.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Config
from transformers import GPT2Tokenizer
from utils.helper import EOS_ID

SmallConst = 1e-15


def top_k_logits(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def get_greedy(args=None, true_hidden=None, model=None, output=None, classifier=None, good_index=None):
    
    if args.bag_of_words:
        logits, true_past = model(output)
        log_probs = F.softmax(logits[:, -1, :], dim=-1)
        one_hot_vectors = []
        for good_list in good_index:
            good_list = list(filter(lambda x: len(x) <= 1, good_list))
            good_list = torch.tensor(good_list).cuda()
            num_good = good_list.shape[0]
            one_hot_good = torch.zeros(num_good, vocab_size).cuda()
            one_hot_good.scatter_(1, good_list, 1)
            one_hot_good = torch.sum(one_hot_good, dim=0)
        one_hot_vectors.append(one_hot_good)

        log_probs = log_probs + args.bow_scale_weight*one_hot_vectors[-1]*log_probs #+ args.bow_scale_weight*one_hot_vectors[-1]
        log_probs = top_k_logits(log_probs, k=args.top_k, probs=True)
        log_probs = log_probs/torch.sum(log_probs)
        # print(log_probs.shape)
        prev = torch.multinomial(log_probs, num_samples=1)
        # print(prev)
    else:
        logits, true_past = model(output)
        # print("current sentence len",output.size())
        # print("true past size",true_past[0].size())
        log_probs = F.softmax(logits[:, -1, :], dim=-1)
        log_probs = top_k_logits(log_probs, k=args.num_iterations, probs=True)
        [_, vocab_size] = log_probs.shape

        idx = torch.nonzero(log_probs).split(1, dim=1)[1]
        # print("non zero index",idx.size())
        # print("non zero index",idx)
        hidden = model.hidden_states
        
        accumulated_hidden = torch.sum(hidden, dim=1)

        [_, _, _, current_length, _] = true_past[0].shape

        batch_size = idx.size(0) # args.num_iterations 
        true_past = [i.repeat(1, batch_size, 1, 1, 1) for i in true_past]
        # print(true_past[0].size())
        # true_past = [jj[:, idx, :, :, :] for jj in true_past]
        # print("true bast after reshaping",true_past[0].size())
        logits, _ = model(idx, past=true_past)
        # print("logits for each candidate",logits.size())
        hidden = model.hidden_states
        # print("hidden for each candidate",hidden.size())
        
        accumulated_mean_hidden = (torch.sum(hidden, dim=1) + accumulated_hidden)/(current_length + 1)
        # print("accumulated hidden for each candidate",accumulated_mean_hidden.size())
        attribute_logits = classifier(accumulated_mean_hidden)
        # print("logit for each candidate",attribute_logits.size())
        attribute_probs = F.softmax(attribute_logits, dim=-1)
        # print("Score for each candidate",attribute_logits)

        vocab_attribute_probs = attribute_probs[:, args.label_class]
        # print("lable class score",vocab_attribute_probs.size())
        weight = torch.zeros(log_probs.size()).to(log_probs.device)
        # print("weight distrib",weight.size())

        weight = weight.scatter_(1,idx.squeeze().unsqueeze(0),vocab_attribute_probs.unsqueeze(0))
        # print("weight scattered distrib",weight.size())
        # print("weight non zero element",torch.nonzero(weight).split(1, dim=1)[1])
        rescaled_probs = log_probs+weight
        # print(rescaled_probs.size())
        
        topk_rescaled_probs = top_k_logits(rescaled_probs, k=args.top_k, probs=True)
        topk_rescaled_probs = topk_rescaled_probs/torch.sum(topk_rescaled_probs, dim=-1)
        prev = torch.multinomial(topk_rescaled_probs, num_samples=1) 
  
    return prev


def weight_decoder(model, enc, args, context=None, sample=True, device='cuda',repetition_penalty=1.0,classifier=None,knowledge=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get tokens for the list of positive words
    def list_tokens(word_list,enc):
        token_list = []
        for word in word_list:
            token_list.append(enc.encode(" " + word))
        return token_list

    good_index = []
    actual_words = None
    if args.bag_of_words:
        bags_of_words = args.bag_of_words.split(";")
        for wordlist in bags_of_words:
            with open(wordlist, "r") as f:
                words = f.read().strip()
                words = words.split('\n')
            good_index.append(list_tokens(words,enc))
            
        # useless for the process
        for good_list in good_index:
            good_list = list(filter(lambda x: len(x) <= 1, good_list))
            actual_words = [(enc.decode(ww).strip(),ww) for ww in good_list]
    knowledge_to_ent = None
    if args.bag_of_words and classifier:
        args.loss_type = 3
    elif args.bag_of_words:
        args.loss_type = 1
    elif 'NLI' in args.discrim:
        knowledge_to_ent = enc.encode(knowledge)
        args.loss_type = 4
    elif classifier is not None:
        args.loss_type = 2
    else:
        raise Exception('Supply either --bag-of-words (-B) or --discrim -D')


    seed_list = [args.seed+i for i in range(args.num_samples)]
    list_output = []
    for i in range(args.num_samples):
        torch.cuda.empty_cache()
        torch.manual_seed(seed_list[i])
        torch.cuda.manual_seed(seed_list[i])
        np.random.seed(seed_list[i])
        perturbed, _, loss_in_time = sample_from_hidden(model=model,args=args, context=context,
                                                        device=device, perturb=False, good_index=good_index,
                                                        classifier=classifier, repetition_penalty=repetition_penalty)
        list_output.append(perturbed)

    ## padding
    perturbed = torch.stack([torch.cat((p,torch.zeros((1,args.length - p.shape[1]),dtype=int).to(p.device)),dim=1)  for p in list_output]).squeeze(1)

    return perturbed, perturbed, None, loss_in_time, actual_words


def sample_from_hidden(model, args, classifier, context=None, past=None, device='cuda',
                       sample=True, perturb=True, good_index=None, repetition_penalty=1.0,
                       knowledge_to_ent=None):
    output = torch.tensor(context, device=device, dtype=torch.long) if context else None
    output_response = output.new_zeros([output.size(0),0])
    loss_in_time = []
    loss_in_time_true_loss = []
    stopped = [0 for _ in range(output.size(0))]
    for i in range(args.length):#, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current-token
        # Therefore, use everything from before current i/p token to generate relevant past

        # if past is None and output is not None:
        #     prev = output[:, -1:]
        #     _, past = model(output[:, :-1])
        #     original_probs, true_past = model(output)
        #     true_hidden = model.hidden_states

        # else:
        #     original_probs, true_past = model(output)
        #     true_hidden = model.hidden_states


        prev = get_greedy(args=args, model=model, output=output, classifier=classifier, good_index=good_index)

        output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
        output_response = torch.cat((output_response, prev), dim=1)

        for i_p, p in enumerate(prev.tolist()):
            if(p[0]) == EOS_ID:
                stopped[i_p] = 1

        if(all(x == 1 for x in stopped)): break

    return output_response, loss_in_time_true_loss, loss_in_time


    # else:
    #     logits, true_past = model(output)
    #     log_probs = F.softmax(logits[:, -1, :], dim=-1)
    #     hidden = model.hidden_states
        
    #     accumulated_hidden = torch.sum(hidden, dim=1)

    #     [_, _, _, current_length, _] = true_past[0].shape

    #     batch_size = 200 
    #     true_past = [i.repeat(1, batch_size, 1, 1, 1) for i in true_past]

    #     num_batches = vocab_size//batch_size + 1
    #     vocab_attribute_probs = None

    #     for i in range(num_batches):
    #         if i < num_batches - 1:
    #             logits, _ = model(torch.unsqueeze(torch.tensor(range(i*batch_size, (i+1)*batch_size)), dim=1).cuda(), 
    #                                         past=true_past)
    #         else:
    #             true_past = [jj[:, :(vocab_size - i*batch_size), :, :, :] for jj in true_past]
    #             logits, _ = model(torch.unsqueeze(torch.tensor(range(i*batch_size, vocab_size)), dim=1).cuda(),
    #                     past=true_past)
            
    #         hidden = model.hidden_states
            
    #         accumulated_mean_hidden = (torch.sum(hidden, dim=1) + accumulated_hidden)/(current_length + 1)
    #         attribute_logits = classifier(accumulated_mean_hidden)
    #         attribute_probs = F.softmax(attribute_logits, dim=-1)
    #         if i == 0:
    #             vocab_attribute_probs = attribute_probs[:, args.label_class]
    #         else:
    #             vocab_attribute_probs = torch.cat((vocab_attribute_probs, attribute_probs[:, args.label_class]), 0)
       
    #     rescaled_probs = log_probs*torch.unsqueeze(vocab_attribute_probs, dim=0)
        
        
    #     topk_rescaled_probs = top_k_logits(rescaled_probs, k=args.top_k, probs=True)
    #     topk_rescaled_probs = topk_rescaled_probs/torch.sum(topk_rescaled_probs, dim=-1)
    #     prev = torch.multinomial(topk_rescaled_probs, num_samples=1) 
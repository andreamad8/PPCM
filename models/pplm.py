#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import os
import sys
import argparse
from tqdm import trange
import time
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

def perturb_past(past, model, prev, args, classifier, good_index=None, stepsize=0.01, vocab_size=50257,
                 original_probs=None, accumulated_hidden=None, current_output=None, true_past=None, grad_norms=None,
                 knowledge_to_ent=None):
    window_length = args.window_length
    gm_scale, kl_scale = args.gm_scale, args.kl_scale
    one_hot_vectors = []
    for good_list in good_index:
        good_list = list(filter(lambda x: len(x) <= 1, good_list))
        good_list = torch.tensor(good_list).cuda()
        num_good = good_list.shape[0]
        one_hot_good = torch.zeros(num_good, vocab_size).cuda()
        one_hot_good.scatter_(1, good_list, 1)
        one_hot_vectors.append(one_hot_good)


    # Generate inital perturbed past
    past_perturb_orig = [(np.random.uniform(0.0, 0.0, p.shape).astype('float32'))
                         for p in past]
    
    
    if accumulated_hidden is None:
        accumulated_hidden = 0

    if args.decay:
        decay_mask = torch.arange(0., 1.0 + SmallConst, 1.0/(window_length))[1:]
    else:
        decay_mask = 1.0

    # Generate a mask is gradient perturbated is based on a past window
    _, batch_size, _, current_length, _ = past[0].shape
    if current_length > window_length and window_length > 0:
        ones_key_val_shape = tuple(past[0].shape[:-2]) + tuple([window_length]) + tuple(
            past[0].shape[-1:])

        zeros_key_val_shape = tuple(past[0].shape[:-2]) + tuple([current_length - window_length]) + tuple(
            past[0].shape[-1:])

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask*ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat((ones_mask, torch.zeros(zeros_key_val_shape)), dim=-2).cuda()
    else:
        window_mask = torch.ones_like(past[0]).cuda()

    loss_per_iter = []
    loss_barrier = 1.0
    for current_iter in range(args.num_iterations):
        # print("Iteration ", i + 1)
        past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
        past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]

        perturbed_past = list(map(add, past, past_perturb))

        _, _, _, current_length, _ = past_perturb[0].shape

        # Compute hidden using perturbed past
        logits, future_past = model(prev, past=perturbed_past)
        hidden = model.hidden_states
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()

        # TODO: Check the layer-norm consistency of this with trained discriminator
        logits = logits[:, -1, :]
        probabs = F.softmax(logits, dim=-1)
        loss = 0.0

        ## BOW
        if args.loss_type == 1 or args.loss_type == 3:

            for one_hot_good in one_hot_vectors:
                good_logits = torch.mm(probabs, torch.t(one_hot_good))
                loss_word = good_logits
                loss_word = torch.sum(loss_word, dim=1)
                loss_word = -torch.log(loss_word)
                loss += loss_word.sum()
            loss_per_iter.append(loss_word.detach().tolist())

        ## DISCRIMINATOR
        if args.loss_type == 2 or args.loss_type == 3:
            ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')
            new_true_past = true_past
            for i in range(args.horizon_length):

                future_probabs = F.softmax(logits, dim=-1)  # Get softmax
                future_probabs = torch.unsqueeze(future_probabs, dim=1)

                _, new_true_past = model(future_probabs, past=new_true_past)
                future_hidden = model.hidden_states  # Get expected hidden states
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(future_hidden, dim=1)

            predicted_sentiment = classifier(new_accumulated_hidden / (current_length + 1 + args.horizon_length))

            label = torch.tensor([args.label_class], device='cuda', dtype=torch.long).repeat(batch_size)
            discrim_loss = ce_loss(predicted_sentiment, label)
            loss += discrim_loss

            ## LOGGING 
            ce_loss_logging = torch.nn.CrossEntropyLoss(reduction='none')
            loss_logging = ce_loss_logging(predicted_sentiment, label).detach().tolist()
            loss_per_iter.append(loss_logging)

        if args.loss_type == 4: # Enteiltment loss
            _ = model(torch.tensor([knowledge_to_ent], device='cuda', dtype=torch.long))
            hidden_p = model.hidden_states.repeat(batch_size,1,1) #torch.mean(model.hidden_states,dim=1).repeat(batch_size,1) 

            ce_loss = torch.nn.CrossEntropyLoss(reduction='sum')
            new_true_past = true_past
            for i in range(args.horizon_length):

                future_probabs = F.softmax(logits, dim=-1)  # Get softmax
                future_probabs = torch.unsqueeze(future_probabs, dim=1)

                _, new_true_past = model(future_probabs, past=new_true_past)
                future_hidden = model.hidden_states  # Get expected hidden states
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(future_hidden, dim=1)

            if current_output.size(1)!=0: 
                hidden_h = torch.cat((current_output,future_hidden), dim=1)
            else:
                hidden_h = future_hidden

            predicted_NLI = classifier(hidden_p,hidden_h)

            label = torch.tensor([args.label_class], device='cuda', dtype=torch.long).repeat(batch_size)
            discrim_loss = ce_loss(predicted_NLI, label)
            loss += discrim_loss

            ## LOGGING 
            ce_loss_logging = torch.nn.CrossEntropyLoss(reduction='none')
            loss_per_iter.append(ce_loss_logging(predicted_NLI, label).detach().tolist())

        if args.loss_type == 5:
            bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
            new_true_past = true_past
            for i in range(args.horizon_length):

                future_probabs = F.softmax(logits, dim=-1)  # Get softmax
                future_probabs = torch.unsqueeze(future_probabs, dim=1)

                _, new_true_past = model(future_probabs, past=new_true_past)
                future_hidden = model.hidden_states  # Get expected hidden states
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(future_hidden, dim=1)

            predicted_sentiment = classifier(new_accumulated_hidden / (current_length + 1 + args.horizon_length))

            label = torch.tensor([1], device='cuda', dtype=torch.float).repeat(batch_size)
            discrim_loss = bce_loss(predicted_sentiment, label.unsqueeze(-1))
            loss += discrim_loss

            ## LOGGING 
            bce_loss_logging = torch.nn.BCEWithLogitsLoss(reduction='none')
            loss_per_iter.append(bce_loss_logging(predicted_sentiment, label.unsqueeze(-1)).detach().tolist())

        kl_loss = 0.0
        if kl_scale > 0.0:
            p = (F.softmax(original_probs[:, -1, :], dim=-1))
            p = p + SmallConst * (p <= SmallConst).type(torch.FloatTensor).cuda().detach()
            correction = SmallConst * (probabs <= SmallConst).type(torch.FloatTensor).cuda().detach()
            corrected_probabs = probabs + correction.detach()
            kl_loss = kl_scale * ((corrected_probabs * (corrected_probabs / p).log()).sum())
         
            ## TODO
            # print(' kl_loss', (kl_loss).data.cpu().numpy())
            loss += kl_loss  # + discrim_loss
        
        ## TODO
        # print(f'pplm_loss {current_iter}', ((loss/batch_size) - kl_loss).data.cpu().numpy())
        # print(f'pplm_min_loss {current_iter}', min(loss_logging))
        # print()

        loss.backward(retain_graph=True)
        if grad_norms is not None and args.loss_type == 1:
            grad_norms = [torch.max(grad_norms[index], 
                torch.norm_except_dim(p_.grad * window_mask, dim=1)) 
                    for index, p_ in enumerate(past_perturb)]
        else:
            grad_norms = [(torch.norm_except_dim(p_.grad * window_mask, dim=1) + SmallConst) for index, p_ in enumerate(past_perturb)]

        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[index] ** args.gamma).data.cpu().numpy()
            for index, p_ in enumerate(past_perturb)]
        past_perturb_orig = list(map(add, grad, past_perturb_orig))

        for p_ in past_perturb:
            p_.grad.data.zero_()

        new_past = []
        for p in past:
            new_past.append(p.detach())

        past = new_past

    past_perturb = [torch.from_numpy(p_) for p_ in past_perturb_orig]
    past_perturb = [to_var(p_, requires_grad=True) for p_ in past_perturb]
    perturbed_past = list(map(add, past, past_perturb))

    return perturbed_past, new_accumulated_hidden, grad_norms, loss_per_iter


def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

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

def latent_perturb(model, enc, args, context=None, sample=True, device='cuda',repetition_penalty=1.0,classifier=None,knowledge=None):
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
    elif args.BCE and classifier is not None:
        args.loss_type = 5
    elif classifier is not None:
        args.loss_type = 2
    else:
        raise Exception('Supply either --bag-of-words (-B) or --discrim -D')


    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    original, _, _ = sample_from_hidden(model=model,args=args, context=context, device=device,
                                perturb=False, good_index=good_index, classifier=classifier, repetition_penalty=repetition_penalty,
                                knowledge_to_ent=knowledge_to_ent)
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    perturbed, _, loss_in_time = sample_from_hidden(model=model, args=args, context=context,
                                                        device=device, perturb=True, good_index=good_index,
                                                        classifier=classifier, repetition_penalty=repetition_penalty,
                                                        knowledge_to_ent=knowledge_to_ent)
    t1 = time.time()
    print("time",t1-t0)
    torch.cuda.empty_cache()        
    return original, perturbed, None, loss_in_time, actual_words


def sample_from_hidden(model, args, classifier, context=None, past=None, device='cuda',
                       sample=True, perturb=True, good_index=None, repetition_penalty=1.0,
                       knowledge_to_ent=None):
    output = torch.tensor(context, device=device, dtype=torch.long) if context else None
    output_response = output.new_zeros([output.size(0),0])
    grad_norms = None
    loss_in_time = []
    loss_in_time_true_loss = []
    stopped = [0 for _ in range(output.size(0))]
    for i in range(args.length):#, ascii=True):

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current-token
        # Therefore, use everything from before current i/p token to generate relevant past

        if past is None and output is not None:
            prev = output[:, -1:]
            _, past = model(output[:, :-1])
            original_probs, true_past = model(output)
            true_hidden = model.hidden_states

        else:
            original_probs, true_past = model(output)
            true_hidden = model.hidden_states

        # Modify the past if necessary

        if i >= args.grad_length:
            current_stepsize = args.stepsize * 0
        else:
            current_stepsize = args.stepsize

        if not perturb or args.num_iterations == 0:
            perturbed_past = past

        else:
            accumulated_hidden = model.hidden_states[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)
            len_prefix = true_hidden.size(1)
            perturbed_past, _, grad_norms, loss_per_iter = perturb_past(past, model, prev, args,
                                                                        good_index=good_index, stepsize=current_stepsize,
                                                                        original_probs=original_probs,
                                                                        true_past=true_past,
                                                                        accumulated_hidden=accumulated_hidden,
                                                                        current_output=true_hidden[:,len_prefix-i:,:],
                                                                        classifier=classifier,
                                                                        grad_norms=grad_norms, 
                                                                        knowledge_to_ent=knowledge_to_ent)
            loss_in_time.append(loss_per_iter)

        logits, past = model(prev, past=perturbed_past)

        hidden = model.hidden_states  # update hidden
        logits = logits[:, -1, :] / args.temperature  # + SmallConst

        for i_o, o_ in enumerate(output):
            for token_idx in set(o_.tolist()):
                if logits[i_o, token_idx] < 0:
                    logits[i_o, token_idx] *= repetition_penalty
                else:
                    logits[i_o, token_idx] /= repetition_penalty

        log_probs = F.softmax(logits, dim=-1)

        # Fuse the modified model and original model
        if perturb:
            original_probs = F.softmax(original_probs[:, -1, :], dim=-1)
            gm_scale = args.gm_scale
            log_probs = ((log_probs ** gm_scale) * (original_probs ** (1 - gm_scale)))  # + SmallConst

            log_probs = top_k_logits(log_probs, k=args.top_k, probs=True)  # + SmallConst

            if torch.sum(log_probs) <= 1:
                log_probs = log_probs / torch.sum(log_probs)
        else:
            logits = top_k_logits(logits, k=args.top_k)  # + SmallConst
            log_probs = F.softmax(logits, dim=-1)

        if sample:
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            _, prev = torch.topk(log_probs, k=1, dim=-1)

        output = prev if output is None else torch.cat((output, prev), dim=1)  # update output
        output_response = torch.cat((output_response, prev), dim=1)

        for i_p, p in enumerate(prev.tolist()):
            if(p[0]) == EOS_ID:
                stopped[i_p] = 1

        if(all(x == 1 for x in stopped)): break

    return output_response, loss_in_time_true_loss, loss_in_time

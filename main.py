import os
import argparse
import torch
import numpy as np
from transformers import GPT2Tokenizer

from interact_adapter import interact
from utils.helper import load_classifier, load_model, load_model_recursive
from evaluate import evaluate

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default="medium", help='Size of dialoGPT')
    parser.add_argument('--model_path', '-M', type=str, default='gpt-2_pt_models/dialoGPT/',
                        help='pretrained model name or path to local checkpoint')
    parser.add_argument('--discrim', '-D', type=str, default=None, 
                        choices=('sentiment',"daily_dialogue_act",
                                 "AG_NEWS"), 
                        help='Discriminator to use for loss-type 2')
    parser.add_argument('--label_class', type=int, default=-1, help='Class label used for the discriminator')
    parser.add_argument('--stepsize', type=float, default=0.03)
    parser.add_argument('--num_iterations', type=int, default=2)
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5555)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.1) #1.1
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--gm_scale", type=float, default=0.95)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument('--nocuda', action='store_true', help='no cuda')
    parser.add_argument('--grad_length', type=int, default=10000)
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate from the modified latents')
    parser.add_argument('--horizon_length', type=int, default=1, help='Length of future to optimize over')
    # parser.add_argument('--force-token', action='store_true', help='no cuda')
    parser.add_argument('--window_length', type=int, default=0,
                        help='Length of past which is being optimizer; 0 corresponds to infinite window length')
    parser.add_argument('--decay', action='store_true', help='whether to decay or not')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument("--max_history", type=int, default=-1)
    parser.add_argument('--evaluate', action='store_true', help='evaluate')
    parser.add_argument('--wd', action='store_true', help='greedy based on rev comments')
    parser.add_argument('--verbose', action='store_true', help='verbose mode, no comet print in the terminal')
    parser.add_argument('--bow_scale_weight', type=float, default=20)
    parser.add_argument('--sample_starter', type=int, default=0)
    parser.add_argument('--all_starter', action='store_true', help='selfchat')
    parser.add_argument("--speaker", type=str, default="PPLM")
    parser.add_argument("--task_ent", type=str, default="data/simple_QA/QA.json")
    parser.add_argument("--load_check_point_adapter", type=str, default="None")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--trial_id", type=int, default=1)
    parser.add_argument("--entailment", type=bool, default=False)
    parser.add_argument("--BCE", type=bool, default=False)
    parser.add_argument("--bag_of_words", type=str, default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if(args.load_check_point_adapter != "None"):
        print("LOADING ADAPTER CONFIG FILE AND INTERACTIVE SCRIPT")
        from models.pytorch_pretrained_bert.modeling_adapter import GPT2LMHeadModel, GPT2Config
    else:
        from models.pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Config

    device = 'cpu' if args.nocuda else 'cuda'
    args.model_path = f'models/dialoGPT/{args.model_size}/'
    config = GPT2Config.from_json_file(os.path.join(args.model_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    if(args.load_check_point_adapter != "None"):
        print("Loading ADAPTERS")
        model = load_model_recursive(GPT2LMHeadModel(config,default_task_id=args.task_id), args.load_check_point_adapter, args, verbose=True)
    else:
        model = load_model(GPT2LMHeadModel(config), args.model_path+f"{args.model_size}_ft.pkl", args, verbose=True)
    model.to(device).eval()

    # Freeze Models weights
    for param in model.parameters():
        param.requires_grad = False

    classifier, class2idx = load_classifier(args, model)

    logger = None

    ## set iter to 0 to run the adapter 
    ## set iter to 50 to run WD
    param_grid = {'iter': [75], 'window': [0], 'steps': [0.02]}

    if(args.evaluate):
        evaluate(args,model,tokenizer,classifier,args.entailment,args.task_ent,class2idx,param_grid,device,logger)
    else:
        interact(args, model, tokenizer, classifier, class2idx, device)

if __name__ == '__main__':
    run_model()

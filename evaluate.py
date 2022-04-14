from tabulate import tabulate
tabulate.PRESERVE_WHITESPACE = True
from utils.helper import cut_seq_to_eos, parse_prefixes, get_name
from utils.helper import EOS_ID, print_loss_matplotlib
from utils.utils_sample import scorer
from sklearn.model_selection import ParameterGrid
from models.pplm import latent_perturb
from models.wd import weight_decoder
import os
import numpy as np
import jsonlines

def make_header(args,id_starter,knowledge):
    str_title = ""
    str_title += "===================================================\n"
    str_title += f"Model={args.model_size} Window={args.window_length} Iteration={args.num_iterations} Step_size={args.stepsize}\n"
    str_title += "===================================================\n"
    name_experiment = f"Iter={args.num_iterations}_Step={args.stepsize}_Start={id_starter}_W={args.window_length}"
    if(knowledge):
        str_title += f"Knowledge={knowledge}\n"
        str_title += "===================================================\n"
        knol = knowledge.replace(" ","_")
        name_experiment = f"Iter={args.num_iterations}_Know={knol}_Step={args.stepsize}_Start={id_starter}_W={args.window_length}"
    return str_title, name_experiment

def logger_conv_ent(args,conv,enc,id_starter,logger,class2idx,classifier,knowledge=None,gold=None):
    str_title, name_experiment = make_header(args,id_starter,knowledge)
    acc_original = []
    acc_pplm = []
    for turn in conv:
        if(turn['speaker']=="PPLM"):
            str_title += "===================================================\n" 
            str_title += "PPLM\n" 
            str_title += "===================================================\n" 
            hypotesis, acc_pplm, plots_array = scorer(args,turn,classifier,enc,class2idx,knowledge,plot=False,gold=gold)
            str_title += tabulate(hypotesis, headers=['Id', 'Loss','Dist','Label', 'BLEU/F1','Text'], tablefmt='simple',floatfmt=".2f",colalign=("center","center","center","center","left"))
            str_title += "\n"
            if(args.verbose):
                print(str_title)
            else:
                print_loss_matplotlib(plots_array,loss_original,str_title,logger,name=name_experiment)
        elif(turn['speaker']=="DGPT"):
            str_title += "===================================================\n" 
            str_title += "DGPT\n" 
            str_title += "===================================================\n" 
            if(not args.bag_of_words):  
                hypotesis_original, acc_original, _ = scorer(args,turn,classifier,enc,class2idx,knowledge,gold=gold)
                str_title += tabulate(hypotesis_original, headers=['Id','Loss','Dist','Label', 'BLEU/F1','Text'], tablefmt='simple',floatfmt=".2f",colalign=("center","center","center","center","left"))
                str_title += "\n"
                loss_original = hypotesis_original[0][1]                 
            else:
                hypotesis_original = [[i, enc.decode(cut_seq_to_eos(t))] for i, t in enumerate(turn['text'])]
                str_title += tabulate(hypotesis_original, headers=['Id','Text'], tablefmt='orgtbl')
                loss_original = 0
            str_title += "===================================================\n"
        else: ## human case
            str_title += f"{turn['speaker']} >>> {turn['text']}\n" 
            loss_original = 0

    return acc_pplm, acc_original, hypotesis, hypotesis_original


def evaluate(args,model,enc,classifier,entailment,task_ent,class2idx,param_grid,device,logger):
    if(entailment):
        list_starters = parse_prefixes(args,entailment=True,task=task_ent)
    else:
        list_starters = parse_prefixes(args,tokenizer=enc,seed=args.seed)
    for param in list(ParameterGrid(param_grid)):
        args.stepsize = param["steps"]
        args.num_iterations = param["iter"]
        args.window_length = param["window"]
        print("===================================================")
        print(f"Model={args.model_size} Discrim={args.discrim} Window={args.window_length} Iteration={args.num_iterations} Step_size={args.stepsize}")
        print("===================================================")
        global_acc_original, global_acc_PPLM = [], []
        lab = class2idx[args.label_class].replace(" ","_").replace("/","")
        base_path = f"results/evaluate/{args.discrim}_class_{lab}/"
        name = get_name(args,base_path,class2idx)
        mode = 'w'
        if os.path.exists(name):
            num_lines = sum(1 for line in open(name,'r'))
            list_starters = list_starters[num_lines:]
            mode = 'a'
        with jsonlines.open(get_name(args,base_path,class2idx), mode=mode) as writer:
            for id_starter, starter in enumerate(list_starters):
                conversation = []
                for t in starter["conversation"]:
                    conversation.append({"speaker":"human", "text":t})
                
                history = starter["conversation"]
                context_tokens = sum([enc.encode(h) + [EOS_ID] for h in history],[]) 

                if(args.wd):
                    context_tokens = [context_tokens]
                    original_sentence, perturb_sentence, _, loss, _ = weight_decoder(model=model, enc=enc, 
                                                                                    args=args, context=context_tokens,
                                                                                    device=device,repetition_penalty=args.repetition_penalty,
                                                                                    classifier=classifier.classifier_head,knowledge=starter["knowledge"])
                else:
                    context_tokens = [context_tokens for _ in range(args.num_samples)]
                    original_sentence, perturb_sentence, _, loss, _ = latent_perturb(model=model, enc=enc, 
                                                                                    args=args, context=context_tokens,
                                                                                    device=device,repetition_penalty=args.repetition_penalty,
                                                                                    classifier=classifier.classifier_head,knowledge=starter["knowledge"])
                conversation.append({"speaker":"DGPT","text":original_sentence.tolist()})
                conversation.append({"speaker":"PPLM","text":perturb_sentence.tolist(),"loss":loss})
                acc_pplm, acc_original, hypotesis, hypotesis_original = logger_conv_ent(args,conversation,enc,id_starter,logger,class2idx=class2idx,classifier=classifier,knowledge=starter["knowledge"],gold=starter["gold"])
                global_acc_PPLM.append(acc_pplm)
                global_acc_original.append(acc_original)
                writer.write({"acc":{"DGPT":acc_original,"PPLM":acc_pplm}, "hyp":{"DGPT":hypotesis_original,"PPLM":hypotesis},"conversation":starter})


        print(f"Global Acc original:{np.mean(global_acc_original)} Acc PPLM:{np.mean(global_acc_PPLM)}")
        print()
        print()
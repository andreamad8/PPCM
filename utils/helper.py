import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
import matplotlib.pyplot as plt
from models.heads import AttentionHead, ClassificationHead, Discriminator
import os
import torch
import torch.nn as nn
import random
import math
import json
from datetime import datetime
EOS_ID = 50256
import logging
logger = logging.getLogger(__name__)

def load_classifier(args,model):
    print(f"Loading Classifier {args.discrim}")
    classifier = None
    class2idx = None
    if args.discrim == 'sentiment':
        idx2class = ["positive", "negative", "very positive", "very negative",
                     "neutral"]
        class2idx = {c: i for i, c in enumerate(idx2class)}
        classes_num = len(idx2class)
        models_weight = "models/discriminators/DIALOGPT_sentiment_classifier_head_epoch_10.pt"

        classifier = Discriminator(
            class_size=classes_num,
            pretrained_model="medium",
            cached_mode=False,
            load_weight=models_weight,
            model_pretrained=model
        ).to("cuda")
        classifier.eval()

    elif args.discrim == 'daily_dialogue_act':
        idx2class = ["inform", "question", "directive", "commissive"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        classifier = Discriminator(
            class_size=len(idx2class),
            pretrained_model="medium",
            cached_mode=False,
            load_weight="models/discriminators/DIALOGPT_daily_dialogue_act.pt",
            model_pretrained=model
        ).to("cuda")
        classifier.eval()

    elif args.discrim == "AG_NEWS":
        idx2class = ["World","Sports","Business","Sci/Tech"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        classifier = Discriminator(
            class_size=len(idx2class),
            pretrained_model="medium",
            cached_mode=False,
            load_weight="models/discriminators/DIALOGPT_TC_AG_NEWS_classifier_head.pt",
            model_pretrained=model
        ).to("cuda")
        classifier.eval()


  

    class2idx = {i: c for i, c in enumerate(idx2class)}
    return classifier, class2idx



def load_model(model, checkpoint, args, verbose=False):
    if checkpoint is None or checkpoint == "None":
        if verbose:
            print('No checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            print('Loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model, "transformer")
            and all(not s.startswith('transformer.')
                    for s in model_state_dict.keys())):
            print('Loading transfomer only')
            start_model = model.transformer
        start_model.load_state_dict(model_state_dict)

    return model


def load_model_recursive(model, checkpoint, args, verbose=False):
    if checkpoint is None or checkpoint == "None":
        if verbose:
            print('No checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            print('Loading finetuned model from %s' % checkpoint)
        model_state_dict = torch.load(checkpoint)

        model_state_dict = fix_state_dict_namespace(model_state_dict)

        start_model = model
        if (hasattr(model, "transformer")
            and all(not s.startswith('transformer.')
                    for s in model_state_dict.keys())):
            print('Loading transfomer only')
            start_model = model.transformer
        

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        def load(module: nn.Module, prefix=""):
            local_metadata = {}
            module._load_from_state_dict(
                model_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(start_model)

        if model.__class__.__name__ != start_model.__class__.__name__:
            base_model_state_dict = start_model.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
            ]

            missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        #model.tie_weights()
    return model


def fix_state_dict_namespace(model_state_dict):
    old_keys = []
    new_keys = []
    for t in model_state_dict:
        new_key = t
        if t.startswith('module.'):
            new_key = t.replace('module.', '')
        old_keys.append(t)
        new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict

def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for id_t, s in enumerate(sentence):
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent

def find_ngrams(input_list, n):
  return len(set(zip(*[input_list[i:] for i in range(n)])))

def dist_score(s,enc):
    if(len(enc.encode(s))>0): return find_ngrams(enc.encode(s), n=2)/len(enc.encode(s))
    else: return 0
    
def truncate(f, n):
    if math.isnan(f):
        return f
    return math.floor(f * 10 ** n) / 10 ** n

def pad_sequences(sequences):
    lengths = [len(seq) for seq in sequences]

    padded_sequences = torch.zeros(
        len(sequences),
        max(lengths)
    ).long()  # padding value = 0

    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_sequences[i, :end] = seq[:end]

    return padded_sequences

def print_loss_matplotlib(plots_array,loss_original,str_title,logger,name):
    plt.figure(figsize=(20,10))
    plt.axhline(y=loss_original, label="Original", linestyle='--')
    for ind, p in enumerate(plots_array):
        plt.plot(p, label=f"Sample{str(ind)}")
    plt.xlabel('Tokens Len')
    plt.ylabel('Loss')
    plt.title(str_title,loc='left')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    logger.log_figure(figure=plt,figure_name=name)
    plt.close()

def parse_prefixes(args, tokenizer=None, entailment=False, seed=1234, task='data/simple_QA/QA.json'):
    list_starters = []
    if(entailment):
        cnt = 0
        with open(task) as json_file:
            data = json.load(json_file)
            for p in random.sample(data, args.sample_starter):
                list_starters.append(p)
                if(cnt>=args.sample_starter): break
                cnt += 1
    else:
        # https://github.com/google-research/google-research/blob/master/meena/human.txt
        with open("data/human.txt") as f:
            data = []
            conversation = []
            for d in f:
                if len(d)>5:
                    if("Human Conversation" in d):
                        if (len(conversation)>0):
                            data.append(conversation)
                        conversation = []
                        i = 0
                    else:
                        if("Human 1: " in d):
                            _, text_turn = d.split("Human 1: ")
                            conversation.append({"turn":i,"speaker":"Human 1","text":text_turn.strip('\n').strip()})
                        else:
                            _, text_turn = d.split("Human 2: ")
                            conversation.append({"turn":i,"speaker":"Human 2","text":text_turn.strip('\n').strip()})
                        i += 1
        
        random.seed(seed)
        if(args.all_starter): 
            conversation = data
            for i_c,conv in enumerate(conversation):
                for index in range(len(conv)-2):
                    history = [conv[index]["text"],conv[index+1]["text"]]
                    context_tokens = len(sum([tokenizer.encode(h) + [1111] for h in history],[]))
                    if(context_tokens <= 70):
                        list_starters.append({"conversation":[conv[index]["text"],conv[index+1]["text"]],"knowledge":None,"gold":None})
                    # else:
                    #     print(f"Skipped len {context_tokens} index=({i_c},{index})")
            if(args.sample_starter!=0):
                list_starters = list_starters[:args.sample_starter+1]
        else:
            conversation = random.sample(data, args.sample_starter)
            for conv in conversation:
                index = random.randint(0,len(conv)-2)
                list_starters.append({"conversation":[conv[index]["text"],conv[index+1]["text"]],"knowledge":None,"gold":None})
    return list_starters

def get_name(args,base_path,class2idx):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    lab = class2idx[args.label_class].replace(" ","_").replace("/","")
    name = f"{args.discrim}_class_{lab}_iter_{args.num_iterations}_step_{args.stepsize}_sample_{args.num_samples}_wd_{args.wd}_bce_{args.BCE}_{args.trial_id}.jsonl"
    return base_path+name


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def make_logdir(args):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # logdir = os.path.join('runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    logdir = os.path.join('runs', f'{args.dataset}_{args.label}_{current_time}') #  current_time + '_' + socket.gethostname() + '_' + model_name)
    
    return logdir
from models.pytorch_pretrained_bert.modeling_adapter import GPT2LMHeadModel, GPT2Config
from transformers import (AdamW,WEIGHTS_NAME, CONFIG_NAME)
from utils.helper import average_distributed_scalar, load_model_recursive
from argparse import ArgumentParser
from transformers import GPT2Tokenizer
from itertools import chain
from torch.utils.data import Dataset, DataLoader
import torch
import os
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
import math
from pprint import pformat
import jsonlines
from nltk import tokenize
from metric.lm_score import get_ppl
                    
######################################################################################
# CUDA_VISIBLE_DEVICES=1 python train_supervised_adapter.py --dataset SENT --label very_negative --iter 75 (1)
# CUDA_VISIBLE_DEVICES=1 python train_supervised_adapter.py --dataset SENT --label very_negative --iter 75 --kl_weight 0.5 (2)
# CUDA_VISIBLE_DEVICES=1 python train_supervised_adapter.py --dataset SENT --label very_negative --iter 75 --kl_weight 0.5 --lr 6.25e-4 (3)
# CUDA_VISIBLE_DEVICES=1 python train_supervised_adapter.py --dataset SENT --label very_negative --iter 75 --lr 6.25e-4 (4)
# CUDA_VISIBLE_DEVICES=3 python main.py -D sentiment --label_class 3 --length 30 --num_samples 10 --evaluate --verbose --all_starter --load_check_point_adapter runs/SENT_very_negative_Apr02_13-16-33/pytorch_model.bin
# CUDA_VISIBLE_DEVICES=6 python main.py -D sentiment --label_class 3 --length 30 --num_samples 10 --interact --verbose --all_starter --load_check_point_adapter runs/question_0.5/pytorch_model.bin --speaker DGPT --repetition_penalty 1.2
# python train_supervised_adapter.py --dataset SENT --label very_negative --iter 75 --lr 6.25e-4 (4)
# python train_supervised_adapter.py --dataset SENT --label very_positive --iter 25
# python train_supervised_adapter.py --dataset TOXI --label toxic --iter 25
# python train_supervised_adapter.py --dataset QUEST --label question --iter 25
######################################################################################

MODEL_INPUTS = ["input_ids", "lm_labels"]
EOS_ID = 50256

TASK_MAP = {"very_negative":0, "very_positive":1, "toxic":2, "question":3, "Business":4, "SciTech":5, "Sports":6, "World":7}

def parse_name(f,cleaner):
    name = f.replace(cleaner,"").replace(".jsonl","").split("_")
    classifer = " ".join(name[:name.index('class')])
    lable = " ".join(name[name.index('class')+1:name.index('iter')])
    itr = " ".join(name[name.index('iter')+1:name.index('step')])
    stp = " ".join(name[name.index('step')+1:name.index('sample')])
    if "wd" not in name:
        sample = " ".join(name[name.index('sample')+1:])
        wd = False
    else:
        sample = " ".join(name[name.index('sample')+1:name.index('wd')])
        wd = " ".join(name[name.index('wd')+1:])
    return classifer,lable,itr,stp,sample,wd

class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""
    def __init__(self, data):
        self.data = data
        self.dataset_len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item
    def __len__(self):
        return self.dataset_len

def collate_fn(data):
    padding = 94
    max_l = max(len(x["input_ids"]) for x in data)
    padded_dataset = {n:[] for n in MODEL_INPUTS}
    for x in data:
        padded_dataset["lm_labels"].append( x["lm_labels"]+ [-100]*(max_l-len(x["lm_labels"]))  )
        padded_dataset["input_ids"].append(x["input_ids"]+ [padding]*(max_l-len(x["input_ids"])))

    for input_name in MODEL_INPUTS:
        padded_dataset[input_name] = torch.tensor(padded_dataset[input_name])
    return padded_dataset

def build_input_from_segments(args, history, reply, tokenizer):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    sequence = [tokenizer.encode(h) + [EOS_ID] for h in history] + [tokenizer.encode(reply)]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + sequence[-1]
    return instance

def make_data_loader(args,tokenizer):
    mapper = {"very_negative":"results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_75_step_0.02_sample_10_wd_False_bce_False.jsonl",
    "very_positive":"results/evaluate/sentiment_class_very_positive/sentiment_class_very_positive_iter_25_step_0.02_sample_10_wd_False_bce_False.jsonl",
    "toxic":"results/evaluate/toxicity_class_toxic/toxicity_class_toxic_iter_75_step_0.02_sample_10_wd_False_bce_False.jsonl",
    "question":"results/evaluate/daily_dialogue_act_class_question/daily_dialogue_act_class_question_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl",
    "Business": "results/evaluate/AG_NEWS_class_Business/AG_NEWS_class_Business_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl",
    "SciTech": "results/evaluate/AG_NEWS_class_SciTech/AG_NEWS_class_SciTech_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl",
    "Sports": "results/evaluate/AG_NEWS_class_Sports/AG_NEWS_class_Sports_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl",
    "World": "results/evaluate/AG_NEWS_class_World/AG_NEWS_class_World_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"
    }
    
    f = mapper[args.label]
    response = []
    with jsonlines.open(f) as reader: 
        for i, obj in enumerate(reader):
            text = " ".join(tokenize.sent_tokenize(obj["hyp"]["PPLM"][0][-1])[:2])
            score = get_ppl(text)
            if score>700:
                continue
            response.append(obj['conversation']['conversation']+[text])
            
    dataset = []
    for r in response:
        seq = build_input_from_segments(args, r[:-1], r[-1], tokenizer)
        dataset.append(seq)
    train_dataset = DatasetTrain(dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,collate_fn=collate_fn)

    return train_loader

def make_logdir(args):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    #current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # logdir = os.path.join('runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    logdir = os.path.join('runs', f'{args.label}_{args.kl_weight}') #  current_time + '_' + socket.gethostname() + '_' + model_name)
    
    return logdir


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_size', type=str, default="medium", help='Size of dialoGPT')
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=15, help="Number of previous exchanges to keep in history")
    parser.add_argument("--max_seq_len", type=int, default=200, help="Max number of tokens")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--debug", action='store_true', help="debugging mode")
    parser.add_argument("--dataset", type=str, default='SENT', help="Choose between SENT|TOXI|EMO|QUEST|TOPI ")
    parser.add_argument("--label", type=str, default='very_negative', help="Choose between very_positive|very_negative|toxic|question")
    parser.add_argument("--kl_weight", type=float, default=0, help="kl constraint for language model")
    parser.add_argument("--iter", type=int, default=75, help="Load data from a certain iteration")
    parser.add_argument("--load_check_point_adapter",type=str,default="")

    args = parser.parse_args()


    args.model_path = f'models/dialoGPT/{args.model_size}/'
    config = GPT2Config.from_json_file(os.path.join(args.model_path, 'config.json'))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    train_loader = make_data_loader(args,tokenizer)

    if(args.load_check_point_adapter != ""):
        print("Loading ADAPTERS")
        model = load_model_recursive(GPT2LMHeadModel(config), args.load_check_point_adapter, args, verbose=True)
    else:
        model = load_model_recursive(GPT2LMHeadModel(config), args.model_path+f"{args.model_size}_ft.pkl", args, verbose=True)
    model.to(args.device)

    #optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    parameters_to_update = [p for n, p in model.named_parameters() if "adapter" in str(n)]
    optimizer = AdamW(parameters_to_update, lr=args.lr, correct_bias=True)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
        input_ids, lm_labels = batch
        lm_loss = model(input_ids=input_ids, lm_labels=lm_labels, task_id=TASK_MAP[args.label], kl_weight=args.kl_weight)
        loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()


    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(batch[input_name].to(args.device) for input_name in MODEL_INPUTS)
            input_ids, lm_labels = batch
            # if we dont send labels to model, it doesnt return losses
            lm_logits, *_ = model(input_ids=input_ids, task_id=TASK_MAP[args.label])
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted,), (lm_labels_flat_shifted,)

    trainer = Engine(update)
    evaluator = Engine(inference)


    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(train_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(train_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(train_loader))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(), output_transform=lambda x: (x[0][0], x[1][0]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    def print_loss(engine,loss):
        with open("results/evaluate/sentiment_class_very_negative/log.txt", "a") as f:
            f.write(f"kl:{args.kl_weight}, lr:{args.lr}, epoch:{engine.state.epoch}, loss:{loss}")
            f.write("\n") 
            
    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))
        #evaluator.add_event_handler(Events.EPOCH_COMPLETED, print_loss, evaluator.state.metrics["nll"])

        log_dir = make_logdir(args)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=1)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

    
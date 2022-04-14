import jsonlines
import numpy as np
import copy
from tabulate import tabulate
from transformers import GPT2Tokenizer
from utils.helper import truncate
from metric.dist_score import eval_distinct
from metric.lm_score import get_ppl_simplified
from metric.sentiment_classifiers import get_sentiment_score, get_vater_score
from metric.torchMoji.examples.score_texts_emojis import get_emoji_score
from metric.text_classifier import get_text_score_AGNEWS
from nltk import tokenize

def parse_name(f, cleaner):
    f = f.replace("_1.jsonl",".jsonl")
    f = f.replace("_119.jsonl",".jsonl")
    name = f.replace(cleaner,"").replace(".jsonl","").split("_")
    classifer = " ".join(name[:name.index('class')])
    lable = " ".join(name[name.index('class')+1:name.index('iter')])
    itr = " ".join(name[name.index('iter')+1:name.index('step')])
    stp = " ".join(name[name.index('step')+1:name.index('sample')])

    sample = " ".join(name[name.index('sample')+1:name.index('wd')])
    wd = eval(" ".join(name[name.index('wd')+1:name.index('bce')]))
    bce = eval(" ".join(name[name.index('bce')+1:]))
    
    return classifer,lable,itr,stp,sample,wd,bce

def get_score(responses, lable, starter, classifer):
    # print(responses[0])
    responses = [" ".join(tokenize.sent_tokenize(r)[:2]) for r in responses]
    # print(responses[0])

    vater_score = get_vater_score(responses,"very positive")
    emoji, _ = get_emoji_score(responses, "terrified")
    if("very" in lable):
        score = get_sentiment_score(responses,lable) ## negative
        vater_score = get_vater_score(responses,lable)

    else:
        if("AG NEWS" in classifer):
            score = get_text_score_AGNEWS(responses,lable)
        else:
            score = 0

    d1,d2,d3 = eval_distinct(responses)
    dist = f"{str(d1)}/{str(d2)}/{str(d3)}"
    ppl = get_ppl_simplified(responses,starter)

    return ppl, dist, score, vater_score, emoji
    
def get_response(conversation):
    temp = []
    for turn in conversation:
        temp.append(turn['text'])
    return temp

def make_table(table,lable,clm_to_remove,text_class=False):
    print(f"Class {lable}")
    temp = [item for item in table if lable == item["lable"]]
    if(text_class):
        temp = [s for s in sorted(temp,key=lambda x: x['Model'])]
        scores = [copy.deepcopy(item['Score']) for item in temp]
        for d in temp:
            del d['Score']
        for i_, s in enumerate(scores):
            temp[i_].update(s)
    else:
        if("Score" in clm_to_remove):
            temp = [s for s in sorted(temp,key=lambda x: (x['Model']))]
        else:
            temp = [s for s in sorted(temp,key=lambda x: (x['Model'],x['Score']))]
    for l in clm_to_remove:
        for d in temp:
            del d[l]
        # map(lambda d: d.pop(l), temp)
    print(tabulate(temp,headers="keys",tablefmt='simple',floatfmt=".2f"))
    print()

def get_ppl_dist_all(rows,starter):
    all_responce = []
    for r in rows:
        all_responce += r["resp"]
    # dist= np.mean(list(eval_distinct(all_responce)))
    d1,d2,d3 = eval_distinct(all_responce)
    dist = f"{str(d1)}/{str(d2)}/{str(d3)}"
    ppl = get_ppl_simplified(all_responce,starter)
    return dist,ppl

def get_avg_measure(rows):
    score = []
    discrim_acc = []
    # valence = []
    ppl = []
    dist1,dist2,dist3 = [],[],[]
    for r in rows:
        score.append(r["Score"])
        discrim_acc.append(r["Discrim."])
        # valence.append(r["vater"])
        ppl.append(r["Ppl."])
        dist1.append(float(r["Dist."].split("/")[0]))
        dist2.append(float(r["Dist."].split("/")[1]))
        dist3.append(float(r["Dist."].split("/")[2]))
    return np.mean(score), np.mean(discrim_acc), np.mean(ppl), f"{str(truncate(np.mean(dist1),2))}/{str(truncate(np.mean(dist2),2))}/{str(truncate(np.mean(dist3),2))}"

def make_row(name,rows,starter):
    print(name,len(rows))
    # d, p = get_ppl_dist_all(rows,starter)
    score, disc, p, d = get_avg_measure(rows)
    return {"Model":name, "Ppl.":p, "Dist.":d, "Discrim.":disc,"Score":score}


def merge_table(table,lable,clm_to_remove,starter):
    starter = starter * len(lable)
    temp = [item for item in table if item["lable"] in lable]
    table = []
    # table.append(make_row("DGPT",[item for item in temp if "DGPT" == item["model"]],starter))
    table.append(make_row("HUMAN",[item for item in temp if "HUMAN" == item["model"] and item["lable"] in lable],starter))
    table.append(make_row("DGPT",[item for item in temp if "DGPT" == item["model"]],starter))
    table.append(make_row("DGPT+WD",[item for item in temp if "DGPT+WD" == item["model"]],starter))
    table.append(make_row("PPLM",[item for item in temp if "PPLM" == item["model"]],starter))
    table.append(make_row("ADAPTER",[item for item in temp if "ADAPTER" == item["model"]],starter))
    print(tabulate(table,headers="keys",tablefmt='latex',floatfmt=".2f"))
    print()


def get_human_responses():
    
    tokenizer = GPT2Tokenizer.from_pretrained('models/dialoGPT/medium/')
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
    
    conversation = data
    list_starters = []
    for i_c,conv in enumerate(conversation):
        for index in range(len(conv)-2):
            history = [conv[index]["text"],conv[index+1]["text"]]
            context_tokens = len(sum([tokenizer.encode(h) + [1111] for h in history],[]))
            if(context_tokens <= 70):
                if(index+2 <=len(conv)):
                    list_starters.append({"conversation":[conv[index]["text"],conv[index+1]["text"]],"response":conv[index+2]["text"]})
                else:
                    list_starters.append({"conversation":[conv[index]["text"],conv[index+1]["text"]],"response":""})
    return list_starters


def score():
    # # evaluate
    row_DGPT = []
    row_PPLM = []
    row = []
    files = [ 
            ("results/evaluate/sentiment_class_very_negative/","results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_75_step_0.02_sample_10_wd_False_bce_False.jsonl"),
            ("results/evaluate/sentiment_class_very_negative/","results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_10_step_0.02_sample_10_wd_True_bce_False.jsonl"),
            ("results/evaluate/sentiment_class_very_negative/","results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_0_step_0.02_sample_10_wd_False_bce_False_119.jsonl"),

            ("results/evaluate/sentiment_class_very_positive/","results/evaluate/sentiment_class_very_positive/sentiment_class_very_positive_iter_25_step_0.02_sample_10_wd_False_bce_False.jsonl"),
            ("results/evaluate/sentiment_class_very_positive/","results/evaluate/sentiment_class_very_positive/sentiment_class_very_positive_iter_10_step_0.02_sample_10_wd_True_bce_False.jsonl"),
            ("results/evaluate/sentiment_class_very_positive/","results/evaluate/sentiment_class_very_positive/sentiment_class_very_positive_iter_0_step_0.02_sample_10_wd_False_bce_False_119.jsonl"),

            ("results/evaluate/daily_dialogue_act_class_question/","results/evaluate/daily_dialogue_act_class_question/daily_dialogue_act_class_question_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/daily_dialogue_act_class_question/","results/evaluate/daily_dialogue_act_class_question/daily_dialogue_act_class_question_iter_10_step_0.02_sample_10_wd_True_bce_False.jsonl"),
            ("results/evaluate/daily_dialogue_act_class_question/","results/evaluate/daily_dialogue_act_class_question/daily_dialogue_act_class_question_iter_0_step_0.02_sample_10_wd_False_bce_False_119.jsonl"),

            ("results/evaluate/AG_NEWS_class_Business/","results/evaluate/AG_NEWS_class_Business/AG_NEWS_class_Business_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_Business/","results/evaluate/AG_NEWS_class_Business/AG_NEWS_class_Business_iter_10_step_0.02_sample_10_wd_True_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_Business/","results/evaluate/AG_NEWS_class_Business/AG_NEWS_class_Business_iter_0_step_0.02_sample_10_wd_False_bce_False_119.jsonl"),

            ("results/evaluate/AG_NEWS_class_SciTech/","results/evaluate/AG_NEWS_class_SciTech/AG_NEWS_class_SciTech_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_SciTech/","results/evaluate/AG_NEWS_class_SciTech/AG_NEWS_class_SciTech_iter_10_step_0.02_sample_10_wd_True_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_SciTech/","results/evaluate/AG_NEWS_class_SciTech/AG_NEWS_class_SciTech_iter_0_step_0.02_sample_10_wd_False_bce_False_119.jsonl"),

            # ("results/evaluate/AG_NEWS_class_World/","results/evaluate/AG_NEWS_class_World/AG_NEWS_class_World_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            # ("results/evaluate/AG_NEWS_class_World/","results/evaluate/AG_NEWS_class_World/AG_NEWS_class_World_iter_10_step_0.02_sample_10_wd_True_bce_False_1.jsonl"),
            # ("results/evaluate/AG_NEWS_class_World/","results/evaluate/AG_NEWS_class_World/AG_NEWS_class_World_iter_0_step_0.02_sample_10_wd_False_bce_False_119.jsonl"),


            ("results/evaluate/AG_NEWS_class_Sports/","results/evaluate/AG_NEWS_class_Sports/AG_NEWS_class_Sports_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_Sports/","results/evaluate/AG_NEWS_class_Sports/AG_NEWS_class_Sports_iter_10_step_0.02_sample_10_wd_True_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_Sports/","results/evaluate/AG_NEWS_class_Sports/AG_NEWS_class_Sports_iter_0_step_0.02_sample_10_wd_False_bce_False_119.jsonl"),
            ]
    human = get_human_responses()
    done = set()
    first_n = 200
    for (cleaner,f) in files:
        classifer,lable,itr,stp,sample, wd,bce = parse_name(f,cleaner)
        print(itr,lable, stp,sample, wd,bce)
        if wd:
            acc_DGPT_RE_WD = []
            resp_DGPT_RE_WD = []
            resp_human = []
            starter = []
            with jsonlines.open(f) as reader: 
                for i_, obj in enumerate(reader):
                    acc_DGPT_RE_WD.append(obj["acc"]["PPLM"])
                    resp_DGPT_RE_WD.append(obj["hyp"]["PPLM"][0][-1])
                    starter.append(obj["conversation"]["conversation"])
                    resp_human.append(human[i_]["response"])
                    if(i_ == first_n):break
            ppl, dist, score, vater_score, emoji = get_score(resp_DGPT_RE_WD, lable, starter, classifer)
            row.append({"Model":"DGPT+WD","lable":lable,"sample":sample,"iter":itr
                        ,"Step":None,"Discrim.":np.mean(acc_DGPT_RE_WD)*100,
                        "Ppl.":ppl,"Dist.":dist,"resp":resp_DGPT_RE_WD,
                        "Score":score*100,"vater":100*vater_score,"emoji":emoji})

            ppl_HUMAN, dist_HUMAN, score_HUMAN, vater_score_HUMAN, emoji_HUMAN = get_score(resp_human, lable, starter, classifer)
            row.append({"Model":"HUMAN","lable":lable,"sample":None,"iter":None
                        ,"Step":None,"Discrim.":0.0,
                        "Ppl.":ppl_HUMAN,"Dist.":dist_HUMAN,"resp":resp_human,
                        "Score":score_HUMAN*100,"vater":100*vater_score_HUMAN,"emoji":emoji_HUMAN})
                
        elif(not wd and int(itr) in [0]):
            acc_ADAPTER = []
            resp_ADAPTER = []
            starter = []
            with jsonlines.open(f) as reader: 
                for i_, obj in enumerate(reader):
                    acc_ADAPTER.append(obj["acc"]["DGPT"])
                    resp_ADAPTER.append(obj["hyp"]["DGPT"][0][-1])
                    starter.append(obj["conversation"]["conversation"])
                    if(i_ == first_n):break
            ppl, dist, score, vater_score, emoji = get_score(resp_ADAPTER, lable, starter, classifer)
            row.append({"Model":"ADAPTER","lable":lable,"sample":sample,"iter":itr
                        ,"Step":None,"Discrim.":np.mean(acc_ADAPTER)*100,
                        "Ppl.":ppl,"Dist.":dist,"resp":resp_ADAPTER,
                        "Score":score*100,"vater":100*vater_score,"emoji":emoji})
        # if(not wd and int(itr)==75 and int(sample)==10 and float(stp)==0.02 and not bce):
        elif(not wd):
            # print(itr,lable, stp,sample, wd,bce)
            acc_PPLM, acc_DGPT, acc_PPLM_WD, acc_DGPT_WD, resp_PPLM, resp_DGPT, resp_PPLM_WD, resp_DGPT_WD = [],[],[],[],[],[],[],[]
            starter = []
            with jsonlines.open(f) as reader: 
                for i_, obj in enumerate(reader):
                    ## DGPT 
                    acc_DGPT.append(str(sorted(obj["hyp"]["DGPT"])[0][-2])==lable)
                    resp_DGPT.append(sorted(obj["hyp"]["DGPT"])[0][-1])
                    ## DGPT +WD
                    acc_DGPT_WD.append(obj["acc"]["DGPT"])
                    resp_DGPT_WD.append(obj["hyp"]["DGPT"][0][-1])

                    ## PPLM 
                    acc_PPLM.append(str(sorted(obj["hyp"]["PPLM"])[0][-2])==lable)
                    resp_PPLM.append(sorted(obj["hyp"]["PPLM"])[0][-1])
                    ## PPLM + WD
                    acc_PPLM_WD.append(obj["acc"]["PPLM"])
                    resp_PPLM_WD.append(obj["hyp"]["PPLM"][0][-1])
                    starter.append(obj["conversation"]["conversation"])
                    if(i_ == first_n):break
            if(len(resp_DGPT) and len(resp_PPLM) and len(acc_DGPT_WD) and len(acc_PPLM_WD)):
                if(lable not in done):
                    # ppl, dist, score, vater_score, emoji = get_score(resp_DGPT, lable, starter, classifer)
                    ppl_WD, dist_WD, score_WD, vater_score_WD, emoji_WD = get_score(resp_DGPT_WD, lable, starter, classifer)
                    # row.append({"Model":"DGPT","lable":lable,"sample":1,"iter":None
                    #             ,"Step":None,"Discrim.":np.mean(acc_DGPT),
                    #             "Ppl.":ppl,"Dist.":dist,"resp":resp_DGPT,
                    #             "Score":score,"vater":100*vater_score,"emoji":emoji})
                    row.append({"Model":"DGPT","lable":lable,"sample":sample,"iter":None,
                                "Step":None,"Discrim.":np.mean(acc_DGPT_WD)*100,
                                "Ppl.":ppl_WD,"Dist.":dist_WD,"resp":resp_DGPT_WD,
                                "Score":score_WD*100,"vater":100*vater_score_WD,"emoji":emoji_WD})
                # ppl_pplm, dist_pplm, score_pplm, vater_score_pplm,emoji_pplm = get_score(resp_PPLM, lable, starter, classifer)
                ppl_pplm_WD, dist_pplm_WD, score_pplm_WD, vater_score_pplm_WD,emoji_pplm_WD = get_score(resp_PPLM_WD, lable, starter, classifer)
                
                # row.append({"Model":"PPLM","lable":lable,"sample":1,"iter":itr,
                #             "Step":stp,"Discrim.":np.mean(acc_PPLM),
                #             "Ppl.":ppl_pplm,"Dist.":dist_pplm,"resp":resp_PPLM,
                #             "Score":score_pplm,"vater":100*vater_score_pplm,"emoji":emoji_pplm})

                row.append({"Model":"PPLM","lable":lable,"sample":sample,"iter":itr,
                            "Step":stp,"Discrim.":np.mean(acc_PPLM_WD)*100,
                            "Ppl.":ppl_pplm_WD,"Dist.":dist_pplm_WD,"resp":resp_PPLM_WD,
                            "Score":score_pplm_WD*100,"vater":100*vater_score_pplm_WD,"emoji":emoji_pplm_WD})
                done.add(lable)

    # merge_table(copy.deepcopy(row),["very negative","very positive","Business","Sports","SciTech","question"],["resp","vater","lable"],starter)
    print("Sentiment")
    make_table(copy.deepcopy(row),"very negative",["resp","lable","sample","iter","Step"])
    make_table(copy.deepcopy(row),"very positive",["resp","lable","sample","iter","Step"])
    # merge_table(copy.deepcopy(row),["very negative","very positive"],["resp","lable"],starter)


    print("Question")
    print()
    make_table(copy.deepcopy(row),"question", ["resp","vater","sample","iter","Score","lable","Step","emoji"])

    print("AG_NEWS")
    print()
    make_table(copy.deepcopy(row),"Business", ["resp","vater","sample","iter","lable","Step","emoji"],text_class=False)
    make_table(copy.deepcopy(row),"Sports", ["resp","vater","sample","iter","lable","Step","emoji"],text_class=False)
    make_table(copy.deepcopy(row),"SciTech", ["resp","vater","sample","iter","lable","Step","emoji"],text_class=False)
    # merge_table(copy.deepcopy(row),["Business","Sports","SciTech"],["resp","emoji","vater","lable"],starter)
    
if __name__ == '__main__':
    score()

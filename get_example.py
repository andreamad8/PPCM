import jsonlines
import numpy as np
import copy
import csv
import argparse
from tabulate import tabulate
from transformers import GPT2Tokenizer

def parse_name(f, cleaner):
    f = f.replace("_1.jsonl",".jsonl")
    f = f.replace("_111.jsonl",".jsonl")
    name = f.replace(cleaner,"").replace(".jsonl","").split("_")
    classifer = " ".join(name[:name.index('class')])
    label = " ".join(name[name.index('class')+1:name.index('iter')])
    itr = " ".join(name[name.index('iter')+1:name.index('step')])
    stp = " ".join(name[name.index('step')+1:name.index('sample')])
    # bce = False
    # if("bce" in name):
    sample = " ".join(name[name.index('sample')+1:name.index('wd')])
    wd = eval(" ".join(name[name.index('wd')+1:name.index('bce')]))
    bce = eval(" ".join(name[name.index('bce')+1:]))
    
    return classifer,label,itr,stp,sample,wd,bce
    
def print_starter(conversation):
    for c in conversation:
        print(f"Human >>> {c}")

def print_example(table,label,starter,human, number=10):
    template = {"HUMAN":8,
                "DGPT":2,
                "DGPT+R":3,
                "DGPT+R+WD":4,
                "PPLM":5,
                "PPLM+R":6,
                "ADAPTER+R":7}
    index_mapper = []
    table = [item for item in table if label == item["label"]]

    for i_s, stat in enumerate(starter):
        temp_table = []
        for s in stat:
            temp_table.append({"Model":"HUMAN","Response":s})
        temp_table += ["null"]*7
        for i_t, item in enumerate(table):
            temp_table[template[item["model"]]] = {"Model":item["model"],"Response":item["resp"][i_s]}

        temp_table[template["HUMAN"]] = {"Model":"HUMAN","Response":human[i_s]["response"]}
        print(tabulate(temp_table,tablefmt='simple').replace(r"\begin{tabular}{ll}",r"\begin{tabularx}{\textwidth}{lX}").replace(r"\end{tabular}",r"\end{tabularx}"))
        print()
        print()
        if(i_s==number):break


def save_example_csv(table, label, starter, human, number=10):
    csv_columns = ['HUMAN1', 'HUMAN2', 'HUMAN3', 'DGPT', 'DGPT+R', 'DGPT+R+WD', 'PPLM', 'PPLM+R', 'ADAPTER+R']
    table = [item for item in table if label == item["label"]]
    gen_table = []
    for i_s, stat in enumerate(starter):
        temp_table = ['null'] * len(csv_columns)
        for i_t, s in enumerate(stat):
            temp_table[i_t] = s
        temp_table[csv_columns.index('HUMAN3')] = human[i_s]["response"]
        for i_t, item in enumerate(table):
            temp_table[csv_columns.index(item['model'])] = item["resp"][i_s]

        gen_table.append(temp_table)
        if (i_s == number): break

    save_file = 'results/evaluate/sample_' + label + '.csv'
    with open(save_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        writer.writerows(gen_table)


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
                    list_starters.append({"conversation":[conv[index]["text"],conv[index+1]["text"]],"response":"DUMMY"})
    return list_starters

def score(save=False):
    # if save=True, CSV file is generated and saved instead of printing results (for figure eight)
    # # evaluate
    row_DGPT = []
    row_PPLM = []
    row = []
    files = { 
            ("results/evaluate/sentiment_class_very_negative/","results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_75_step_0.02_sample_10_wd_False_bce_False.jsonl"),
            ("results/evaluate/sentiment_class_very_negative/","results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_10_step_0.02_sample_10_wd_True_bce_False.jsonl"),
            ("results/evaluate/sentiment_class_very_negative/","results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_0_step_0.02_sample_10_wd_False_bce_False_111.jsonl"),
            # ("results/evaluate/sentiment_class_very_negative/","results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_0_step_0.02_sample_10_wd_False_bce_False.jsonl"),
            # ("results/evaluate/sentiment_class_very_negative/","results/evaluate/sentiment_class_very_negative/sentiment_class_very_negative_iter_3_step_0.02_sample_10_wd_False_bce_False.jsonl"),

            ("results/evaluate/sentiment_class_very_positive/","results/evaluate/sentiment_class_very_positive/sentiment_class_very_positive_iter_25_step_0.02_sample_10_wd_False_bce_False.jsonl"),
            ("results/evaluate/sentiment_class_very_positive/","results/evaluate/sentiment_class_very_positive/sentiment_class_very_positive_iter_10_step_0.02_sample_10_wd_True_bce_False.jsonl"),
            ("results/evaluate/sentiment_class_very_positive/","results/evaluate/sentiment_class_very_positive/sentiment_class_very_positive_iter_0_step_0.02_sample_10_wd_False_bce_False_111.jsonl"),

            # ("results/evaluate/sentiment_class_very_positive/","results/evaluate/sentiment_class_very_positive/sentiment_class_very_positive_iter_0_step_0.02_sample_10_wd_False_bce_False.jsonl"),

            ("results/evaluate/daily_dialogue_act_class_question/","results/evaluate/daily_dialogue_act_class_question/daily_dialogue_act_class_question_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/daily_dialogue_act_class_question/","results/evaluate/daily_dialogue_act_class_question/daily_dialogue_act_class_question_iter_10_step_0.02_sample_10_wd_True_bce_False.jsonl"),
            ("results/evaluate/daily_dialogue_act_class_question/","results/evaluate/daily_dialogue_act_class_question/daily_dialogue_act_class_question_iter_0_step_0.02_sample_10_wd_False_bce_False_111.jsonl"),
            
            # ("results/evaluate/daily_dialogue_act_class_question/","results/evaluate/daily_dialogue_act_class_question/daily_dialogue_act_class_question_iter_0_step_0.02_sample_10_wd_False_bce_False.jsonl"),

            ("results/evaluate/toxicity_class_toxic/","results/evaluate/toxicity_class_toxic/toxicity_class_toxic_iter_75_step_0.02_sample_10_wd_False_bce_False.jsonl"),
            ("results/evaluate/toxicity_class_toxic/","results/evaluate/toxicity_class_toxic/toxicity_class_toxic_iter_10_step_0.02_sample_10_wd_True_bce_False.jsonl"),
            ("results/evaluate/toxicity_class_toxic/","results/evaluate/toxicity_class_toxic/toxicity_class_toxic_iter_0_step_0.02_sample_10_wd_False_bce_False_111.jsonl"),
            # ("results/evaluate/toxicity_class_toxic/","results/evaluate/toxicity_class_toxic/toxicity_class_toxic_iter_0_step_0.02_sample_10_wd_False_bce_False.jsonl"),

            ("results/evaluate/AG_NEWS_class_Business/","results/evaluate/AG_NEWS_class_Business/AG_NEWS_class_Business_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_Business/","results/evaluate/AG_NEWS_class_Business/AG_NEWS_class_Business_iter_10_step_0.02_sample_10_wd_True_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_Business/","results/evaluate/AG_NEWS_class_Business/AG_NEWS_class_Business_iter_0_step_0.02_sample_10_wd_False_bce_False_111.jsonl"),

            ("results/evaluate/AG_NEWS_class_SciTech/","results/evaluate/AG_NEWS_class_SciTech/AG_NEWS_class_SciTech_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_SciTech/","results/evaluate/AG_NEWS_class_SciTech/AG_NEWS_class_SciTech_iter_10_step_0.02_sample_10_wd_True_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_SciTech/","results/evaluate/AG_NEWS_class_SciTech/AG_NEWS_class_SciTech_iter_0_step_0.02_sample_10_wd_False_bce_False_111.jsonl"),

            ("results/evaluate/AG_NEWS_class_World/","results/evaluate/AG_NEWS_class_World/AG_NEWS_class_World_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_World/","results/evaluate/AG_NEWS_class_World/AG_NEWS_class_World_iter_10_step_0.02_sample_10_wd_True_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_World/","results/evaluate/AG_NEWS_class_World/AG_NEWS_class_World_iter_0_step_0.02_sample_10_wd_False_bce_False_111.jsonl"),


            ("results/evaluate/AG_NEWS_class_Sports/","results/evaluate/AG_NEWS_class_Sports/AG_NEWS_class_Sports_iter_75_step_0.02_sample_10_wd_False_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_Sports/","results/evaluate/AG_NEWS_class_Sports/AG_NEWS_class_Sports_iter_10_step_0.02_sample_10_wd_True_bce_False_1.jsonl"),
            ("results/evaluate/AG_NEWS_class_Sports/","results/evaluate/AG_NEWS_class_Sports/AG_NEWS_class_Sports_iter_0_step_0.02_sample_10_wd_False_bce_False_111.jsonl"),
            }

    human = get_human_responses()
    done = set()
    for (cleaner,f) in files:
        classifer,label,itr,stp,sample, wd,bce = parse_name(f,cleaner)
        print(itr,label, stp,sample, wd,bce)
        if wd:
            acc_DGPT_RE_WD = []
            resp_DGPT_RE_WD = []
            starter = []
            with jsonlines.open(f) as reader: 
                for i_, obj in enumerate(reader):
                    acc_DGPT_RE_WD.append(obj["acc"]["PPLM"])
                    resp_DGPT_RE_WD.append(obj["hyp"]["PPLM"][0][-1])
                    starter.append(obj["conversation"]["conversation"])
                    if(i_ == 10):break
            row.append({"model":"DGPT+R+WD","label":label,"sample":sample,"iter":itr
                        ,"Step":None,"Acc":np.mean(acc_DGPT_RE_WD),
                        "ppl":0,"dist":0,"resp":resp_DGPT_RE_WD,
                        "score":0,"vater":0,"emoji":0})
                
        elif(not wd and int(itr) in [0]):
            acc_ADAPTER = []
            resp_ADAPTER = []
            starter = []
            with jsonlines.open(f) as reader: 
                for i_, obj in enumerate(reader):
                    acc_ADAPTER.append(obj["acc"]["DGPT"])
                    resp_ADAPTER.append(obj["hyp"]["DGPT"][0][-1])
                    starter.append(obj["conversation"]["conversation"])
                    if(i_ == 10):break

            row.append({"model":"ADAPTER+R","label":label,"sample":sample,"iter":itr
                        ,"Step":None,"Acc":np.mean(acc_ADAPTER),
                        "ppl":0,"dist":0,"resp":resp_ADAPTER,
                        "score":0,"vater":0,"emoji":0})
        # if(not wd and int(itr)==75 and int(sample)==10 and float(stp)==0.02 and not bce):
        elif(not wd):
            # print(itr,label, stp,sample, wd,bce)
            acc_PPLM, acc_DGPT, acc_PPLM_WD, acc_DGPT_WD, resp_PPLM, resp_DGPT, resp_PPLM_WD, resp_DGPT_WD = [],[],[],[],[],[],[],[]
            starter = []
            with jsonlines.open(f) as reader: 
                for i_, obj in enumerate(reader):
                    ## DGPT 
                    acc_DGPT.append(str(sorted(obj["hyp"]["DGPT"])[0][-2])==label)
                    resp_DGPT.append(sorted(obj["hyp"]["DGPT"])[0][-1])
                    ## DGPT +WD
                    acc_DGPT_WD.append(obj["acc"]["DGPT"])
                    resp_DGPT_WD.append(obj["hyp"]["DGPT"][0][-1])

                    ## PPLM 
                    acc_PPLM.append(str(sorted(obj["hyp"]["PPLM"])[0][-2])==label)
                    resp_PPLM.append(sorted(obj["hyp"]["PPLM"])[0][-1])
                    ## PPLM + WD
                    acc_PPLM_WD.append(obj["acc"]["PPLM"])
                    resp_PPLM_WD.append(obj["hyp"]["PPLM"][0][-1])
                    starter.append(obj["conversation"]["conversation"])
                    if(i_ == 10):break
            if(len(resp_DGPT) and len(resp_PPLM) and len(acc_DGPT_WD) and len(acc_PPLM_WD)):
                if(label not in done):
                    row.append({"model":"DGPT","label":label,"sample":1,"iter":None
                                ,"Step":None,"Acc":np.mean(acc_DGPT),
                                "ppl":0,"dist":0,"resp":resp_DGPT,
                                "score":0,"vater":0,"emoji":0})
                    row.append({"model":"DGPT+R","label":label,"sample":sample,"iter":None,
                                "Step":None,"Acc":np.mean(acc_DGPT_WD),
                                "ppl":0,"dist":0,"resp":resp_DGPT_WD,
                                "score":0,"vater":0,"emoji":0})
                
                row.append({"model":"PPLM","label":label,"sample":1,"iter":itr,
                            "Step":stp,"Acc":np.mean(acc_PPLM),
                            "ppl":0,"dist":0,"resp":resp_PPLM,
                            "score":0,"vater":0,"emoji":0})

                row.append({"model":"PPLM+R","label":label,"sample":sample,"iter":itr,
                            "Step":stp,"Acc":np.mean(acc_PPLM_WD),
                            "ppl":0,"dist":0,"resp":resp_PPLM_WD,
                            "score":0,"vater":0,"emoji":0})
                done.add(label)

    print("Sentiment")
    if save:
        save_example_csv(copy.deepcopy(row), "very negative",starter, human)
        save_example_csv(copy.deepcopy(row), "very positive",starter, human)
    else:
        print_example(copy.deepcopy(row), "very positive",starter, human)


    print("Toxic")
    print()
    save_example_csv(copy.deepcopy(row), "toxic", starter, human)
    # merge_table(copy.deepcopy(row),["toxic"],["resp","lable"],starter)


    # print("Emotion")
    # print()
    # make_table(copy.deepcopy(row),"angry",["resp","vater","lable"])
    # make_table(copy.deepcopy(row),"disgusted",["resp","vater","lable"])
    # make_table(copy.deepcopy(row),"terrified",["resp","vater","lable"])
    # make_table(copy.deepcopy(row),"joyful",["resp","vater","lable"])
    # make_table(copy.deepcopy(row),"sad",["resp","vater","lable"])
    # make_table(copy.deepcopy(row),"surprised",["resp","vater","lable"])
    # merge_table(copy.deepcopy(row),["angry","disgusted","terrified","joyful","sad","surprised"],["resp","vater","lable"],starter)

    print("Question")
    print()
    save_example_csv(copy.deepcopy(row), "question", starter, human)


    # print("AG_NEWS")
    # print()
    # make_table(copy.deepcopy(row),"Business", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Sports", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"SciTech", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"World", ["resp","emoji","vater","lable"],text_class=False)
    # merge_table(copy.deepcopy(row),["Business","Sports","SciTech","World"],["resp","emoji","vater","lable"],starter)
    
    # print("DBpedia")
    # print()
    # make_table(copy.deepcopy(row),"Company", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"EducationalInstitution", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Artist", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Athlete", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"OfficeHolder", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"MeanOfTransportation", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Building", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"NaturalPlace", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Village", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Animal", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Plant", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Album", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"Film", ["resp","emoji","vater","lable"],text_class=False)
    # make_table(copy.deepcopy(row),"WrittenWork", ["resp","emoji","vater","lable"],text_class=False)
    # merge_table(copy.deepcopy(row),["Company","EducationalInstitution","Artist","Athlete",
    #                                 "OfficeHolder","MeanOfTransportation","Building",
    #                                 "NaturalPlace","Village","Animal","Plant","Album",
    #                                 "Film","WrittenWork"],["resp","emoji","vater","lable"],starter)
    # row = [s for s in sorted(row,key=lambda x: (x['model'],x['score']))]
    # print(tabulate(row,headers="keys",tablefmt='simple',floatfmt=".2f"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', help="saving the results in csv", action='store_true')
    args = parser.parse_args()
    score(save=args.save)

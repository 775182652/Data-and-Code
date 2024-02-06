# from transformers import AutoModel
#

# model = AutoModel.from_pretrained('xlm-roberta-large', proxies=proxies)
import replicate
import os
import json
import collections
import torch
#os.environ['REPLICATE_API_TOKEN'] = 'r8_1NLjuPD6KKxm8dtzBo3j0uYZMEyD9D21IQyAj'
#export REPLICATE_API_TOKEN=r8_FYixhqR4ajlEW7C1n5oK0cbXWHkMt0A0pnTU2



# -*- coding: utf-8 -*-
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


import openpyxl




def getCot():
    dic_list= []
    base_path = r'./DataAll'
    files = os.listdir(base_path)

    xlsx = openpyxl.load_workbook("./DataAll.xlsx")
    table = xlsx['Sheet1']
    nrows = table.rows
    # with open("transST.json", "r") as f:
    #     st_list = json.load(f)


    i = 0
    tag = 0
    for row in nrows:
        if i == 0:
            i += 1
            continue
        # if i > 1000 :
        #     break
        line = [col.value for col in row]
        if line[0] == "1.jpg":
            tag = 1
        if tag == 0:
            continue
        if line[3] != 1:
            continue
        if line[2] == None:
            continue
        if line[6] == None:
            continue

        with open("cot_llava_cn.json", "r") as f:
            dic_list = json.load(f)
        # if path == "._.DS_Store" : continue
        pickid = str(line[0])
        # stdict = [c for c in st_list if c['ID'] == line[0]]
        target = str(line[4])
        source = str(line[5])
        full_path = os.path.join(base_path, pickid)
        if os.path.exists(full_path):
            output = getLLavaCot3(full_path, target, source)
            dictframe = collections.defaultdict()
            dictframe['ID'] = pickid
            print(pickid)
            dictframe['cot_zh'] = output
            print("zh:" + dictframe['cot_zh'])
            dic_list.append(dictframe)
        #print(output)
        i = i+1
        with open("cot_llava_cn.json", "w", encoding='utf-8') as f:
            json.dump(dic_list, f)





from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model







def getLLavaCot3(full_path, target, source):

    result1 = getLLavaCot2(full_path, target)
    result2 = getLLavaCot1(full_path, target, source)
    model_path = "./llava-v1.5-7b"
    prompt = "根据" + result1 + "和图片, " + "对" + source + "和" + target + "的以下相似点进行筛选，并总结，" + result2 + "，回答不要包含问题且不超过20个字"
    image_file = full_path

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    outputs = eval_model(args)
    return outputs

def getLLavaCot2(full_path, target):
    model_path = "./llava-v1.5-7b"
    prompt = "根据图片说出" + target + "在图中的情感倾向"
    image_file = full_path

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    outputs = eval_model(args)
    return outputs

def getLLavaCot1(full_path, target, source):
    model_path = "./llava-v1.5-7b"
    prompt = "请回答在图中" + target + "和" + source + "二者的相似点"
    image_file = full_path

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()
    outputs = eval_model(args)
    return outputs








if __name__ == '__main__':
    getCot()
    # with open("cot_llava.json", "r") as f:
    #     dic_list = json.load(f)
    # # if path == "._.DS_Store" : continue
    # dict = [c for c in dic_list if c['ID'] == '27.jpg']
    # print(dict)

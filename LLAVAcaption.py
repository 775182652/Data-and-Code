# from transformers import AutoModel
#

import replicate
import os
import json
import collections
import torch




# -*- coding: utf-8 -*-
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


import openpyxl




def getCaption():
    dic_list= []
    base_path = r'./DataAll'
    files = os.listdir(base_path)

    xlsx = openpyxl.load_workbook("./DataAll.xlsx")
    table = xlsx['Sheet1']
    nrows = table.rows



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

        with open("caption_list_llava.json", "r") as f:
            dic_list = json.load(f)
        # if path == "._.DS_Store" : continue
        pickid = str(line[0])
        full_path = os.path.join(base_path, pickid)
        if os.path.exists(full_path):
            output = getcap(full_path)
            dictframe = collections.defaultdict()
            dictframe['ID'] = pickid
            print(pickid)
            dictframe['caption_en'] = output
            print("en:" + dictframe['caption_en'])
            dictframe['caption_zh'] = baiduTranslate(output)
            print("zh:" + dictframe['caption_zh'])
            dic_list.append(dictframe)
        #print(output)
        i = i+1
        with open("caption_list_llava.json", "w", encoding='utf-8') as f:
            json.dump(dic_list, f)





from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


def getcap(full_path):

    model_path = "./llava-v1.5-7b"
    prompt = "What are the things I should be cautious about when I visit here?"
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


import random
import hashlib
import urllib
import requests
import http.client
import json

def baiduTranslate(translate_text, flag=1):
    '''
    :param translate_text: 待翻译的句子，len(q)<2000
    :param flag: 1:原句子翻译成英文；0:原句子翻译成中文
    :return: 返回翻译结果。
    For example:
    q=我今天好开心啊！
    result = {'from': 'zh', 'to': 'en', 'trans_result': [{'src': '我今天好开心啊！', 'dst': "I'm so happy today!"}]}
    '''

    appid = ''  # 填写你的appid
    secretKey = ''  # 填写你的密钥
    httpClient = None
    myurl = '/api/trans/vip/translate'  # 通用翻译API HTTP地址
    fromLang = 'en'  # 原文语种

    if flag:
        toLang = 'zh'  # 译文语种
    else:
        toLang = 'en'  # 译文语种

    salt = random.randint(3276, 65536)

    sign = appid + translate_text + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(translate_text) + '&from=' + fromLang + \
            '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

    # 建立会话，返回结果
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        # return result
        return result['trans_result'][0]['dst']

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()





def transform(input):
    tokenizer = AutoTokenizer.from_pretrained("./opus-mt-en-zh")
    device = torch.device('cuda:0')
    model = AutoModelForSeq2SeqLM.from_pretrained("./opus-mt-en-zh").to(device)

    text = input
    # Tokenize the text
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[text], return_tensors="pt")

    # Make sure that the tokenized text does not exceed the maximum
    # allowed size of 512
    batch["input_ids"] = batch["input_ids"][:, :512].to(device)
    batch["attention_mask"] = batch["attention_mask"][:, :512].to(device)

    # Perform the translation and decode the output
    translation = model.generate(**batch)
    result = tokenizer.batch_decode(translation, skip_special_tokens=True)
    print(result[0])
    return result[0]

if __name__ == '__main__':
    getCaption()
    # with open("cot_llava.json", "r") as f:
    #     dic_list = json.load(f)
    # # if path == "._.DS_Store" : continue
    # dict = [c for c in dic_list if c['ID'] == '27.jpg']
    # print(dict)

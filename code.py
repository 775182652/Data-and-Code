# import pickle
#
# with open('./archive/train_idx.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print(data)
# ====================================================
# Directory settings
# ====================================================
import os

import json
import random
PIC_DIR='./DataAll/'
INPUT_DIR = './DataAll.xlsx'
INDEX_DIR = './archive/'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ====================================================
# CFG
# ====================================================
class CFG:
    wandb = False
    _wandb_kernel = 'mirukuuu'
    debug = False
    apex = True
    print_freq = 10
    num_workers = 4
    num_labels = 10
    label_id = 8
    label_id2 = 6
    loss_method = 'cse'
    modelName = "./xlm-roberta-base"
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = False
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 20
    encoder_lr = 3e-6
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    batch_size = 16
    fc_dropout = 0.2
    target_size = 1
    max_len = 128
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1
    seed = 42
    n_fold = 1
    trn_fold = [0, 1, 2, 3]
    train = True
    test = True
    middle_hidden_size = 256
    resnet_dropout = 0.2
    bert_dropout = 0.2
    attention_nheads = 16
    attention_dropout = 0.2
    fuse_dropout = 0.2
    out_hidden_size = 256


if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]


# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

# os.system('pip uninstall -y transformers')
# os.system('pip uninstall -y tokenizers')
# os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset transformers')
# os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset tokenizers')
# print(f"tokenizers.__version__: {tokenizers.__version__}")
# print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torchvision import models
from PIL import Image, ImageFile
import torchvision.transforms as T
# %env TOKENIZERS_PARALLELISM=false
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)




# ====================================================
# Data Loading
# ====================================================
import openpyxl
import collections
from tqdm import tqdm


def get_pic(pic_id, transform):
    # print(pic_id)
    path = os.path.join(PIC_DIR, pic_id)
    #     print("path:",path)
    try:
        image = Image.open(path).convert('RGB')
        if transform is not None:
            image = transform(image)
    #         print('get image.')
    except Exception:
        print('no image:', path)
        image = torch.zeros([3, 224, 224])
        # image = None
    return image


def load_data():
    with open("./cot_llava_cn.json", "r") as f:
        cotText_list = json.load(f)
    with open("./caption_list_llava.json", "r") as a:
        caption_list = json.load(a)
    xlsx = openpyxl.load_workbook(INPUT_DIR)
    table = xlsx['Sheet1']
    nrows = table.rows  # 获得行数 类型为迭代器
    totaldata = []
    traindata = []
    validdata = []
    testdata = []
    tokenizer = AutoTokenizer.from_pretrained(CFG.modelName)
    i = 0
    for row in tqdm(nrows):
        if i==0 :
            i += 1
            continue
        dataframe = []
        #print("row:", row)  # 包含了页名，cell，值
        line = [col.value for col in row]  # 取值
        if line[3] != 1 :
            continue
        if line[2] == None:
            continue
        if line[6] == None:
            continue
        # print(line[0])
        cotdict = [d for d in cotText_list if d['ID'] == line[0]]
        # print(cotdict)
        cotText = cotdict[0]['cot_zh']
        capdict = [c for c in caption_list if c['ID'] == line[0]]
        # print(capdict)
        capText = capdict[0]['caption_zh']
        # print("line2",line[2])
        combain = line[2]+","+cotText+","+capText
        #combain = line[2] + "," + cotText
        #combain = line[2] + "," + capText
        # print("combine", combain)
        #print(line[1])
        sample = collections.defaultdict()
        #cotText = WXYY.getSimilarity(line[4], line[5])
        #print(cotText)
        if not isinstance(combain, str):
            #         print("convert")
            #line[2] = str(line[2])
            combain = str(combain)
        tokenized = tokenizer(combain, truncation=True, padding='max_length',
                              max_length=CFG.max_len)
        for k, v in tokenized.items():
            #print('k:{},v:{}'.format(k, v))  # 将一句话的分词存入词典
            sample[k] = torch.LongTensor(v)
        combain = sample
        #print('line',line[1])
        #print(line[0])
        dataframe = {'id':line[0], 'texts':combain, 'label':int(line[6]-1)}
        totaldata.append(dataframe)
        i += 1
        # print(i)

    train_ratio = 0.7
    test_ratio = 0.15
    val_ratio = 0.15

    # 计算每个集合应该包含的元素数量
    total_elements = len(totaldata)
    train_elements = int(total_elements * train_ratio)
    test_elements = int(total_elements * test_ratio)
    val_elements = total_elements - train_elements - test_elements

    # 确保每个集合的元素数量是有效的（不会超过总元素数量）
    assert train_elements >= 0 and test_elements >= 0 and val_elements >= 0, "Invalid split ratios"

    # 使用random.sample随机选择元素，而不是按照索引顺序选择，以确保随机性
    random_indices = random.sample(range(total_elements), train_elements + test_elements + val_elements)
    #random_indices.sort()

    # 根据元素数量划分训练集、测试集和验证集的索引范围
    train_indices = random_indices[:train_elements]
    test_indices = random_indices[train_elements:train_elements + test_elements]
    val_indices = random_indices[train_elements + test_elements:]
    # print("train_indices",train_indices)
    # print("val_indices",val_indices)

    # 根据索引范围从原始数据中创建训练集、测试集和验证集
    train_data = [totaldata[i] for i in train_indices]
    test_data = [totaldata[i] for i in test_indices]
    val_data = [totaldata[i] for i in val_indices]

    traindata = np.array(train_data)
    validdata = np.array(val_data)
    testdata = np.array(test_data)
    #print(traindata)
    #print(len(validdata))
    return traindata, validdata, testdata



class TrainDataset(Dataset):
    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data
        self.transform = T.Compose([
            T.Resize((224, 224)),
            # T.RandomCrop((384,384)),
            # T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self):
        return len(self.data)
    def get_pic(self, pic_id, transform):
        # print(pic_id)
        path = os.path.join(PIC_DIR, pic_id)
        #     print("path:",path)
        try:
            image = Image.open(path).convert('RGB')
            if transform is not None:
                image = transform(image)
                #print(pic_id + 'get image.')
        except Exception:
            print('no image:',path)
            image = torch.zeros([3, 224, 224])
            # image = None
        return image

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.get_pic((str(item['id'])), self.transform)
        return item['texts'], img, item['label']



class MHAtt(nn.Module):
    def __init__(self, HIDDEN_SIZE):
        super(MHAtt, self).__init__()
        self.hideensize = HIDDEN_SIZE
        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(0.2)

    def forward(self, q, v, k, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            8,
            int(self.hideensize/8)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            8,
            int(self.hideensize/8)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            8,
            int(self.hideensize/8)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hideensize
        )

        atted = self.linear_merge(atted)
        atted = torch.squeeze(atted, dim=1)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = AutoModelForSequenceClassification.from_pretrained(cfg.modelName,
                                                                               num_labels=cfg.middle_hidden_size)
        #         self.text_encoder = AutoModel.from_pretrained(cfg.modelName)
        #         self.t_classifier = nn.Linear(768, 128)
        self.img_encoder = models.resnet50(pretrained=True)
        fc_inputs = self.img_encoder.fc.in_features
        self.img_encoder.fc = nn.Sequential(
            nn.Linear(fc_inputs, cfg.middle_hidden_size * 2),
            nn.BatchNorm1d(cfg.middle_hidden_size * 2),
            nn.Dropout(0.4),
            nn.Linear(cfg.middle_hidden_size * 2, cfg.middle_hidden_size),
            nn.ReLU(),
        )
        #         self.img_encoder.fc = nn.Sequential(nn.Linear(fc_inputs, 128),
        #                                           nn.Tanh())
        self.mtta = MHAtt(cfg.middle_hidden_size)

        self.attention1 = torch.nn.TransformerEncoderLayer(
            d_model=cfg.middle_hidden_size * 2,
            nhead=cfg.attention_nheads,
            dropout=cfg.attention_dropout
        )
        self.attention2 = torch.nn.TransformerEncoderLayer(
            d_model=cfg.middle_hidden_size * 2,
            nhead=cfg.attention_nheads,
            dropout=cfg.attention_dropout
        )
        self.attention3 = torch.nn.TransformerEncoderLayer(
            d_model=cfg.middle_hidden_size * 2,
            nhead=cfg.attention_nheads,
            dropout=cfg.attention_dropout
        )
        self.mergeliner = nn.Linear(int(cfg.middle_hidden_size*3), cfg.middle_hidden_size)
        self.output_layer = nn.Sequential(
            #nn.Linear(cfg.middle_hidden_size * 2, cfg.middle_hidden_size),
            nn.BatchNorm1d(cfg.middle_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(cfg.middle_hidden_size, cfg.num_labels),
        )

    #         self.init_weights(self.img_encoder.fc)
    #         self.init_weights(self.output_layer)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    def forward(self, texts, imgs):
        t_encoded = self.text_encoder(input_ids=texts['input_ids'], attention_mask=texts['attention_mask']).logits
        i_encoded = self.img_encoder(imgs)
        #x = torch.cat((t_encoded, i_encoded), 1)
        #x = self.attention1(x)
        #x = self.attention2(x)
        #x = self.attention3(x)
        #print(t_encoded.shape)
        #print(i_encoded.shape)
        x = self.mtta(t_encoded, i_encoded, i_encoded, None)
        # x = self.mtta(i_encoded, t_encoded, t_encoded, None)
        x = torch.concat((x, t_encoded),1)
        x = torch.concat((x, i_encoded),1)
        x = self.mergeliner(x)
        x = self.output_layer(x)
        return x

def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

# ====================================================
# Helper functions
# ====================================================
from torchmetrics import F1Score, Recall, Accuracy
import wandb
from sklearn.metrics import accuracy_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))






def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, model2=None, optimizer2=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    accs = AverageMeter()
    f1s = AverageMeter()
    rs = AverageMeter()
    global_step = 0
    for step, (inputs, imgs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
            #print('k:{},v:{}'.format(k, v))
        labels = labels.to(device)
        imgs = imgs.to(device)
        batch_size = labels.size(0) #CFG.batch_size = 16
        #print("batch_size", batch_size)

        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs, imgs)
            loss = criterion(y_preds, labels.to(torch.long))
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        y_preds = y_preds.squeeze(1)
        y_hat, y = y_preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
        preds = y_hat.argmax(1)
        acc = Accuracy(task="multiclass", average='micro', top_k=1, num_classes=CFG.num_labels)(
            torch.from_numpy(preds), torch.from_numpy(y))
        f1 = F1Score(task="multiclass", average='macro', top_k=1, num_classes=CFG.num_labels)(
            torch.from_numpy(preds), torch.from_numpy(y))
        r = Recall(task="multiclass", average='macro', top_k=1, num_classes=CFG.num_labels)(torch.from_numpy(preds),
                                                                                            torch.from_numpy(y))
        accs.update(acc, batch_size)
        f1s.update(f1, batch_size)
        rs.update(r, batch_size)
        scaler.scale(loss).backward()
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            #             loss.backward()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
    print("losses: ", losses.avg)
    print("acc_train: ", accs.avg)
    return model






def valid_fn(valid_loader, model, criterion, device, epoch, model2=None):
    losses = AverageMeter()
    accs = AverageMeter()
    f1s = AverageMeter()
    rs = AverageMeter()
    model.eval()
    preds = []
    for step, (inputs, imgs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        imgs = imgs.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs, imgs)
            loss = criterion(y_preds, labels.to(torch.long))
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        y_preds = y_preds.squeeze(1)
        y_hat, y = y_preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
        preds = y_hat.argmax(1)
        acc = Accuracy(task="multiclass", average='micro', top_k=1, num_classes=CFG.num_labels)(
            torch.from_numpy(preds), torch.from_numpy(y))
        f1 = F1Score(task="multiclass", average='macro', top_k=1, num_classes=CFG.num_labels)(
            torch.from_numpy(preds), torch.from_numpy(y))
        r = Recall(task="multiclass", average='macro', top_k=1, num_classes=CFG.num_labels)(torch.from_numpy(preds),
                                                                                            torch.from_numpy(y))
        accs.update(acc, batch_size)
        f1s.update(f1, batch_size)
        rs.update(r, batch_size)
        if step == (len(valid_loader) - 1):
            print('valid EPOCH: {0} '
                  '\t Acc: {acc.avg:.4f} '
                  '\t Recall: {r.avg:.4f} '
                  ' F1: {f1.avg:.4f}'
                  .format(epoch,
                          loss=losses,
                          acc=accs,
                          r=rs,
                          f1=f1s,
                          ))
    return accs.avg




def test_fn(valid_loader, model, criterion, device, epoch, model2=None):
    losses = AverageMeter()
    accs = AverageMeter()
    f1s = AverageMeter()
    rs = AverageMeter()
    model.eval()
    preds = []
    for step, (inputs, imgs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        imgs = imgs.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs, imgs)
            loss = criterion(y_preds, labels.to(torch.long))
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        y_preds = y_preds.squeeze(1)
        y_hat, y = y_preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
        preds = y_hat.argmax(1)
        acc = Accuracy(task="multiclass", average='micro', top_k=1, num_classes=CFG.num_labels)(
            torch.from_numpy(preds), torch.from_numpy(y))
        f1 = F1Score(task="multiclass", average='macro', top_k=1, num_classes=CFG.num_labels)(
            torch.from_numpy(preds), torch.from_numpy(y))
        r = Recall(task="multiclass", average='macro', top_k=1, num_classes=CFG.num_labels)(torch.from_numpy(preds),
                                                                                            torch.from_numpy(y))
        accs.update(acc, batch_size)
        f1s.update(f1, batch_size)
        rs.update(r, batch_size)
        if step == (len(valid_loader) - 1):
            print('test EPOCH: {0} '
                  '\t Acc: {acc.avg:.4f} '
                  '\t Recall: {r.avg:.4f} '
                  ' F1: {f1.avg:.4f}'
                  .format(epoch,
                          loss=losses,
                          acc=accs,
                          r=rs,
                          f1=f1s,
                          ))
    return accs.avg, f1s.avg, rs.avg








def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
        )
    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles
        )
    return scheduler





def train_loop(train, valid, test):
    fold = 0
    train_dataset = TrainDataset(CFG, train)
    valid_dataset = TrainDataset(CFG, valid)
    test_dataset = TrainDataset(CFG, test)
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    print("DataLoader info:\ttrain_loader:{}\tvalid_loader:{}\ttest_loader:{}".format(len(train_loader), len(valid_loader),len(test_loader)))
    model = CustomModel(CFG, config_path=None, pretrained=True)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    num_train_steps = int(len(train_loader) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    best_score = 0.
    best_test = 0.
    f1_test = 0.
    r_test = 0.
    print("EPOCH\t\t Acc\t\t Recall\t\t F1")
    for epoch in range(CFG.epochs):
        model = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)
        acc_avg = valid_fn(valid_loader, model, criterion, device, epoch)
        if acc_avg > best_score:
            best_score = acc_avg
            print("本次valid集上acc高于之前的，进入test")
            acc_test,f1_temp,r_temp = test_fn(test_loader, model, criterion, device, epoch)
            if acc_test > best_test :
                print("本次test集上acc高于之前的")
                best_test = acc_test
                f1_test = f1_temp
                r_test = r_temp
    torch.cuda.empty_cache()
    gc.collect()

    return best_test, f1_test, r_test

def calculateTime(start, end):
    run_time = round(end-start)
    # 计算时分秒
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
#     print (f'该程序运行时间：{hour}小时{minute}分钟{second}秒')
    return hour,minute,second

if __name__ == '__main__':
    start = end = time.time()
    train, valid, test = load_data()
    best_acc, best_f, best_r = train_loop(train, valid, test)
    end = time.time()
    h, m, s = calculateTime(start, end)
    print(f'-----耗时{h}小时{m}分钟{s}秒，最佳Acc:{best_acc:.4f}--,f1:{best_f:.4f}--,r:{best_r:.4f}-----')
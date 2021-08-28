from pickle import load
from random import shuffle
import torch
import torch.nn as nn
import numpy as np
from model import MLpretrainLayer
from utils.table_read import *
from pyspark.sql import SparkSession
from config.default_config import get_cfg_defaults
from utils.local_w2v import word_preprocess,load_dict
from torch.optim import AdamW
from tqdm import tqdm

def text2token(text, dic, lq, batch = 1):
    tokenlist = []
    if batch == 1:
        for seq in text[0]:
            wordlist = word_preprocess(seq)
            tokenseq = []
            for w in wordlist:
                if w in dic.keys():
                    tokenseq.append(dic[w])
                else:
                    tokenseq.append(dic['[UNK]'])
                
            if len(tokenseq)>=lq:
                tokenseq = tokenseq[:lq]
            else:
                tokenseq += [dic['[EOS]']] + [dic['[PAD]']] * (lq - len(tokenseq) -1)
            tokenlist.append(tokenseq)
    
    return tokenlist

def v_tp2adj(v_tp, e_mx, pretrain = False): #可能要改为张量拼接的实现方式
    node_type_dict = {'VH':0, 'H':1,'VE':2,'Cell':3,'Caption':4}
    edge_type = [('VH','H'), ('VH','VE'), ('VE','Cell'), ('VH','Caption'), ('Caption','VE'), ('Cell','H')]
    filter_mx = torch.zeros(6,len(v_tp), len(v_tp))
    adj = torch.from_numpy(e_mx).unsqueeze(0)
    adj = torch.cat([torch.add(adj, torch.eye(len(v_tp))).clone() for i in range(6)], dim = 0)

    for id,i in enumerate(v_tp):
        for dim, e_t in enumerate(edge_type):
            if i in e_t:
                filter_mx[dim,id,:] = 1
                filter_mx[dim,:,id] = 1
    v_tp2arr = []
    for i in v_tp:
        v_tp2arr.append(node_type_dict[i])

    multi_adj = torch.where(torch.add(filter_mx, adj)>1, filter_mx, torch.zeros(6,len(v_tp), len(v_tp)))
    arr_tp = torch.tensor(v_tp2arr)
    
    return multi_adj, arr_tp
    

def pretrain(dfgraphpath, cfg):
    spark = SparkSession \
        .builder \
        .appName('para_extract') \
        .master("local[8]") \
        .enableHiveSupport() \
        .config("spark.driver.memory", "15g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.maxResultSize", "6g")\
        .getOrCreate()
    spark.sparkContext.setCheckpointDir(dirName="graphframes_cps")
    g, num = read_graph(dfgraphpath, spark)


    save_dir = cfg.DATA.save_dir

    model = MLpretrainLayer.GATMLModelfortable(cfg)
    model.to(cfg.TRAIN.device)
    model.train()
    dic = load_dict(cfg.DATA.dicpath)[0]
    optimizer = AdamW(model.parameters(), lr=cfg.TRAIN.learning_rate)
    criterion = nn.CrossEntropyLoss()
    loss100 = []

    graphseq = list(range(num))
    if cfg.TRAIN.shuffle:
        shuffle(graphseq)
    for i in tqdm(graphseq):
        v_tx, v_tp, e_mx = search_graph(g, i)
        v_token = np.array(text2token(v_tx,dic,cfg.TRANS.lq))
        e_adj, v_tp = v_tp2adj(v_tp = v_tp[0], e_mx = e_mx[0], pretrain=True)

        optimizer.zero_grad()
        res = model(input = [v_token, v_tp, e_adj])

        
        loss = criterion(res[1], res[0])
        loss.backward(retain_graph=False)
        optimizer.step()
        loss100.append(loss.item())

    print("graph:{}, loss = {}".format(i, sum(loss100)/len(loss100)))
    
    model.save(save_dir)
        
    pass

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    pretrain(cfg.DATA.dfgraph, cfg)
    pass
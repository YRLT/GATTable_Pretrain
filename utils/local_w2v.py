from sys import warnoptions
from tqdm import tqdm, trange
import numpy as np
from gensim.models import word2vec
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.types import (
    StringType,
    ArrayType
)
from pyspark.sql.functions import udf
import time
import sys


def w2v_train_gensim(text_file):
    w2v_model = word2vec.Word2Vec(vector_size=300, sg=1, min_count=1)
    epoch = 20
    w2v_model.build_vocab_from_freq({'-':1})
    strlist = []
    with open(text_file + '/text.csv') as ft:
        for line in tqdm(ft.readlines()):
            sentences = line.split("[SEP]")
            for s in sentences:
                wl = word_preprocess(s)
                w2v_model.build_vocab(wl, update=True)
                if len(wl)>1:
                    strlist += [word_preprocess(s)]
        # 语料库过大，需要拆分为至少10片处理
        strlist = filter(lambda x:len(x)>1, strlist)
        w2v_model.train([strlist], epochs=1, total_examples=w2v_model.corpus_count)

        for i in range(epoch):
            print("current epoch:" + i)
            for line in tqdm(ft.readlines()):
                sentences = line.split("[SEP]")
                for s in sentences:
                    strlist = word_preprocess(s)
                    w2v_model.build_vocab(strlist, update=True)
                    if len(strlist)>1:
                        w2v_model.train(strlist, epochs=1)
        
    w2v_model.save("textEmbedding.bin")

def w2v_train_spark(fpath):
    spark = SparkSession \
        .builder \
        .appName('para_extract') \
        .master("local[8]") \
        .enableHiveSupport() \
        .config("spark.driver.memory", "20g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.maxResultSize", "10g") \
        .getOrCreate()  

    epoch = 1

    trainset = spark.read.csv(fpath + '/text/' + "part-00000-8560ee87-33f5-4665-ae61-d0beb36cb37f-c000.csv", inferSchema='true')
    trainset = trainset.withColumnRenamed(trainset.schema.names[0],'text')
    spark_proc_udf = udf(lambda x:word_preprocess(x), ArrayType(StringType()))
    trainset = trainset.withColumn('words', spark_proc_udf(trainset['text']))
    w2v = Word2Vec(vectorSize=100, minCount=1, inputCol="words", outputCol="result").fit(trainset)
    for i in range(epoch):
        print("epoch:" + str(i))
        for j in range(1,16):
            trainset = spark.read.csv(fpath + '/text/' + "part-0000{}-8560ee87-33f5-4665-ae61-d0beb36cb37f-c000.csv".format(j), inferSchema='true')
            trainset = trainset.withColumnRenamed(trainset.schema.names[0],'text')
            spark_proc_udf = udf(lambda x:word_preprocess(x), ArrayType(StringType()))
            trainset = trainset.withColumn('words', spark_proc_udf(trainset['text']))
            startt = time.time()

            w2v = w2v.fit(trainset)
            #trainset.show()
            w2v.getVectors().count()
            endt = time.time()
            print(endt - startt)

def dict_init(path):
    dict = {}
    for i in range(425):
        num =str(i)
        fpath = path + "/text/part-00" + '0'*(3- len(num)) + num + "-8560ee87-33f5-4665-ae61-d0beb36cb37f-c000.csv"
        with open(fpath) as f:
            print(num)
            for line in tqdm(f.readlines()):
                words = word_preprocess(line)
                for w in words:
                    dict.setdefault(w, 0)
                    dict[w] += 1
    print("dict length:{}".format(len(dict)))

    with open(path + "/text/dict.csv", 'w') as f:
        for k,v in dict.items():
            if v>3:
                f.write(str(k) +' '  + str(v) +'\n')

def word_preprocess(strobj):
    if strobj is None:
        return ['-']

    strlist = ' '.join(strobj.split('[SEP]'))
    strlist = ' '.join(strlist.split("\\\""))
    strlist = strlist.split( )
    wordlist = []
    sem_pun = [',', '.', '!', '?', ')',':',';']
    for s in strlist:
        if s[0] == '(':
            wordlist += ['(']
            s = s[1:]
        s.replace("‘", '\'')
        s.replace("’", '\'')
        s.replace("”", '\'')
        s.replace("“", '\'')
        s = s.strip("\"\\'")
        if len(s) == 0:
            wordlist += ['-']
        elif s[-1] in sem_pun and len(s)>=2:
            wordlist += [s[:-1], s[-1]]
        else:
            wordlist += [s]
    return wordlist

def load_dict(dict_path):
    dict2id = {'[UNK]':0, '[MASK]':1, '[PAD]':2, '[EOS]':3}
    with open(dict_path, 'r') as f:
        for line in f.readlines():
            w = line.split()[0]
            dict2id[w] = len(dict2id)
    id2dict = {}
    for k,v in dict2id.items():
        id2dict[v] = k

    return dict2id, id2dict


if __name__ ==  "__main__":
    rel_path = "/home/zhaohaolin/my_project/pyHGT/MyGTable/utils/test"
    
    dict_init(rel_path)



#非语义标点 [",", "\",\"", \"], null -> -
#语义标点 [',' , '.' ]
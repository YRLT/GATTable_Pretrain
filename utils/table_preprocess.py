from google.protobuf import text_format
import interaction_pb2

from pyspark import  SparkContext,SparkConf
from pyspark.sql import SparkSession

import time


def parallel_comp(table_path):
    spark = SparkSession \
        .builder \
        .appName('para_extract') \
        .master("local[*]") \
        .enableHiveSupport() \
        .getOrCreate() 
    # spark.sparkConf.set("spark.driver.host", "localhost")
    spark.sparkContext.addPyFile("/home/zhaohaolin/my_project/pyHGT/MyGTable/utils/interaction_pb2.py")
    table_extract(table_path, spark)
    pass

def str2pb2(x):
    return text_format.Parse(x, interaction_pb2.Interaction())
 
def table_extract(table_path, spark):

    dataRDD = spark.sparkContext.textFile(table_path)
    tablesRDD = dataRDD.map(str2pb2)

    table2GFRDD(tablesRDD)


def table2GFRDD(table):
    time1 = time.time()

    tablelistRDD = table.filter(lambda x:len(x.table.rows)*len(x.table.columns)<3000).zipWithIndex().map(lambda x: table2list(x))
    
    nodeRDD = tablelistRDD.map(lambda x:x[0]).flatMap(lambda x:x).toDF(['id','type','text'])
    edgeRDD = tablelistRDD.map(lambda x:x[1]).flatMap(lambda x:x).toDF(["src", "dst",'type'])
    textRDD = tablelistRDD.map(lambda x:x[0]).flatMap(lambda x:x).map(lambda x:[x[2]]).toDF(['text'])

    savepath = "/home/zhaohaolin/my_project/pyHGT/MyGTable/utils/test"
    savegraph(nodeRDD, edgeRDD, textRDD, savepath, tablelistRDD.count())

    time2 = time.time()
    print("ok")
    print(time2 - time1)

def savegraph(v, e, text, Outpath, countnum):
    #v.write.mode("overwrite").parquet(Outpath + "/node.parquet")
    #e.write.mode("overwrite").parquet(Outpath + "/edge.parquet")
    text.write.csv(Outpath + "/text", mode='overwrite')
    with open(Outpath + "/numid.txt", 'w') as f:
        f.write(str(countnum))
    f.close()

def table2list(inp):
    table = inp[0]
    Gid = inp[1]
    tablelist = []
    edgelist = []

    # eventural node format for heads: ['tableid-0-0', 'VH', '[VH]'] + ['tableid-0-colid', 'H', 'text']
    # edge format for each row: ['tableid-rowid-0', 'tableid-rowid-colid', 'type1-type2']
    col = list(table.table.columns)
    col_num = len(col)

    
    tablelist += [['{}-0-0'.format(Gid), 'VH', '[VH]']] + [['{}-0-{}'.format(Gid,id+1),'H', i.text] for id,i in enumerate(col)]
    edgelist += [['{}-0-0'.format(Gid),'{}-0-{}'.format(Gid,id+1), 'VH-H'] for id,i in enumerate(col)]
    edgelist += [['{}-0-{}'.format(Gid,id+1),'{}-0-0'.format(Gid), 'H-VH'] for id,i in enumerate(col)]
    
    # eventural format for cells: ['tableid-rowid-0', 'E', '[VE]'] + ['tableid-rowid-id', 'Cell', 'text']
    for rowid,i in enumerate(list(table.table.rows)):
        row = i.cells
        tablelist += [['{}-{}-0'.format(Gid,rowid + 1), 'VE', '[VE]']] + \
            [['{}-{}-{}'.format(Gid, rowid + 1,colid + 1), 'Cell', j.text] for colid,j in enumerate(row)]
        edgelist += [['{}-{}-0'.format(Gid,rowid + 1),'{}-{}-{}'.format(Gid, rowid + 1, colid + 1), 'VE-Cell'] for colid,j in enumerate(row)]
        edgelist += [['{}-{}-{}'.format(Gid, rowid + 1, colid + 1), '{}-{}-0'.format(Gid,rowid + 1), 'Cell-VE'] for colid,j in enumerate(row)]

    item_num = len(list(table.table.rows))
    
    # eventural format for caption: ["tableid-rowid+1 - 0", 'Caption', '[ClS]' + '[SEP]'.join(text)]
    tablelist += [['{}-{}-0'.format(Gid, item_num+1), 'Caption', '[SEP]'.join([i.original_text for i in table.questions])]]
    edgelist += [['{}-{}-0'.format(Gid, item_num+1), '{}-0-0'.format(Gid), 'Caption-VH']] + \
        [['{}-{}-0'.format(Gid, item_num+1), '{}-{}-0'.format(Gid, i), 'Caption-VE'] for i in range(1,item_num+1)]
    edgelist += [['{}-0-0'.format(Gid), '{}-{}-0'.format(Gid, item_num+1),  'VH-Caption']] + \
        [['{}-{}-0'.format(Gid, i), '{}-{}-0'.format(Gid, item_num+1),  'VE-Caption'] for i in range(1,item_num+1)]

    for colid in range(1, col_num+1):
        edgelist += [['{}-0-{}'.format(Gid, colid), '{}-{}-{}'.format(Gid, rowid, colid), 'H-Cell'] for rowid in range(1, item_num+1)]
        edgelist += [['{}-{}-{}'.format(Gid, rowid, colid), '{}-0-{}'.format(Gid, colid), 'Cell-H'] for rowid in range(1, item_num+1)]

    return tablelist, edgelist

if __name__ == "__main__":
    parallel_comp("/home/zhaohaolin/my_project/pyHGT/interactions.txtpb")

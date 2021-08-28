from imp import SEARCH_ERROR
from pyspark.sql import SparkSession
from graphframes import *
import time
import numpy as np

def read_graph(filepath: str, spark = None):
    nodepath = filepath + "/node.parquet"
    edgepath = filepath + "/edge.parquet"
    with open(filepath+"/numid.txt", 'r') as f:
        num = int(f.readline())
    print(num)
    if spark == None:
        spark = SparkSession \
            .builder \
            .appName('para_extract') \
            .master("local[8]") \
            .enableHiveSupport() \
            .config("spark.driver.memory", "20g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.executor.cores", "4") \
            .config("spark.driver.maxResultSize", "10g")\
            .getOrCreate()

        spark.sparkContext.setCheckpointDir(dirName="graphframes_cps")
    nodedf = spark.read.parquet(nodepath)
    edgedf = spark.read.parquet(edgepath)

    g = GraphFrame(nodedf, edgedf)
    return g, num

def graph_convert(g, batch):
    rows = g.vertices.collect()
    node_id = {}
    v_text, v_type = [[] for i in range(batch)] , [[] for i in range(batch)]
    for row in rows:
        gid, x, y = row[0].split('-')
        if gid not in node_id.keys():
            node_id[gid] = [{},len(node_id)]
        node_id[gid][0][gid + '-' + x+'-'+y] = len(node_id[gid][0])
        v_text[node_id[gid][1]].append(row[2])
        v_type[node_id[gid][1]].append(row[1]) 

    e_mx = [np.zeros([len(v[0]), len(v[0])]) for k,v in node_id.items()]
    for row in g.edges.collect():
        gid = row[0].split('-')[0]
        e_mx[node_id[gid][1]][node_id[gid][0][row[0]]][node_id[gid][0][row[1]]] = 1

    return v_text, v_type, e_mx

def search_graph(g, g_id, batch = 1):
    v_text, v_type, e_mx = [[]*batch], [[]*batch], [[]*batch]
    if batch == 1:
        nodeid = str(g_id) + '-'
        gnode = g.filterVertices("id LIKE'" + nodeid + "%'")
    else:
        id_seq = range(g_id, g_id + batch-1)
        seach_Expr = ("id LIKE'{}-%' OR " * (batch-1)).format(*id_seq) + " id LIKE'{}-%'".format(g_id + batch -1)
        gnode = g.filterVertices(seach_Expr)

    v_text, v_type, e_mx = graph_convert(gnode, batch)

    return v_text, v_type, e_mx

if __name__ == "__main__":
    g, num = read_graph("/home/zhaohaolin/my_project/pyHGT/MyGTable/utils/test")
    t1 = time.time()
    v_text, v_type, e_mx = search_graph(g, 100, batch = 7)

    t2 = time.time()
    print(t2-t1)
 
    print(v_type)
    pass
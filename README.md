# GATTable_Pretrain

模型目标是处理完成对互联网上海量表格数据的表示学习并针对于语义解析等相关下游任务进行实验

目前完成了预训练部分的基本实现，下一步即将完善模型的GPU化部署代码（数据层问题）和语义解析任务的实现，以及之后的一个关于GAT学习的新猜想的实现和实验

主要涉及的模型及框架：

Transformer, HGAT, MaskedML 

pyspark, pytorch, graphframes

预训练数据的相关信息和下载请移步至TAPAS，格式为protobuf， 大小约为10G左右

https://github.com/google-research/tapas/blob/master/PRETRAIN_DATA.md

==================================================================================

在数据预处理阶段完成文本到图的格式转化，预训练时从图的存储数据中进行数据喂入

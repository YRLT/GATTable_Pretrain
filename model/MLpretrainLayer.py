import torch
import torch.nn as nn
from model.GAT import HGAT_multi as GLayer
from model.textLayer import textLayer
import copy


class GATMLModelfortable(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.textLayer = textLayer(config)
        self.GATlayer = GLayer(config)
        self.pretrain = config.TRAIN.pretrain
        self.mask_texttoken = config.DATA.maskfortext
        
        if self.pretrain:
            self.Mask_rate = config.TRAIN.mask_rate
            self.softmax = nn.LogSoftmax(dim = 1)


    def forward(self, input): 
        # input = [v_text.tokenized, v_type.tokenized, e_mx]
        # v_type('VH', 'H','VE','Cell','Caption','MASK')
        # e_type('VH-H, VE-CELL, VE-VH, H-CELL, C-VE, C-VH)
        graph_size = len(input[0])
        label = torch.tensor(range(graph_size))
        

        textembedding = self.textLayer(input[0])

        GEmbedding = self.GATlayer(textembedding, input[2]) 

        if self.pretrain: 
            input_MASK,label_mask = self.MLM(input, copy.deepcopy(label))                                           
            textembedding_MASK = self.textLayer(input_MASK[0])

            mask_adj = self.mask_adj(input_MASK[1], input[2]) #输入节点类型list，第一维邻接矩阵
            GEmbedding_MASK = self.GATlayer(textembedding_MASK.detach(), mask_adj)


            softmax =  self.softmax(torch.mm(GEmbedding.detach(), GEmbedding_MASK.T)) #两矩阵相乘
            label_mask = torch.argmax(softmax, dim = 0)
            
            if label.size()!= label_mask.size():
                print(label, label_mask)
                return None
            return [label, softmax]
            
        else:
            return GEmbedding

    def MLM(self, input, label):
        ntext = input[0]
        ntype = input[1]
        mask_label = label.clone()
        masked_indices = torch.bernoulli(torch.full([label.size(0)], self.Mask_rate)).bool()

        mask_label[~masked_indices] = -1
        indices_replaced = torch.bernoulli(torch.full([label.size(0)], 0.8)).bool() & masked_indices

        ntext[indices_replaced] = self.mask_texttoken
        ntype[indices_replaced] = -1
        return [ntext, ntype, input[2]],mask_label
    
    def mask_adj(self, type_list, adj):
        mask_adj = torch.zeros(len(type_list),len(type_list))

        e_mx = torch.sum(adj, dim = 0)

        for i in range(len(type_list)):
            if type_list[i] == -1:
                mask_adj[i, :] = 1
                mask_adj[:, i] = 1

        mask_adj = torch.where(torch.add(e_mx,mask_adj)>1, mask_adj, torch.zeros(len(type_list), len(type_list)))
        mask_list = [mask_adj.clone().unsqueeze(0) for i in range(6)]
        filter_mx = torch.cat(mask_list, dim = 0)

        mask_adj = torch.where(torch.add(filter_mx, adj) >= 1, filter_mx, adj )
        

        return mask_adj

        
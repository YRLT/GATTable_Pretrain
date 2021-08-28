from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from transformer.Models import Transformer
import torch


class DealDataset(Dataset):
    def __init__(self, nparr):

        self.x = torch.from_numpy(nparr)
        self.len = self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index]
 
    def __len__(self):
        return self.len

class textLayer(nn.Module):
    def __init__(self, config):
        super(textLayer, self).__init__()
        self.batch_size = config.TRANS.batch_size
        self.transformers = Transformer(
                config.TRANS.src_vocab_size, #source  vocab
                src_pad_idx=config.TRANS.src_pad_idx, #pad_id
                d_k=config.TRANS.d_k,
                d_v=config.TRANS.d_v,
                lq = config.TRANS.lq, #句子长度（包含pad_mask）
                d_model=config.TRANS.d_model, #输出维度
                d_word_vec=config.TRANS.d_word_vec, #dim_word
                d_inner=config.TRANS.d_inner_hid, #隐藏层维度
                n_layers=config.TRANS.n_layers,
                n_head=config.TRANS.n_head,
                dropout=config.TRANS.dropout,
                scale_emb_or_prj=config.TRANS.scale_emb_or_prj
            ) 

    def forward(self, text):
        data = DealDataset(text) #句子token_seqs
        train_data = DataLoader(dataset = data, batch_size = self.batch_size, shuffle=False)
        output = []
        for _,data in enumerate(train_data):
            output += [self.transformers(data)]
        
        return torch.cat(output, dim = 0)
        

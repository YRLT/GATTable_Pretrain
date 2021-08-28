import pathlib

from os import path
from yacs.config import CfgNode as CN

root = pathlib.Path(__file__).parent.parent.absolute()

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 6

_C.DATA = CN()
_C.DATA.save_dir = "/home/zhaohaolin/my_project/pyHGT/MyGTable/data"
_C.DATA.dicpath = "/home/zhaohaolin/my_project/pyHGT/MyGTable/data/dict.csv"
_C.DATA.dfgraph = "/home/zhaohaolin/my_project/pyHGT/MyGTable/utils/test"
_C.DATA.maskfortext = 1


_C.TRAIN = CN()
_C.TRAIN.batch_size = 1
_C.TRAIN.save_dir = path.join(root, 'model/outputs')
_C.TRAIN.gradient_accumulation_steps = 1
_C.TRAIN.num_train_epochs = 5
_C.TRAIN.mask_rate = 0.1
_C.TRAIN.learning_rate = 2*1e-5
_C.TRAIN.weight_decay = 0.0
_C.TRAIN.device = 'cpu'
_C.TRAIN.log_steps = 200
_C.TRAIN.adam_epsilon = 1e-6
_C.TRAIN.shuffle = True
_C.TRAIN.pretrain = True



_C.TRANS = CN()
_C.TRANS.batch_size = 32
_C.TRANS.src_vocab_size = 4061237 #字典长度
_C.TRANS.src_pad_idx = 2 
_C.TRANS.d_k = 64
_C.TRANS.d_v = 64
_C.TRANS.lq = 128
_C.TRANS.d_model = 512
_C.TRANS.d_inner_hid = 512
_C.TRANS.d_word_vec = 512
_C.TRANS.n_layers = 2
_C.TRANS.n_head = 8
_C.TRANS.dropout = 0.1
_C.TRANS.scale_emb_or_prj = 'prj'



_C.GAT = CN()
_C.GAT.outputsz = 100
_C.GAT.n_heads = 6
_C.GAT.ffd_drop = 0.1
_C.GAT.attn_drop = 0.1


def get_cfg_defaults(merge_from=None):
    cfg = _C.clone()
    if merge_from is not None:
        cfg.merge_from_other_cfg(merge_from)
    return cfg
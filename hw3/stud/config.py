import os
import sys
sys.path.append('..')
from load import parseRelation2Id, parseRel2Pred



DEBUG = 0


# PATHS
DATA_PATH = 'data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train.jsonl')
DEV_PATH = os.path.join(DATA_PATH, 'dev.jsonl')
TEST_PATH = os.path.join(DATA_PATH, 'test.jsonl')
RELATION2ID_PATH = os.path.join('hw3/', 'relations2id.json')
REL2PRED_PATH = os.path.join('hw3/', 'rel2pred.json')
CKPT_PATH = 'UniRel/'
LOAD_MODEL_PATH = 'model/' + 'UniRel-epoch=08-val_loss=0.00112.ckpt'#'UniRel-epoch=05-val_loss=0.0011.ckpt'



PRETRAINED_MODEL = 'bert-base-cased'
MAX_LEN = 180
BATCH_SIZE = 32
EPOCHS = 10
THRESHOLD = 0.5
LR = 3e-5
WEIGHT_DECAY = 0.01
BIDIRECTIONAL = True
SEED = 42


relation2Id = parseRelation2Id(RELATION2ID_PATH)
rel2pred = parseRel2Pred(REL2PRED_PATH)
id2relation = {v:k for k,v in relation2Id.items()}
encoded_preds = []
# rel_abrv = " ".join([rel.split('/')[-1] for rel in list(relation2Id.keys())])
REL_NUM = len(relation2Id)
index_shift = {}


# TABLE_SIZE = (MAX_LEN + REL_NUM)**2
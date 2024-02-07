import os
import sys
sys.path.append('..')
from load import parseRelation2Id, parseRel2Pred






# PATHS
DATA_PATH = 'data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train.jsonl')
DEV_PATH = os.path.join(DATA_PATH, 'dev.jsonl')
TEST_PATH = os.path.join(DATA_PATH, 'test.jsonl')
RELATION2ID_PATH = os.path.join(DATA_PATH, 'relations2id.json')
REL2PRED_PATH = os.path.join(DATA_PATH, 'rel2pred.json')
CKPT_PATH = 'UniRel/'



PRETRAINED_MODEL = 'bert-base-cased'
# MAX_LEN = 180
BATCH_SIZE = 32
EPOCHs = 10
THRESHOLD = 0.5
LR = 3e-5

SEED = 42


relation2Id = parseRelation2Id(RELATION2ID_PATH)
rel2pred = parseRel2Pred(REL2PRED_PATH)
id2relation = {v:k for k,v in relation2Id.items()}
encoded_preds = []
# rel_abrv = " ".join([rel.split('/')[-1] for rel in list(relation2Id.keys())])
REL_NUM = len(relation2Id)
index_shift = {}


# TABLE_SIZE = (MAX_LEN + REL_NUM)**2
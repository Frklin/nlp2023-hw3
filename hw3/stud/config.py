import os
import sys
sys.path.append('..')
from load import parseRelation2Id, parseRel2Pred



DEBUG = 0
TEST_MODE = 0


# PATHS
DATA_PATH = 'data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train.jsonl')
DEV_PATH = os.path.join(DATA_PATH, 'dev.jsonl')
TEST_PATH = os.path.join(DATA_PATH, 'test.jsonl')
RELATION2ID_PATH = os.path.join(DATA_PATH, 'relations2id.json')
REL2PRED_PATH = os.path.join(DATA_PATH, 'rel2pred.json')
CKPT_PATH = 'UniRel/'
LOAD_MODEL_PATH = CKPT_PATH + 'UniRel-epoch=01-val_loss=0.01.ckpt'



PRETRAINED_MODEL = 'bert-base-cased'
BATCH_SIZE = 32
EPOCHS = 10
THRESHOLD = 0.5
LR = 3e-5
WEIGHT_DECAY = 0.01

SEED = 42
BIDIRECTIONAL = True


relation2Id = parseRelation2Id(RELATION2ID_PATH)
rel2pred = parseRel2Pred(REL2PRED_PATH)
id2relation = {v:k for k,v in relation2Id.items()}
encoded_preds = []
REL_NUM = len(relation2Id)
index_shift = {}



import os
import sys
sys.path.append('..')
from load import parseRelation2Id 






# PATHS
DATA_PATH = '../../data'
TRAIN_PATH = os.path.join(DATA_PATH, 'train.jsonl')
DEV_PATH = os.path.join(DATA_PATH, 'dev.jsonl')
TEST_PATH = os.path.join(DATA_PATH, 'test.jsonl')
RELATION2ID_PATH = os.path.join(DATA_PATH, 'relations2id.json')




PRETRAINED_MODEL = 'bert-base-uncased'
MAX_LEN = 102
BATCH_SIZE = 8
EPOCHs = 10

SEED = 42







relation2Id = parseRelation2Id(RELATION2ID_PATH)
rel_abrv = [rel.split('/')[-1] for rel in list(relation2Id.keys())]
REL_NUM = len(relation2Id)
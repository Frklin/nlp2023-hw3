import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoTokenizer
from pytorch_lightning import Trainer

sys.path.append("../")
import config
from utils import seed_everything, collate_fn
from load import RelationDataset
from UniRel import UniRE
from transformers import BertConfig
from pytorch_lightning.callbacks import ModelCheckpoint




if __name__ == '__main__':
    seed_everything(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)

    train_data = RelationDataset(config.TRAIN_PATH, tokenizer)
    dev_data = RelationDataset(config.DEV_PATH, tokenizer)
    test_data = RelationDataset(config.TEST_PATH, tokenizer)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    bert_config = BertConfig.from_pretrained(config.PRETRAINED_MODEL,
                                            finetuning_task="UniRel")
    bert_config.num_rels = config.REL_NUM

    model = UniRE(bert_config=bert_config)


    trainer = Trainer(gpus=1, max_epochs=100, callbacks=[ModelCheckpoint(monitor='val_loss')])#, logger=wandb_logger)
    trainer.fit(model, train_loader, dev_loader)
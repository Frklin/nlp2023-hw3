import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
import sys
sys.path.append("../")

from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from pytorch_lightning import Trainer

import config
from utils import seed_everything, collate_fn
from load import RelationDataset
from UniRel import UniRE
from transformers import BertConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger



if __name__ == '__main__':
    seed_everything(config.SEED)
    # added_token = [f"[unused{i}]" for i in range(1, 17)] # to removce
    # tokenizer = BertTokenizerFast.from_pretrained(config.PRETRAINED_MODEL, additional_special_tokens=added_token, do_basic_tokenize=True)
    tokenizer = BertTokenizerFast.from_pretrained(config.PRETRAINED_MODEL, do_basic_tokenize=True)

    train_data = RelationDataset(config.TRAIN_PATH, tokenizer)
    dev_data = RelationDataset(config.DEV_PATH, tokenizer)
    test_data = RelationDataset(config.TEST_PATH, tokenizer)

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    

    bert_config = BertConfig.from_pretrained(config.PRETRAINED_MODEL,
                                            finetuning_task="UniRel")
    bert_config.num_rels = config.REL_NUM
    bert_config.num_labels = config.REL_NUM
    bert_config.threshold = config.THRESHOLD
    config.is_additional_att = False
    config.is_separate_ablation = False
    config.test_data_type = False

    if config.TEST_MODE:
        model = UniRE.load_from_checkpoint(config.LOAD_MODEL_PATH, config=bert_config)
        model.eval()
        print("TEST MODE:")
        trainer = Trainer(accelerator="gpu", devices=1)
        trainer.test(model, test_loader)
        exit()


    model = UniRE(config=bert_config) 
    model.resize_token_embeddings(len(tokenizer))

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.CKPT_PATH,
        filename='UniRel-{epoch:02d}-{val_loss:.4f}',
        verbose=True,
        save_top_k=2,
        mode='min',
    )


    wandb_logger = WandbLogger(name='Run #1', project='UniRel')
    trainer = Trainer(max_epochs=100, callbacks=checkpoint_callback, logger=wandb_logger, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader, dev_loader) 
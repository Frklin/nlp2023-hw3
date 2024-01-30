import pytorch_lightning as pl
from transformers import AutoModel

import config




class UniRE(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODEL)


    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def training_epoch_end(self, outputs):
        pass
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel
from modify_bert import BertModel


import config




class UniRE(pl.LightningModule):

    def __init__(self, bert_config):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.PRETRAINED_MODEL, config=bert_config)

        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()


    def forward(self, input_ids, attention_mask, token_type_ids):
        
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions_scores=True)

        attention_scores = out.attentions_scores[-1]

        h_logits = self.sigmoid(attention_scores[:, :4, :, :].mean(1))
        t_logits = self.sigmoid(attention_scores[:, 4:8, :, :].mean(1))
        span_logits = self.sigmoid(attention_scores[:, 8:, :, :].mean(1))

        return h_logits, t_logits, span_logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)

        h_loss = self.loss(h_logits.float().view(-1), labels["head_matrices"].view(-1).float())
        t_loss = self.loss(t_logits.float().view(-1), labels["tail_matrices"].view(-1).float())
        span_loss = self.loss(span_logits.float().view(-1), labels["span_matrices"].view(-1).float())

        loss = h_loss + t_loss + span_loss

        h_pred = h_logits > config.THRESHOLD
        t_pred = t_logits > config.THRESHOLD
        span_pred = span_logits > config.THRESHOLD

        h_correct = (h_pred == labels["head_matrices"]).sum().item()
        t_correct = (t_pred == labels["tail_matrices"]).sum().item()
        span_correct = (span_pred == labels["span_matrices"]).sum().item()

        h_acc = h_correct / (config.MAX_LEN * config.MAX_LEN)
        t_acc = t_correct / (config.MAX_LEN * config.MAX_LEN)
        span_acc = span_correct / (config.MAX_LEN * config.MAX_LEN)

        print(f"Train Loss: {loss}, H Acc: {h_acc}, T Acc: {t_acc}, Span Acc: {span_acc}")

        return {
            "loss": loss,
            "h_acc": h_acc,
            "t_acc": t_acc,
            "span_acc": span_acc
        }




    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)

        h_loss = self.loss(h_logits.float().view(-1), labels["head_matrices"].view(-1).float())
        t_loss = self.loss(t_logits.float().view(-1), labels["tail_matrices"].view(-1).float())
        span_loss = self.loss(span_logits.float().view(-1), labels["span_matrices"].view(-1).float())

        loss = h_loss + t_loss + span_loss

        h_pred = h_logits > config.THRESHOLD
        t_pred = t_logits > config.THRESHOLD
        span_pred = span_logits > config.THRESHOLD

        h_correct = (h_pred == labels["head_matrices"]).sum().item()
        t_correct = (t_pred == labels["tail_matrices"]).sum().item()
        span_correct = (span_pred == labels["span_matrices"]).sum().item()

        h_acc = h_correct / (config.MAX_LEN * config.MAX_LEN)
        t_acc = t_correct / (config.MAX_LEN * config.MAX_LEN)
        span_acc = span_correct / (config.MAX_LEN * config.MAX_LEN)

        print(f"Val Loss: {loss}, H Acc: {h_acc}, T Acc: {t_acc}, Span Acc: {span_acc}")

        return {
            "loss": loss,
            "h_acc": h_acc,
            "t_acc": t_acc,
            "span_acc": span_acc
        }
    
    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.LR)

    def training_epoch_end(self, outputs):
        pass
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel
from modify_bert import BertModel
from metrics import compute_metrics
from utils import reconstruct_relations_from_matrices


import config




class UniRE(pl.LightningModule):

    def __init__(self, bert_config):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.PRETRAINED_MODEL, config=bert_config)

        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()

        self.train_labels = []
        self.train_preds = []
        self.val_labels = []
        self.val_preds = []


    def forward(self, input_ids, attention_mask, token_type_ids):
        
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions_scores=True)

        attention_scores = out.attentions_scores[-1]

        h_logits = self.sigmoid(attention_scores[:, :4, :, :].mean(1))
        t_logits = self.sigmoid(attention_scores[:, 4:8, :, :].mean(1))
        span_logits = self.sigmoid(attention_scores[:, 8:, :, :].mean(1))

        return h_logits, t_logits, span_logits

    def training_step(self, batch, batch_idx):
        indices, input_ids, attention_mask, token_type_ids, labels = batch

        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)
        
        # h_logits = h_logits[attention_mask == 1]

        h_loss = self.loss(h_logits.float().view(-1), labels["head_matrices"].view(-1).float())
        t_loss = self.loss(t_logits.float().view(-1), labels["tail_matrices"].view(-1).float())
        span_loss = self.loss(span_logits.float().view(-1), labels["span_matrices"].view(-1).float())

        loss = h_loss + t_loss + span_loss

        h_pred = h_logits > config.THRESHOLD
        t_pred = t_logits > config.THRESHOLD
        span_pred = span_logits > config.THRESHOLD

        # reconstruct prediction
        preds = reconstruct_relations_from_matrices(h_pred, t_pred, span_pred)

        # h_correct = (h_pred == labels["head_matrices"]).sum().item()
        # t_correct = (t_pred == labels["tail_matrices"]).sum().item()
        # span_correct = (span_pred == labels["span_matrices"]).sum().item()


        # h_wrong = (h_pred != labels["head_matrices"]).sum().item()
        # t_wrong = (t_pred != labels["tail_matrices"]).sum().item()
        # span_wrong = (span_pred != labels["span_matrices"]).sum().item()

        # acc, prec, rec, f1 = compute_metrics(input_ids, h_pred, t_pred, span_pred, labels)

        self.log("train_loss", loss, prog_bar = True)
        # self.log("accuracy", acc, prog_bar = True)
        # self.log("precision", prec, prog_bar = True)
        # self.log("recall", rec, prog_bar = True)
        # self.log("f1_score", f1, prog_bar = True)
        # self.log("train_h_acc", h_acc, prog_bar = True)
        # self.log("train_t_acc", t_acc, prog_bar = True)
        # self.log("train_span_acc", span_acc, prog_bar = True)
        # self.log("h_correct", h_correct, prog_bar = True)
        # self.log("h_wrong", h_wrong, prog_bar = True)
        # self.log("attention_mask_sum", attention_mask.sum().item(), prog_bar = True)
        

        return {
            "loss": loss,
            "preds": preds,
            "labels": labels["spo"]
        }




    def validation_step(self, batch, batch_idx):
        indices, input_ids, attention_mask, token_type_ids, labels = batch

        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)

        h_loss = self.loss(h_logits.float().view(-1), labels["head_matrices"].view(-1).float())
        t_loss = self.loss(t_logits.float().view(-1), labels["tail_matrices"].view(-1).float())
        span_loss = self.loss(span_logits.float().view(-1), labels["span_matrices"].view(-1).float())

        loss = h_loss + t_loss + span_loss

        h_pred = h_logits > config.THRESHOLD
        t_pred = t_logits > config.THRESHOLD
        span_pred = span_logits > config.THRESHOLD

        preds = reconstruct_relations_from_matrices(h_pred, t_pred, span_pred)
       
        self.log("val_loss", loss, prog_bar = True)

        return {
            "loss": loss,
            "preds": preds,
            "labels": labels["spo"]
        }
    


    def test_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, outputs):
        preds = []
        labels = []
        losses = []

        for output in outputs:
            preds.append(output["preds"])
            labels.append(output["labels"])
            losses.append(output["loss"])

        acc, prec, rec, f1 = compute_metrics(preds, labels)

        loss = torch.stack(losses).mean()

        print(f"Epoch {self.current_epoch} (TRAIN): Loss: {loss}, train accuracy: {acc}, precision: {prec}, recall: {rec}, f1_score: {f1}")
        self.log("train_accuracy", acc)
        self.log("train_precision", prec)
        self.log("train_recall", rec)
        self.log("train_f1_score", f1)


    def validation_epoch_end(self, outputs):
        preds = []
        labels = []
        losses = []

        for output in outputs:
            preds.append(output["preds"])
            labels.append(output["labels"])

        acc, prec, rec, f1 = compute_metrics(preds, labels)

        loss = torch.stack(losses).mean()

        print(f"Epoch {self.current_epoch} (VAL): Loss: {loss}, Val accuracy: {acc}, precision: {prec}, recall: {rec}, f1_score: {f1}")
        self.log("val_accuracy", acc)
        self.log("val_precision", prec)
        self.log("val_recall", rec)
        self.log("val_f1_score", f1)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config.LR)

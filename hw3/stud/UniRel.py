import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel, BertPreTrainedModel
from modify_bert import BertModel
from metrics import compute_metrics
from utils import reconstruct_relations_from_matrices


import config as cfg




class UniRE(BertPreTrainedModel, pl.LightningModule):

    def __init__(self, config):
        super(UniRE, self).__init__(config=config)

        self.bert = BertModel.from_pretrained(cfg.PRETRAINED_MODEL, config=config)

        # for param in self.bert.parameters():
        #     param.requires_grad = True

        self.unfreezed_layers = 0

        # self.unfreeze_bert()

        self.epoch_num = 0

        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()
        # self.loss = nn.CrossEntropyLoss(ignore_index=0)

        self.train_h_preds = []
        self.train_t_preds = []
        self.train_span_preds = []
        self.train_labels = []
        self.train_losses = []

        self.val_h_preds = []
        self.val_t_preds = []
        self.val_span_preds = []
        self.val_labels = []
        self.val_losses = []

    def unfreeze_bert(self):
        self.unfreezed_layers += 1
        for param in self.bert.encoder.layer[-self.unfreezed_layers:].parameters():
            param.requires_grad = True


    def forward(self, input_ids, attention_mask, token_type_ids):
        
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=False, output_attentions_scores=True)

        attention_scores = out.attentions_scores[-1]

        h_logits = self.sigmoid(attention_scores[:, :4, :, :].mean(1))
        t_logits = self.sigmoid(attention_scores[:, 4:8, :, :].mean(1))
        span_logits = self.sigmoid(attention_scores[:, 8:, :, :].mean(1))

        return h_logits, t_logits, span_logits

    def training_step(self, batch, batch_idx):
        indices, input_ids, attention_mask, token_type_ids, labels = batch

        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)
        
        h_pred = h_logits > cfg.THRESHOLD
        t_pred = t_logits > cfg.THRESHOLD
        span_pred = span_logits > cfg.THRESHOLD

        h_loss = self.loss(h_logits.float().reshape(-1), labels["head_matrices"].reshape(-1).float())
        t_loss = self.loss(t_logits.float().reshape(-1), labels["tail_matrices"].reshape(-1).float())
        span_loss = self.loss(span_logits.float().reshape(-1), labels["span_matrices"].reshape(-1).float())


        # compute how many ones of the labels are predicted as ones
        labels_head_ones_indices = set(tuple(el) for el in labels["head_matrices"].nonzero().tolist())
        labels_tail_ones_indices = set(tuple(el) for el in labels["tail_matrices"].nonzero().tolist())
        labels_span_ones_indices = set(tuple(el) for el in labels["span_matrices"].nonzero().tolist())

        h_pred_ones_indices = set(tuple(el) for el in h_pred.nonzero().tolist())
        t_pred_ones_indices = set(tuple(el) for el in t_pred.nonzero().tolist())
        span_pred_ones_indices = set(tuple(el) for el in span_pred.nonzero().tolist())

        h_pred_percentage = len(h_pred_ones_indices.intersection(labels_head_ones_indices)) / len(labels_head_ones_indices)
        t_pred_percentage = len(t_pred_ones_indices.intersection(labels_tail_ones_indices)) / len(labels_tail_ones_indices)
        span_pred_percentage = len(span_pred_ones_indices.intersection(labels_span_ones_indices)) / len(labels_span_ones_indices)

        # if(len(set(h_pred_ones_indices).intersection(set(labels_head_ones_indices))) > 0):
        #     print("FOUND AT LEAST ONE")
        loss = (h_loss + t_loss + span_loss) #+ \
                # (h_logits.sum() - labels["head_matrices"].sum()).abs() + \
                # (t_logits.sum() - labels["tail_matrices"].sum()).abs() + \
                # (span_logits.sum() - labels["span_matrices"].sum()).abs()

        self.train_h_preds.extend(h_pred)
        self.train_t_preds.extend(t_pred)
        self.train_span_preds.extend(span_pred)

        self.train_labels.extend(labels["spo"])
        self.train_losses.append(loss)

        # h_wrong = (h_pred != labels["head_matrices"]).sum().item()
        # t_wrong = (t_pred != labels["tail_matrices"]).sum().item()
        # span_wrong = (span_pred != labels["span_matrices"]).sum().item()

        self.log("train_loss", loss, prog_bar = True)
        self.log("h_%", h_pred_percentage, prog_bar = True)
        self.log("t_%", t_pred_percentage, prog_bar = True)
        self.log("span_%", span_pred_percentage, prog_bar = True)

        return {
            "loss": loss,
            # "preds": preds,
            # "labels": labels["spo"]
        }




    def validation_step(self, batch, batch_idx):
        indices, input_ids, attention_mask, token_type_ids, labels = batch

        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)

        h_loss = self.loss(h_logits.float().view(-1), labels["head_matrices"].view(-1).float())
        t_loss = self.loss(t_logits.float().view(-1), labels["tail_matrices"].view(-1).float())
        span_loss = self.loss(span_logits.float().view(-1), labels["span_matrices"].view(-1).float())

        h_pred = h_logits > cfg.THRESHOLD
        t_pred = t_logits > cfg.THRESHOLD
        span_pred = span_logits > cfg.THRESHOLD

        loss = (h_loss + t_loss + span_loss)# + \
                # (h_logits.sum() - labels["head_matrices"].sum()).abs() + \
                # (t_logits.sum() - labels["tail_matrices"].sum()).abs() + \
                # (span_logits.sum() - labels["span_matrices"].sum()).abs()

        self.val_h_preds.extend(h_pred)
        self.val_t_preds.extend(t_pred)
        self.val_span_preds.extend(span_pred)
        self.val_labels.extend(labels["spo"])
        self.val_losses.append(loss)
       
        self.log("val_loss", loss, prog_bar = True)

        return {
            "loss": loss,
            # "labels": labels["spo"]
        }
    


    def test_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):

        self.epoch_num += 1


        h_preds = torch.stack(self.train_h_preds)
        t_preds = torch.stack(self.train_t_preds)
        span_preds = torch.stack(self.train_span_preds)
        labels = self.train_labels
        losses = self.train_losses

        if self.epoch_num > 0:
            preds = reconstruct_relations_from_matrices(h_preds, t_preds, span_preds, labels=labels)

            acc, prec, rec, f1 = compute_metrics(preds, labels)

            if prec > 0.0:
                print(prec)

            loss = torch.stack(losses).mean()
            print()
            print(f"Epoch {self.epoch_num} (TRAIN): Loss: {loss}, train accuracy: {acc}, precision: {prec}, recall: {rec}, f1_score: {f1}")
            self.log("train_accuracy", acc)
            self.log("train_precision", prec)
            self.log("train_recall", rec)
            self.log("train_f1_score", f1)


        self.train_h_preds = []
        self.train_t_preds = []
        self.train_span_preds = []
        self.train_labels = []
        self.train_losses = []



    def on_validation_epoch_end(self):
        h_preds = torch.stack(self.val_h_preds)
        t_preds = torch.stack(self.val_t_preds)
        span_preds = torch.stack(self.val_span_preds)
        labels = self.val_labels
        losses = self.val_losses

        # preds = reconstruct_relations_from_matrices(h_preds, t_preds, span_preds, labels=labels)

        # acc, prec, rec, f1 = compute_metrics(preds, labels)

        # if prec > 0.0:
        #     print(prec)

        # loss = torch.stack(losses).mean()

        # print(f"Epoch {self.epoch_num} (VAL): Loss: {loss}, val accuracy: {acc}, precision: {prec}, recall: {rec}, f1_score: {f1}")
        # self.log("val_accuracy", acc)
        # self.log("val_precision", prec)
        # self.log("val_recall", rec)
        # self.log("val_f1_score", f1)

        self.val_h_preds = []
        self.val_t_preds = []
        self.val_span_preds = []
        self.val_labels = []
        self.val_losses = []



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.LR)

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel, BertPreTrainedModel
from modify_bert import BertModel
from metrics import compute_metrics
from utils import reconstruct_relations_from_matrices
import wandb

import config as cfg




class UniRE(BertPreTrainedModel, pl.LightningModule):

    def __init__(self, config):
        super(UniRE, self).__init__(config=config)

        self.bert = BertModel.from_pretrained(cfg.PRETRAINED_MODEL, config=config)

        self.epoch_num = 0

        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()

        self.train_h_preds = []
        self.train_t_preds = []
        self.train_span_preds = []
        self.train_labels = []
        self.train_losses = []
        self.train_h_CM = []
        self.train_t_CM = []
        self.train_span_CM = []

        self.val_h_preds = []
        self.val_t_preds = []
        self.val_span_preds = []
        self.val_labels = []
        self.val_losses = []
        self.val_h_CM = []
        self.val_t_CM = []
        self.val_span_CM = []

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
        h_TP, h_FP, h_FN = self.compute_confusion_matrix(h_pred, labels["head_matrices"])
        t_TP, t_FP, t_FN = self.compute_confusion_matrix(t_pred, labels["tail_matrices"])
        span_TP, span_FP, span_FN = self.compute_confusion_matrix(span_pred, labels["span_matrices"])



        loss = (h_loss + t_loss + span_loss)

        self.train_losses.append(loss)
        self.train_h_preds.extend(h_pred)
        self.train_t_preds.extend(t_pred)
        self.train_span_preds.extend(span_pred)
        self.train_labels.extend(labels["spo"])
        self.train_h_CM.append((h_TP, h_FP, h_FN))
        self.train_t_CM.append((t_TP, t_FP, t_FN))
        self.train_span_CM.append((span_TP, span_FP, span_FN))

        
        self.log("train_loss", loss, prog_bar = True, on_step=False, on_epoch=True)
        wandb.log({"h_TP_train": h_TP, "h_FP_train": h_FP, "h_FN_train": h_FN, "t_TP_train": t_TP, "t_FP_train": t_FP, "t_FN_train": t_FN, "span_TP_train": span_TP, "span_FP_train": span_FP, "span_FN_train": span_FN})

        return {
            "loss": loss
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

        h_TP, h_FP, h_FN = self.compute_confusion_matrix(h_pred, labels["head_matrices"])
        t_TP, t_FP, t_FN = self.compute_confusion_matrix(t_pred, labels["tail_matrices"])
        span_TP, span_FP, span_FN = self.compute_confusion_matrix(span_pred, labels["span_matrices"])


        loss = (h_loss + t_loss + span_loss)

        self.val_losses.append(loss)
        self.val_h_preds.extend(h_pred)
        self.val_t_preds.extend(t_pred)
        self.val_span_preds.extend(span_pred)
        self.val_labels.extend(labels["spo"])
        self.val_h_CM.append((h_TP*len(labels["head_matrices"].nonzero()), len(labels["head_matrices"].nonzero())))
        self.val_t_CM.append((t_TP*len(labels["tail_matrices"].nonzero()), len(labels["tail_matrices"].nonzero())))
        self.val_span_CM.append((span_TP*len(labels["span_matrices"].nonzero()), len(labels["span_matrices"].nonzero())))
       
        self.log("val_loss", loss, prog_bar = True, on_step=False, on_epoch=True)
        wandb.log({"h_TP_val": h_TP, "h_FP_val": h_FP, "h_FN_val": h_FN, "t_TP_val": t_TP, "t_FP_val": t_FP, "t_FN_val": t_FN, "span_TP_val": span_TP, "span_FP_val": span_FP, "span_FN_val": span_FN})

        return {
            "loss": loss,
        }
    


    def test_step(self, batch, batch_idx):
        pass

    def on_train_epoch_end(self):

        self.epoch_num += 1


        losses = self.train_losses
        h_preds = torch.stack(self.train_h_preds)
        t_preds = torch.stack(self.train_t_preds)
        span_preds = torch.stack(self.train_span_preds)
        labels = self.train_labels
        # h_CM_tot = [item[0] for item in self.train_h_CM] 
        # t_CM = self.train_t_CM
        # span_CM = self.train_span_CM

        preds = reconstruct_relations_from_matrices(h_preds, t_preds, span_preds, labels=labels)

        acc, prec, rec, f1 = compute_metrics(preds, labels)


        loss = torch.stack(losses).mean()
        print(f"Epoch {self.epoch_num} (TRAIN): Loss: {loss}, train accuracy: {acc}, precision: {prec}, recall: {rec}, f1_score: {f1}")
        self.log("train_accuracy", acc)
        self.log("train_precision", prec)
        self.log("train_recall", rec)
        self.log("train_f1_score", f1)
        self.log("epoch_train_loss", loss)
        # wandb.log({"train_accuracy": acc, "train_precision": prec, "train_recall": rec, "train_f1_score": f1, "train_loss": loss})
        

        self.train_h_preds = []
        self.train_t_preds = []
        self.train_span_preds = []
        self.train_labels = []
        self.train_losses = []
        self.train_h_CM = []
        self.train_t_CM = []
        self.train_span_CM = []



    def on_validation_epoch_end(self):
        h_preds = torch.stack(self.val_h_preds)
        t_preds = torch.stack(self.val_t_preds)
        span_preds = torch.stack(self.val_span_preds)
        labels = self.val_labels
        losses = self.val_losses

        _, _, _ = self.matrix_precision(self.val_h_CM, "h"), self.matrix_precision(self.val_t_CM, "t"), self.matrix_precision(self.val_span_CM, "span")

        preds = reconstruct_relations_from_matrices(h_preds, t_preds, span_preds, labels=labels)

        acc, prec, rec, f1 = compute_metrics(preds, labels)

        loss = torch.stack(losses).mean()

        # wandb.log({"val_accuracy": acc, "val_precision": prec, "val_recall": rec, "val_f1_score": f1, "val_loss": loss})

        print(f"Epoch {self.epoch_num} (VAL): Loss: {loss}, val accuracy: {acc}, precision: {prec}, recall: {rec}, f1_score: {f1}")
        self.log("val_accuracy", acc)
        self.log("val_precision", prec)
        self.log("val_recall", rec)
        self.log("val_f1_score", f1)
        self.log("epoch_val_loss", loss)

        self.val_h_preds = []
        self.val_t_preds = []
        self.val_span_preds = []
        self.val_labels = []
        self.val_losses = []
        self.val_h_CM = []
        self.val_t_CM = []
        self.val_span_CM = []




    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.LR)
    

    def compute_confusion_matrix(self, preds, labels):
        labels_one = set(tuple(el) for el in labels.nonzero().tolist())
        preds_one = set(tuple(el) for el in preds.nonzero().tolist())
        intersection = set(labels_one).intersection(set(preds_one))

        TPR = len(intersection) / len(labels_one) if len(labels_one) > 0 else 0

        FPR = len(preds_one - intersection) / len(preds_one) if len(preds_one) > 0 else 0 # Da CORREGGERE

        FNR = len(labels_one - intersection) / len(labels_one)

        return TPR, FPR, FNR
        
    def matrix_precision(self, matrix, matrix_type):
        n_labels = 0
        correct = 0
        for tp, n in matrix:
            correct += tp
            n_labels += n
        
        precision = correct / n_labels

        wandb.log({f"{matrix_type}_precision": precision})

        # print(f"precision for {matrix_type}: {precision}")
        return precision

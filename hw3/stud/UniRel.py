import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from modify_bert import BertModel
from metrics import compute_metrics
from utils import reconstruct_relations_from_matrices, convert_to_string_format
import wandb
import torch.nn.functional as F
import config as cfg




class UniRE(BertPreTrainedModel, pl.LightningModule):
    """
    The UniRE model class
    
    """

    def __init__(self, config):
        super(UniRE, self).__init__(config=config)

        # bert model
        self.bert = BertModel.from_pretrained(cfg.PRETRAINED_MODEL, config=config)

        self.epoch_num = 0

        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()

        # Train Metrics
        self.train_h_preds = []
        self.train_t_preds = []
        self.train_span_preds = []
        self.train_labels = []
        self.train_losses = []
        self.train_h_CM = []
        self.train_t_CM = []
        self.train_span_CM = []

        # Validation Metrics
        self.val_h_preds = []
        self.val_t_preds = []
        self.val_span_preds = []
        self.val_labels = []
        self.val_losses = []
        self.val_h_CM = []
        self.val_t_CM = []
        self.val_span_CM = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        """
        Forward pass of the model, given the input sequence and the attention mask it computes the logits of the head, tail and span matrices
        
        Args:
            input_ids: The input ids of the input sequence              [B, L]
            attention_mask: The attention mask of the input sequence    [B, L]
            token_type_ids: The token type ids of the input sequence    [B, L]
            
        Returns:
            h_logits: The logits of the head matrix                     [B, L, L]
            t_logits: The logits of the tail matrix                     [B, L, L]
            span_logits: The logits of the span matrix                  [B, L, L]
        """

        # bert output
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=False, output_attentions_scores=True)

        # take the last layer attention scores
        attention_scores = out.attentions_scores[-1]

        # compute the logits for the head, tail and span matrices
        h_logits = self.sigmoid(attention_scores[:, :4, :, :].mean(1))
        t_logits = self.sigmoid(attention_scores[:, 4:8, :, :].mean(1))
        span_logits = self.sigmoid(attention_scores[:, 8:, :, :].mean(1))

        return h_logits, t_logits, span_logits

    def training_step(self, batch, batch_idx):
        """
        Training step of the model, given a batch of input sequences and labels it computes the loss and the metrics
        
        Args:
            batch: The batch of input sequences and labels
            
        Returns:
            loss: The total loss of the batch
        """

        indices, input_ids, attention_mask, token_type_ids, labels = batch

        # Forward pass
        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)
        
        # Compute the predictions for the head, tail and span matrices using the threshold
        h_pred = h_logits > cfg.THRESHOLD
        t_pred = t_logits > cfg.THRESHOLD
        span_pred = span_logits > cfg.THRESHOLD

        # Compute the loss for the head, tail and span matrices
        h_loss = self.loss(h_logits.float().reshape(-1), labels["head_matrices"].reshape(-1).float())
        t_loss = self.loss(t_logits.float().reshape(-1), labels["tail_matrices"].reshape(-1).float())
        span_loss = self.loss(span_logits.float().reshape(-1), labels["span_matrices"].reshape(-1).float())

        # Compute how many ones of the labels are predicted as ones
        h_TP, h_FP, h_FN = self.compute_confusion_matrix(h_pred, labels["head_matrices"])
        t_TP, t_FP, t_FN = self.compute_confusion_matrix(t_pred, labels["tail_matrices"])
        span_TP, span_FP, span_FN = self.compute_confusion_matrix(span_pred, labels["span_matrices"])

        # Compute the total loss
        loss = (h_loss + t_loss + span_loss)

        # Load the metrics into the lists
        self.train_losses.append(loss)
        self.train_h_preds.extend(h_pred)
        self.train_t_preds.extend(t_pred)
        self.train_span_preds.extend(span_pred)
        self.train_labels.extend(labels["spo"])
        self.train_h_CM.append((h_TP, h_FP, h_FN))
        self.train_t_CM.append((t_TP, t_FP, t_FN))
        self.train_span_CM.append((span_TP, span_FP, span_FN))

        # Log the metrics
        self.log("train_loss", loss, prog_bar = True, on_step=False, on_epoch=True)
        wandb.log({"h_TP_train": h_TP, "h_FP_train": h_FP, "h_FN_train": h_FN, "t_TP_train": t_TP, "t_FP_train": t_FP, "t_FN_train": t_FN, "span_TP_train": span_TP, "span_FP_train": span_FP, "span_FN_train": span_FN})

        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model, given a batch of input sequences and labels it computes the loss and the metrics
        
        Args:
            batch: The batch of input sequences and labels
        
        Returns:
            loss: The total loss of the batch
        """

        indices, input_ids, attention_mask, token_type_ids, labels = batch

        # Forward pass
        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)

        # Compute the predictions for the head, tail and span matrices using the threshold
        h_loss = self.loss(h_logits.float().view(-1), labels["head_matrices"].view(-1).float())
        t_loss = self.loss(t_logits.float().view(-1), labels["tail_matrices"].view(-1).float())
        span_loss = self.loss(span_logits.float().view(-1), labels["span_matrices"].view(-1).float())

        # Compute the predictions for the head, tail and span matrices using the threshold  
        h_pred = h_logits > cfg.THRESHOLD
        t_pred = t_logits > cfg.THRESHOLD
        span_pred = span_logits > cfg.THRESHOLD

        # Compute the loss for the head, tail and span matrices
        h_TP, h_FP, h_FN = self.compute_confusion_matrix(h_pred, labels["head_matrices"])
        t_TP, t_FP, t_FN = self.compute_confusion_matrix(t_pred, labels["tail_matrices"])
        span_TP, span_FP, span_FN = self.compute_confusion_matrix(span_pred, labels["span_matrices"])

        # Compute the total loss
        loss = (h_loss + t_loss + span_loss)

        # Load the metrics into the lists
        self.val_losses.append(loss)
        self.val_h_preds.extend(h_pred)
        self.val_t_preds.extend(t_pred)
        self.val_span_preds.extend(span_pred)
        self.val_labels.extend(labels["spo"])
        self.val_h_CM.append((h_TP*len(labels["head_matrices"].nonzero()), len(labels["head_matrices"].nonzero())))
        self.val_t_CM.append((t_TP*len(labels["tail_matrices"].nonzero()), len(labels["tail_matrices"].nonzero())))
        self.val_span_CM.append((span_TP*len(labels["span_matrices"].nonzero()), len(labels["span_matrices"].nonzero())))
       
        # Log the metrics
        self.log("val_loss", loss, prog_bar = True, on_step=False, on_epoch=True)
        wandb.log({"h_TP_val": h_TP, "h_FP_val": h_FP, "h_FN_val": h_FN, "t_TP_val": t_TP, "t_FP_val": t_FP, "t_FN_val": t_FN, "span_TP_val": span_TP, "span_FP_val": span_FP, "span_FN_val": span_FN})

        return {
            "loss": loss,
        }

    def on_train_epoch_end(self):
        """
        Method called at the end of the training epoch, it computes the metrics and logs them
        """

        self.epoch_num += 1

        # Compute the loss
        losses = self.train_losses
        loss = torch.stack(losses).mean()

        preds = []
        
        # The predictions are computed in batches to avoid memory issues
        for idx in range(0, len(self.train_h_preds), cfg.BATCH_SIZE):
            
            max_len = self.train_h_preds[idx].shape[0]
            selected_train_h_preds = self.train_h_preds[idx:min(len(self.train_h_preds), idx+cfg.BATCH_SIZE)]
            selected_train_t_preds = self.train_t_preds[idx:min(len(self.train_h_preds), idx+cfg.BATCH_SIZE)]
            selected_train_span_preds = self.train_span_preds[idx:min(len(self.train_h_preds), idx+cfg.BATCH_SIZE)]

            h_preds = torch.stack(selected_train_h_preds)
            t_preds = torch.stack(selected_train_t_preds)
            span_preds = torch.stack(selected_train_span_preds)

            # reconstruct the relations from the matrices into a list of triplets
            preds.extend(reconstruct_relations_from_matrices(h_preds, t_preds, span_preds, max_len))

            del h_preds, t_preds, span_preds

        labels = self.train_labels

        # Compute the metrics
        acc, prec, rec, f1 = compute_metrics(preds, labels)

        # Log the metrics
        self.log("train_accuracy", acc)
        self.log("train_precision", prec)
        self.log("train_recall", rec)
        self.log("train_f1_score", f1)
        self.log("epoch_train_loss", loss)
        
        # Reset the lists
        self.train_h_preds = []
        self.train_t_preds = []
        self.train_span_preds = []
        self.train_labels = []
        self.train_losses = []
        self.train_h_CM = []
        self.train_t_CM = []
        self.train_span_CM = []

    def on_validation_epoch_end(self):
        """
        Method called at the end of the validation epoch, it computes the metrics and logs them
        """

        # Pad the predictions to the maximum length
        max_len = torch.max(torch.tensor([el.shape[0] for el in self.val_h_preds])).item()
        h_preds = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0], 0, max_len - tensor.shape[0])) for tensor in self.val_h_preds])
        t_preds = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0], 0, max_len - tensor.shape[0])) for tensor in self.val_t_preds])
        span_preds = torch.stack([F.pad(tensor, (0, max_len - tensor.shape[0], 0, max_len - tensor.shape[0])) for tensor in self.val_span_preds])
        labels = self.val_labels
        losses = self.val_losses

        # Compute the confusion matrices precision for statistics
        _, _, _ = self.matrix_precision(self.val_h_CM, "h"), self.matrix_precision(self.val_t_CM, "t"), self.matrix_precision(self.val_span_CM, "span")

        # Reconstruct the relations from the matrices into a list of triplets
        preds = reconstruct_relations_from_matrices(h_preds, t_preds, span_preds, max_len)

        # Compute the metrics
        acc, prec, rec, f1 = compute_metrics(preds, labels)

        # Compute the loss
        loss = torch.stack(losses).mean()

        # Log the metrics
        self.log("val_accuracy", acc)
        self.log("val_precision", prec)
        self.log("val_recall", rec)
        self.log("val_f1_score", f1)
        self.log("epoch_val_loss", loss)

        # Reset the lists
        self.val_h_preds = []
        self.val_t_preds = []
        self.val_span_preds = []
        self.val_labels = []
        self.val_losses = []
        self.val_h_CM = []
        self.val_t_CM = []
        self.val_span_CM = []

    def predict(self, data, max_len):
        """
        Predicts the triplets given the input sequences
        
        Args:
            data: The input sequences, as:
                indices: The indices of the input sequences
                input_ids: The input ids of the input sequences
                attention_mask: The attention mask of the input sequences
                token_type_ids: The token type ids of the input sequences
                
            max_len: The maximum length of the input sequences
            
            Returns:
                preds: The list of predicted triplets
        """

        indices, input_ids, attention_mask, token_type_ids = data

        # Forward pass in evaluation mode
        h_logits, t_logits, span_logits = self(input_ids, attention_mask, token_type_ids)

        # Compute the predictions for the head, tail and span matrices using the threshold
        h_pred = h_logits > cfg.THRESHOLD
        t_pred = t_logits > cfg.THRESHOLD
        span_pred = span_logits > cfg.THRESHOLD

        # Reconstruct the relations from the matrices into a list of triplets
        triplets_preds = reconstruct_relations_from_matrices(h_pred, t_pred, span_pred, max_len)

        # convert the predictions to the string format
        preds = []
        for idx, pred in zip(indices, triplets_preds):
            preds.append(convert_to_string_format(idx, pred, max_len))

        return preds
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    
    def compute_confusion_matrix(self, preds: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Compute the True Positive Rate, False Positive Rate and False Negative Rate of the prediction matrices
        
        Args:
            preds: The predictions
            labels: The labels
            
        Returns:
            TPR: The True Positive Rate
            FPR: The False Positive Rate
            FNR: The False Negative Rate
        """

        # Take the non zero elements of the labels and the predictions
        labels_one = set(tuple(el) for el in labels.nonzero().tolist())
        preds_one = set(tuple(el) for el in preds.nonzero().tolist())
        intersection = set(labels_one).intersection(set(preds_one))

        # Compute TPR
        TPR = len(intersection) / len(labels_one) if len(labels_one) > 0 else 0

        # Compute FPR
        FPR = len(preds_one - intersection) / len(preds_one) if len(preds_one) > 0 else 0 # Da CORREGGERE

        # Compute FNR
        FNR = len(labels_one - intersection) / len(labels_one)

        return TPR, FPR, FNR
        
    def matrix_precision(self, matrix: list, matrix_type: str) -> float:
        """
        Compute the precision of the prediction matrices

        Args:
            matrix: The confusion matrix
            matrix_type: The type of the matrix
        
        Returns:
            precision: The precision of the matrix
        """

        n_labels = 0
        correct = 0

        # count the true positives and the total number of labels
        for tp, n in matrix:
            correct += tp
            n_labels += n
        
        # Compute the precision
        precision = correct / n_labels

        # Log the precision
        wandb.log({f"{matrix_type}_precision": precision})

        return precision

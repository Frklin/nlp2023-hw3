import os
import random
import numpy as np
import torch
import config
from dataclasses import dataclass



@dataclass
class Entity:
    start_idx: int
    end_idx: int
    text: str = ""
    entity_type: str = ""

    def __str__(self):
        return f"[{self.entity_type}]{self.text if self.text else 'No Text'}: ({self.start_idx}, {self.end_idx})"
    
    def __repr__(self):
        return f"[{self.entity_type}]{self.text if self.text else 'No Text'}: ({self.start_idx}, {self.end_idx})"
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Entity):
            return False
        return self.start_idx == __value.start_idx and self.end_idx == __value.end_idx and self.entity_type == __value.entity_type
    
    def __hash__(self) -> int:
        return hash((self.start_idx, self.end_idx, self.entity_type))

@dataclass
class LabelFormat:
    subject: Entity
    relation: int
    object: Entity #add id for the line?


    def __str__(self):
        return f"<{self.subject} | {self.relation} | {self.object}>"
    
    def __repr__(self):
        return f"<{self.subject} | {self.relation} | {self.object}>"
    
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, LabelFormat):
            return False
        return self.subject == __value.subject and self.relation == __value.relation and self.object == __value.object
    
    def __hash__(self) -> int:
        return hash((self.subject, self.relation, self.object))


def seed_everything(seed=42):
    """
    Seeds basic parameters for reproductibility of results

    Args:
        seed (int, optional): Number of the seed. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def collate_fn(batch):
    indices, input_ids, attention_masks, token_type_ids, position_ids, relations = zip(*batch)

    labels = {"head_matrices": [], "tail_matrices": [], "span_matrices": [], "spo": []}

    for i, relation in enumerate(relations):
        get_label_matrices(labels, relation, position_ids[i])

    indices = torch.tensor(indices)
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    token_type_ids = torch.stack(token_type_ids)
    labels["head_matrices"] = torch.stack(labels["head_matrices"])
    labels["tail_matrices"] = torch.stack(labels["tail_matrices"])
    labels["span_matrices"] = torch.stack(labels["span_matrices"])

    return indices, input_ids, attention_masks, token_type_ids, labels





def get_label_matrices(labels, relation, position_ids):
    e2e = set()
    h2r = set()
    t2r = set()
    spo_span = set()
    spo_text = set()

    head_matrix = torch.zeros([config.MAX_LEN + config.REL_NUM, config.MAX_LEN + config.REL_NUM])
    tail_matrix = torch.zeros([config.MAX_LEN + config.REL_NUM, config.MAX_LEN + config.REL_NUM])
    span_matrix = torch.zeros([config.MAX_LEN + config.REL_NUM, config.MAX_LEN + config.REL_NUM])

    for spo in relation:
        subject = spo['subject']
        s_text = subject['text']
        s_start = position_ids.index(subject['start_idx'])
        s_end = position_ids.index(subject['end_idx'])  

        predicate = spo['relation']
        pred_idx = config.relation2Id[predicate]
        pred_shifted_idx = pred_idx + config.MAX_LEN # pred_idx is wrong?

        object = spo['object']
        o_start = position_ids.index(object['start_idx'])
        o_end = position_ids.index(object['end_idx']) 
        o_text = object['text']


        spo_text.add((s_text, predicate, o_text))

        del subject, object

        # Entity-Entity
        head_matrix[s_start, o_start] = 1
        head_matrix[o_start, s_start] = 1
        tail_matrix[s_start, o_start] = 1
        tail_matrix[o_start, s_start] = 1
        span_matrix[s_start, s_end] = 1
        span_matrix[s_end, s_start] = 1
        span_matrix[o_start, o_end] = 1
        span_matrix[o_end, o_start] = 1

        # Entity-Relation (Subject)
        head_matrix[s_start, pred_shifted_idx] = 1
        tail_matrix[s_end, pred_shifted_idx] = 1
        span_matrix[s_start, pred_shifted_idx] = 1
        span_matrix[s_end, pred_shifted_idx] = 1
        span_matrix[o_start, pred_shifted_idx] = 1
        span_matrix[o_end, pred_shifted_idx] = 1

        # Relation-Entity (Object)
        head_matrix[pred_shifted_idx, o_start] = 1
        tail_matrix[pred_shifted_idx, o_end] = 1
        span_matrix[pred_shifted_idx, o_start] = 1
        span_matrix[pred_shifted_idx, o_end] = 1
        span_matrix[pred_shifted_idx, s_start] = 1
        span_matrix[pred_shifted_idx, s_end] = 1

        e2e.add((s_start, o_start))
        e2e.add((o_start, s_start))
        h2r.add((s_start, pred_shifted_idx))
        t2r.add((o_start, pred_shifted_idx))

        # spo_span.add(LabelFormat(Entity(s_start, s_end, s_text, "subject"), pred_idx, Entity(o_start, o_end, o_text, "object")))
        spo_span.add(((s_start, s_end), pred_idx, (o_start, o_end)))
    
    labels["head_matrices"].append(head_matrix)
    labels["tail_matrices"].append(tail_matrix)
    labels["span_matrices"].append(span_matrix)
    # add label as the output
    labels["spo"].append(spo_span)
    # labels["spo_span"].append(spo_span)
    # labels["spo_text"].append(spo_text)




def reconstruct_relations_from_matrices(head_matrices, tail_matrices, span_matrices, labels=None):
    """
    head_matrices: torch.Tensor, shape (batch_size, max_len + rel_num, max_len + rel_num)
    """
    relations = []

    for i in range(head_matrices.shape[0]):
        # with open("h_pred.txt", "a") as f:
        #     f.write(str(head_matrices[i]))
        # with open("t_pred.txt", "a") as f:
        #     f.write(str(tail_matrices[i]))
        # with open("span_pred.txt", "a") as f:
        #     f.write(str(span_matrices[i]))
        if head_matrices[i].sum() == 0 or tail_matrices[i].sum() == 0 or span_matrices[i].sum() == 0:
            relations.append([])
            continue
        rel = matrices2relations(head_matrices[i], tail_matrices[i], span_matrices[i])
        if rel:
            relations.append(rel)
        else:
            relations.append([])

    return relations



def matrices2relations(head_matrix, tail_matrix, span_matrix):
    relations = set()
    # Assuming head_matrix, tail_matrix, and span_matrix are numpy arrays or similar
    max_len = config.MAX_LEN
    rel_num = config.REL_NUM

    for i in range(max_len):
        for j in range(max_len):  # Ensure i < j to avoid duplicate relations
            if span_matrix[i, j]:
                # Entity-Relation mapping found, now find the corresponding relation type and entities
                for rel_idx in range(max_len, max_len + rel_num):
                    if head_matrix[i, rel_idx] and tail_matrix[j, rel_idx]:
                        # relation = relation2Id_inv[rel_idx - max_len]
                        # subject = {"start_idx": i, "end_idx": j}
                        # Find object for the relation
                        for k in range(max_len):
                            if head_matrix[rel_idx, k]:  # Subject-Object mapping
                                # object_ = {"start_idx": k, "end_idx": k}  # Simplified; in practice, you'd identify the span of the object
                                for l in range(max_len):
                                    if span_matrix[k, l]:
                                        relations.add(((i, j), rel_idx-max_len, (k, l)))
    return list(relations)
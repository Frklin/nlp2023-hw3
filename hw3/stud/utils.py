import os
import random
import numpy as np
import torch
import config
import matplotlib.pyplot as plt
from transformers import  BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence



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
    indices, input_ids, position_ids, relations = zip(*batch)

    labels = {"head_matrices": [], "tail_matrices": [], "span_matrices": [], "spo": []}

    max_len = max([len(input_id) for input_id in input_ids])

    # Pad input_ids
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    preds_type_ids = torch.ones(len(indices), config.REL_NUM, dtype=torch.int32)


    # concatenate input_ids with encoded_preds
    input_ids = torch.cat([input_ids, config.encoded_preds.unsqueeze(0).expand(input_ids.shape[0], -1)], dim=1)
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0).tolist()
    attention_masks = input_ids != 0
    attention_masks = attention_masks.int()
    token_type_ids = torch.cat([token_type_ids, preds_type_ids], dim=1)

    for i, relation in enumerate(relations):
        get_label_matrices(labels, relation, position_ids[i], max_len)

    indices = torch.tensor(indices)
    labels["head_matrices"] = torch.stack(labels["head_matrices"])
    labels["tail_matrices"] = torch.stack(labels["tail_matrices"])
    labels["span_matrices"] = torch.stack(labels["span_matrices"])

    return indices, input_ids, attention_masks, token_type_ids, labels
    



def get_label_matrices(labels, relation, position_ids, max_len):
    spo_span = set()

    head_matrix = torch.zeros([max_len + config.REL_NUM, max_len + config.REL_NUM])
    tail_matrix = torch.zeros([max_len + config.REL_NUM, max_len + config.REL_NUM])
    span_matrix = torch.zeros([max_len + config.REL_NUM, max_len + config.REL_NUM])

    for spo in relation:
        subject = spo['subject']
        s_start = position_ids.index(subject['start_idx'])
        s_end = position_ids.index(subject['end_idx']+1)-1  

        predicate = spo['relation']
        pred_idx = config.relation2Id[predicate]
        pred_shifted_idx = pred_idx + max_len

        object = spo['object']
        o_start = position_ids.index(object['start_idx'])
        o_end = position_ids.index(object['end_idx']+1)-1 

        del subject, object

        # Entity-Entity
        head_matrix[s_start][o_start] = 1
        head_matrix[o_start][s_start] = 1
        tail_matrix[s_end][o_end] = 1
        tail_matrix[o_end][s_end] = 1
        span_matrix[s_start][s_end] = 1
        span_matrix[s_end][s_start] = 1
        span_matrix[o_start][o_end] = 1
        span_matrix[o_end][o_start] = 1
        # Subject-Relation Interaction
        head_matrix[s_start][pred_shifted_idx] = 1
        tail_matrix[s_end][pred_shifted_idx] = 1
        span_matrix[s_start][pred_shifted_idx] = 1
        span_matrix[s_end][pred_shifted_idx] = 1
        span_matrix[o_start][pred_shifted_idx] = 1
        span_matrix[o_end][pred_shifted_idx] = 1
        # Relation-Object Interaction
        head_matrix[pred_shifted_idx][o_start] = 1
        tail_matrix[pred_shifted_idx][o_end] = 1
        span_matrix[pred_shifted_idx][o_start] = 1
        span_matrix[pred_shifted_idx][o_end] = 1
        span_matrix[pred_shifted_idx][s_start] = 1
        span_matrix[pred_shifted_idx][s_end] = 1

        spo_span.add(((s_start, s_end), pred_shifted_idx, (o_start, o_end)))
    
    head_ones_coordinates = [(i.item(), j.item()) for i, j in head_matrix.nonzero()]
    tail_ones_coordinates = [(i.item(), j.item()) for i, j in tail_matrix.nonzero()]
    span_ones_coordinates = [(i.item(), j.item()) for i, j in span_matrix.nonzero()]

    if config.DEBUG:
        rel = matrices2relations(head_ones_coordinates, tail_ones_coordinates, span_ones_coordinates)
        assert len(set(rel).intersection(spo_span)) == len(spo_span), f"Matrices2Relations failed to reconstruct the original spo_span: {rel} != {spo_span}"

    labels["head_matrices"].append(head_matrix)
    labels["tail_matrices"].append(tail_matrix)
    labels["span_matrices"].append(span_matrix)
    labels["spo"].append(spo_span)




def reconstruct_relations_from_matrices(head_matrices, tail_matrices, span_matrices, max_len):
    """
    head_matrices: torch.Tensor, shape (batch_size, max_len + rel_num, max_len + rel_num)
    """
    relations = []

    for k in range(head_matrices.shape[0]):
        if head_matrices[k].sum() == 0 or tail_matrices[k].sum() == 0 or span_matrices[k].sum() == 0:
            relations.append([])
            continue
        tail_ones_coordinates = [(i.item(), j.item()) for i, j in tail_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item() and not(i.item() > max_len and j.item() > max_len)]
        head_ones_coordinates = [(i.item(), j.item()) for i, j in head_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item() and not(i.item() > max_len and j.item() > max_len)]
        span_ones_coordinates = [(i.item(), j.item()) for i, j in span_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item() and not(i.item() > max_len and j.item() > max_len)]

        rel = matrices2relations(head_ones_coordinates, tail_ones_coordinates, span_ones_coordinates, max_len)

        relations.append(rel)

    return relations


def matrices2relations(head, tail, span, max_len):
    # Convert to set for faster operations
    head_set = set(head)
    span_set = set(span)
    tail_set = set(tail)
    
    # Step 1: Extract initial triples where relation > max_len
    initial_triples = find_head_spo_triples(head, max_len)
    
    # Step 2 & 3: Find s_end and o_end for each triple
    final_relations = []
    for s_start, r, o_start in initial_triples:
        s_end_candidates = {s_end for s_start_i, s_end in span_set if s_start_i == s_start and s_start < s_end < max_len}
        o_end_candidates = {o_end for o_start_i, o_end in span_set if o_start_i == o_start and o_start < o_end < max_len}
        
        # Optional Step 4: Verify existence of (s_end, r, o_end)
        for s_end in s_end_candidates:
            for o_end in o_end_candidates:
                if ((s_end, r) in tail_set and (r, o_end) in tail_set):
                    final_relations.append(((s_start, s_end), r, (o_start, o_end)))
                
                if config.BIDIRECTIONAL:
                    if ((o_end, r) in head_set and (r, s_end) in head_set):
                        final_relations.append(((s_start, s_end), r, (o_start, o_end)))

    return final_relations
        
def find_head_spo_triples(head, max_len):
    s_o_candidates = {(x, y) for x, y in head if x < max_len or y < max_len}
    r_candidates = {x for x, y in head if x > max_len or y > max_len}
    
    # Find triples (s, r, o)
    triples = []
    for r in r_candidates:
        # Find all s starting with r
        s_candidates = {s for s, o in head if o == r}
        # Find all o ending with r
        o_candidates = {o for s, o in head if s == r}
        
        # Combine s and o candidates to form triples
        for s in s_candidates:
            for o in o_candidates:
                if (s, o) in s_o_candidates:  # Ensure there's a direct connection between s and o
                    triples.append((s, r, o))
                
                if config.BIDIRECTIONAL:
                    if (o, s) in s_o_candidates:
                        triples.append((s, r, o))
    
    return triples




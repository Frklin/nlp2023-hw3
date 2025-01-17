import os
import random
import numpy as np
import torch
import config
from torch.nn.utils.rnn import pad_sequence




def seed_everything(seed: int=42):
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
    """
    Collate function to be used in the DataLoader. It pads the input_ids and position_ids to the maximum length in the batch and returns the indices, input_ids, attention_masks, token_type_ids and labels.

    Args:
        batch (list): List of tuples of the form (indices, input_ids, position_ids, relations)

    Returns:
        indices, input_ids, attention_masks, token_type_ids, labels
    """

    indices, input_ids, position_ids, relations = zip(*batch)

    # Initialize labels
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

    # Construct the label matrices from the relations tuples
    for i, relation in enumerate(relations):
        get_label_matrices(labels, relation, position_ids[i], max_len)

    # Convert labels to tensors
    indices = torch.tensor(indices)
    labels["head_matrices"] = torch.stack(labels["head_matrices"])
    labels["tail_matrices"] = torch.stack(labels["tail_matrices"])
    labels["span_matrices"] = torch.stack(labels["span_matrices"])

    return indices, input_ids, attention_masks, token_type_ids, labels
    

def get_label_matrices(labels: dict, relation: list, position_ids: list, max_len: int):
    """
    The function constructs the label matrices from the relations tuples

    Args:
        relation: list of dicts of the form {"subject": 
                                                {
                                                    "start_idx": int, 
                                                    "end_idx": int,
                                                    "text": str
                                                }, 
                                            "relation": str,
                                            "object": 
                                                {
                                                    "start_idx": int, 
                                                    "end_idx": int, 
                                                    "text": str
                                                }
                                            }
        position_ids: list of int
        max_len: int
    """

    spo_span = set()

    # Initialize matrices
    head_matrix = torch.zeros([max_len + config.REL_NUM, max_len + config.REL_NUM])
    tail_matrix = torch.zeros([max_len + config.REL_NUM, max_len + config.REL_NUM])
    span_matrix = torch.zeros([max_len + config.REL_NUM, max_len + config.REL_NUM])

    for spo in relation:
        subject = spo['subject']
        s_start = position_ids.index(subject['start_idx'])
        s_end = position_ids.index(subject['end_idx']) 

        predicate = spo['relation']
        pred_idx = config.relation2Id[predicate]
        pred_shifted_idx = pred_idx + max_len

        object = spo['object']
        o_start = position_ids.index(object['start_idx'])
        o_end = position_ids.index(object['end_idx'])

        del subject, object

        # Entity-Entity Interaction
        head_matrix[s_start][o_start] = 1
        head_matrix[o_start][s_start] = 1   if config.BIDIRECTIONAL else 0
        tail_matrix[s_end][o_end] = 1
        tail_matrix[o_end][s_end] = 1       if config.BIDIRECTIONAL else 0
        span_matrix[s_start][s_end] = 1
        span_matrix[s_end][s_start] = 1     if config.BIDIRECTIONAL else 0
        span_matrix[o_start][o_end] = 1
        span_matrix[o_end][o_start] = 1     if config.BIDIRECTIONAL else 0
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
    


    if config.DEBUG:
        # Check if the built matrices are consistent with the original spo_span
        head_ones_coordinates = [(i.item(), j.item()) for i, j in head_matrix.nonzero()]
        tail_ones_coordinates = [(i.item(), j.item()) for i, j in tail_matrix.nonzero()]
        span_ones_coordinates = [(i.item(), j.item()) for i, j in span_matrix.nonzero()]

        rel = matrices2relations(head_ones_coordinates, tail_ones_coordinates, span_ones_coordinates)

        assert len(set(rel).intersection(spo_span)) == len(spo_span), f"Matrices2Relations failed to reconstruct the original spo_span: {rel} != {spo_span}"

    labels["head_matrices"].append(head_matrix)
    labels["tail_matrices"].append(tail_matrix)
    labels["span_matrices"].append(span_matrix)
    labels["spo"].append(spo_span)

def reconstruct_relations_from_matrices(head_matrices: torch.Tensor, tail_matrices: torch.Tensor, span_matrices: torch.Tensor, max_len: int) -> list:
    """
    Reconstructs the relations from the head, tail and span matrices

    Args:
        head_matrices: torch.Tensor [B, N, N]
        tail_matrices: torch.Tensor [B, N, N]
        span_matrices: torch.Tensor [B, N, N]
        max_len: int

    Returns:
        list of list of tuples of the form ((s_start, s_end), r, (o_start, o_end))
    """

    relations = []

    for k in range(head_matrices.shape[0]):

        # if no interaction is detected, append an empty list
        if head_matrices[k].sum() == 0 or tail_matrices[k].sum() == 0 or span_matrices[k].sum() == 0:
            relations.append([])
            continue

        # Extract coordinates of ones
        tail_ones_coordinates = [(i.item(), j.item()) for i, j in tail_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item()]# and not(i.item() > max_len and j.item() > max_len)]
        head_ones_coordinates = [(i.item(), j.item()) for i, j in head_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item()]# and not(i.item() > max_len and j.item() > max_len)]
        span_ones_coordinates = [(i.item(), j.item()) for i, j in span_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item()]# and not(i.item() > max_len and j.item() > max_len)]

        # Reconstruct relations
        rel = matrices2relations(head_ones_coordinates, tail_ones_coordinates, span_ones_coordinates, max_len)

        relations.append(rel)

    return relations

def matrices2relations(head: list, tail: list, span: list, max_len: int) -> list:
    """
    Reconstructs the relations from the head, tail and span matrices

    Args:
        head: list of tuples of the form (int, int)
        tail: list of tuples of the form (int, int)
        span: list of tuples of the form (int, int)
        max_len: int

    Returns:
        list of tuples of the form ((s_start, s_end), r, (o_start, o_end))
    """

    span_set = set(span)
    tail_set = set(tail)
    
    # Extract initial triples where relation > config.MAX_LEN
    initial_triples = find_head_spo_triples(head, max_len)    

    # Find s_end and o_end for each triple
    final_relations = []
    for s_start, r, o_start in initial_triples:
        
        # Find all s_end and o_end candidates in the span_matrix
        s_end_candidates = {s_end for s_start_i, s_end in span_set if s_start_i == s_start}
        o_end_candidates = {o_end for o_start_i, o_end in span_set if o_start_i == o_start}
         
        # Verify existence of (s_end, r, o_end)
        for s_end in s_end_candidates:
            for o_end in o_end_candidates:
                if ((s_end, r) in tail_set and (r, o_end) in tail_set):
                    final_relations.append(((s_start, s_end), r, (o_start, o_end)))

    return final_relations
        
def find_head_spo_triples(head: list, max_len: int) -> list:
    """
    Finds the initial triples (s, r, o) where r > max_len

    Args:
        head: list of tuples of the form (int, int)
        max_len: int

    Returns:
        list of tuples of the form (int, int, int)
    """

    s_o_candidates = {(x, y) for x, y in head}
    r_candidates = {x for x, y in head}
    
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
                # Ensure there's a direct connection between s and o
                if (s, o) in s_o_candidates:  
                    triples.append((s, r, o))
    return triples

def convert_to_string_format(idx: int, relations: list, max_len: int) -> list:
    '''
    Converts the relations to the string format

    Args:
        idx: int
        relations: list of tuples of the form ((s_start, s_end), r, (o_start, o_end))
        max_len: int

    Returns:
        list of dicts of the form right format for the output
    '''
    
    formatted_relations = []
    for rel in relations:
        formatted_relations.append({"subject": 
                                    {
                                        "start_idx": config.index_shift[idx][rel[0][0]],
                                        "end_idx": config.index_shift[idx][rel[0][1]]
                                    },
                                    "relation": config.id2relation[rel[1]-max_len],
                                    "object": 
                                    {
                                        "start_idx": config.index_shift[idx][rel[2][0]],
                                        "end_idx": config.index_shift[idx][rel[2][1]]
                                    }
                                    })
    return formatted_relations

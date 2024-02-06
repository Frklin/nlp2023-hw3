import os
import random
import numpy as np
import torch
import config
from dataclasses import dataclass
import matplotlib.pyplot as plt



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
        s_end = position_ids.index(subject['end_idx']+1)-1  

        predicate = spo['relation']
        pred_idx = config.relation2Id[predicate]
        pred_shifted_idx = pred_idx + config.MAX_LEN# pred_idx is wrong?

        object = spo['object']
        o_start = position_ids.index(object['start_idx'])
        o_end = position_ids.index(object['end_idx']+1)-1 
        o_text = object['text']


        spo_text.add((s_text, predicate, o_text))

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

        e2e.add((s_start, o_start))
        e2e.add((o_start, s_start))
        h2r.add((s_start, pred_shifted_idx))
        t2r.add((o_start, pred_shifted_idx))

        # spo_span.add(LabelFormat(Entity(s_start, s_end, s_text, "subject"), pred_idx, Entity(o_start, o_end, o_text, "object")))
        spo_span.add(((s_start, s_end), pred_shifted_idx, (o_start, o_end)))
    
    head_ones_coordinates = [(i.item(), j.item()) for i, j in head_matrix.nonzero()]
    tail_ones_coordinates = [(i.item(), j.item()) for i, j in tail_matrix.nonzero()]
    span_ones_coordinates = [(i.item(), j.item()) for i, j in span_matrix.nonzero()]

    rel = matrices2relations(head_ones_coordinates, tail_ones_coordinates, span_ones_coordinates)
    # assert len(set(rel).intersection(spo_span)) == len(spo_span), f"Matrices2Relations failed to reconstruct the original spo_span: {rel} != {spo_span}"

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

    for k in range(head_matrices.shape[0]):
        if head_matrices[k].sum() == 0 or tail_matrices[k].sum() == 0 or span_matrices[k].sum() == 0:
            relations.append([])
            continue
        head_ones_coordinates = [(i.item(), j.item()) for i, j in head_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item() and not(i.item() > config.MAX_LEN and j.item() > config.MAX_LEN)]
        tail_ones_coordinates = [(i.item(), j.item()) for i, j in tail_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item() and not(i.item() > config.MAX_LEN and j.item() > config.MAX_LEN)]
        span_ones_coordinates = [(i.item(), j.item()) for i, j in span_matrices[k].nonzero() if i.item() != 0 and j.item() != 0 and i.item() != j.item() and not(i.item() > config.MAX_LEN and j.item() > config.MAX_LEN)]

        rel = matrices2relations(head_ones_coordinates, tail_ones_coordinates, span_ones_coordinates)

        # if k == 0:
        #     plot_images(head_ones_coordinates, tail_ones_coordinates, span_ones_coordinates)
        relations.append(rel)

    return relations


def matrices2relations(head, tail, span):
    # Convert to set for faster operations
    head_set = set(head)
    span_set = set(span)
    tail_set = set(tail)
    
    # Step 1: Extract initial triples where relation > config.MAX_LEN
    initial_triples = find_head_spo_triples(head)
    
    # Step 2 & 3: Find s_end and o_end for each triple
    final_relations = []
    for s_start, r, o_start in initial_triples:
        s_end_candidates = {s_end for s_start_i, s_end in span_set if s_start_i == s_start and s_end < config.MAX_LEN and s_end > s_start}
        o_end_candidates = {o_end for o_start_i, o_end in span_set if o_start_i == o_start and o_end < config.MAX_LEN and o_end > o_start}
        
        # Optional Step 4: Verify existence of (s_end, r, o_end)
        for s_end in s_end_candidates:
            for o_end in o_end_candidates:
                if ((s_end, r) in tail_set and (r, o_end) in tail_set):
                    final_relations.append(((s_start, s_end), r, (o_start, o_end)))



    return final_relations
        
def find_head_spo_triples(head):
    s_o_candidates = {(x, y) for x, y in head if x < config.MAX_LEN and y < config.MAX_LEN}
    r_candidates = {x for x, y in head if x > config.MAX_LEN or y > config.MAX_LEN}
    
    # Find triples (s, r, o)
    triples = []
    for r in r_candidates:
        # Find all s starting with r
        s_candidates = {s for s, o in head if o == r and s < config.MAX_LEN}
        # Find all o ending with r
        o_candidates = {o for s, o in head if s == r and o < config.MAX_LEN}
        
        # Combine s and o candidates to form triples
        for s in s_candidates:
            for o in o_candidates:
                if (s, o) in s_o_candidates:  # Ensure there's a direct connection between s and o
                    triples.append((s, r, o))
    
    return triples




def plot_images(head, tail, span):
    max_value = config.MAX_LEN+ config.REL_NUM+2# max(max(max(head), max(tail), key=lambda x: x[1]), max(max(span), key=lambda x: x[1]))
    matrix_size = max_value + 1  # Adding 1 to include the max value in the index

    # Initialize matrices for head, tail, span, and combined
    head_matrix = np.zeros((matrix_size, matrix_size))
    tail_matrix = np.zeros((matrix_size, matrix_size))
    span_matrix = np.zeros((matrix_size, matrix_size))

    # Populate the matrices
    for (i, j) in head:
        head_matrix[i-1, j-1] = 1
    for (i, j) in tail:
        tail_matrix[i, j] = 1
    for (i, j) in span:
        span_matrix[i, j] = 1

    # Combine all the matrices
    combined_matrix = head_matrix + tail_matrix + span_matrix

    plot_and_save_matrix(head_matrix, 'Head Relations', 'data/images/head.png', colorscale='Greens')
    plot_and_save_matrix(tail_matrix, 'Tail Relations', 'data/images/tail_relations.png', colorscale='Blues')
    plot_and_save_matrix(span_matrix, 'Span Relations', 'data/images/span_relations.png', colorscale='Purples')
    plot_and_save_matrix(combined_matrix, 'Combined Relations', 'data/images/combined_relations.png', colorscale='Reds')




def plot_and_save_matrix(matrix, title, filename, colorscale='Blues'):
   

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap=colorscale, interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.close()


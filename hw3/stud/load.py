from torch.utils.data import Dataset
from dataclasses import dataclass
import torch
import config
import json


@dataclass
class Entity:
    start: int
    end: int
    text: str = ""

    def __repr__(self):
        return f"({self.text}, [{self.start}, {self.end}])"

    def get_text(self, tokens):
        return " ".join(tokens[self.start:self.end+1])

@dataclass
class Relation:
    subject: Entity
    relation: str
    object: Entity

    def __repr__(self):
        return f"<{self.subject} - {self.relation} - {self.object}>"

    def to_dict(self):
        return {
            "subject": {
                "start_idx": self.subject.start,
                "end_idx": self.subject.end,
            },
            "relation": self.relation,
            "object": {
                "start_idx": self.object.start,
                "end_idx": self.object.end,
            }
        }
    
@dataclass
class Label:
    head_matrix: torch.Tensor
    tail_matrix: torch.Tensor
    span_matrix: torch.Tensor
    spo_span: set
    spo_text: set

    def __repr__(self):
        return f"head_matrix: {self.head_matrix}\ntail_matrix: {self.tail_matrix}\nspan_matrix: {self.span_matrix}\nspo_span: {self.spo_span}\nspo_text: {self.spo_text}"

class RelationDataset(Dataset):

    def __init__(self, data_path, tokenizer='bert-base-uncased'): 
        self.data_path = data_path
        self.tokens = []
        self.relations = []
        self.tokenizer = tokenizer
        self.rel2pred_str = " ".join([rel for rel in config.rel2pred.values()])
        self.encoded_preds = tokenizer.encode_plus(self.rel2pred_str, add_special_tokens=False)
        self.labels = {
            "head_matrices": [],
            "tail_matrices": [],
            "span_matrices": [],
            "spo_span": [],
            "spo_text": []
        }
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_path, 'r') as f:
            l = 0
            for line in f:
                # if l == 100:
                #     break
                # l+=1
                data = json.loads(line)
                self.tokens.append(data['tokens'])
                # for relation in data["relations"]:
                self.relations.append(data["relations"])
                # get_label_matrices(self.labels, data["relations"])


    def __len__(self):
        return len(self.tokens)


    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        inputs = self.tokenizer.encode_plus(tokens, max_length=config.MAX_LEN, truncation=True, padding='max_length')
        sep_idx = inputs["input_ids"].index(self.tokenizer.sep_token_id)
        
        input_ids = inputs["input_ids"] + self.encoded_preds['input_ids'] 
        input_ids[sep_idx] = self.tokenizer.sep_token_id

        attention_mask = inputs["attention_mask"] + [1] * config.REL_NUM#len(self.encoded_preds["input_ids"])
        attention_mask[sep_idx] = 0

        token_type_ids = inputs["token_type_ids"] + [1] * config.REL_NUM#len(self.encoded_preds["input_ids"])

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        # label = {}
        # label["head_matrices"] = self.labels["head_matrices"][idx]
        # label["tail_matrices"] = self.labels["tail_matrices"][idx]
        # label["span_matrices"] = self.labels["span_matrices"][idx]
        # label["spo_span"] = self.labels["spo_span"][idx]
        # label["spo_text"] = self.labels["spo_text"][idx]

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "token_type_ids": token_type_ids,
        #     # "labels": label
        #     "relations": self.relations[idx],
        # }
        return input_ids, attention_mask, token_type_ids, self.relations[idx]


def parseRelation2Id(data_path):
    with open(data_path, 'r') as f:
        relation2Id = json.load(f)
        relation2Id = {k: v for k, v in sorted(relation2Id.items(), key=lambda item: item[1])}
    return relation2Id

def parseRel2Pred(data_path):
    with open(data_path, 'r') as f:
        rel2pred = json.load(f)
    return rel2pred

def get_label_matrices(labels, relation):
    e2e = set()
    h2r = set()
    t2r = set()
    spo_span = set()
    spo_text = set()

    head_matrix = torch.zeros((config.MAX_LEN + 2 + config.REL_NUM, config.MAX_LEN + 2 + config.REL_NUM))
    tail_matrix = torch.zeros((config.MAX_LEN + 2 + config.REL_NUM, config.MAX_LEN + 2 + config.REL_NUM))
    span_matrix = torch.zeros((config.MAX_LEN + 2 + config.REL_NUM, config.MAX_LEN + 2 + config.REL_NUM))

    for spo in relation:
        subject = spo['subject']
        s_text = subject['text']
        s_start = subject['start_idx']
        s_end = subject['end_idx']

        predicate = spo['relation']
        pred_idx = config.relation2Id[predicate]
        pred_shifted_idx = pred_idx + config.MAX_LEN + 2

        object = spo['object']
        o_start = object['start_idx']
        o_end = object['end_idx']
        o_text = object['text']

        spo_span.add(((s_start, s_end), pred_idx, (o_start, o_end)))
        spo_text.add((s_text, predicate, o_text))

        del subject, object

        # Entity-Entity
        head_matrix[s_start+1, o_start+1] = 1
        head_matrix[o_start+1, s_start+1] = 1
        tail_matrix[s_start, o_start] = 1
        tail_matrix[o_start, s_start] = 1
        span_matrix[s_start+1, s_end] = 1
        span_matrix[s_end, s_start+1] = 1
        span_matrix[o_start+1, o_end] = 1
        span_matrix[o_end, o_start+1] = 1

        # Entity-Relation (Subject)
        head_matrix[s_start+1, pred_shifted_idx] = 1
        tail_matrix[s_end, pred_shifted_idx] = 1
        span_matrix[s_start+1, pred_shifted_idx] = 1
        span_matrix[s_end, pred_shifted_idx] = 1
        span_matrix[o_start+1, pred_shifted_idx] = 1
        span_matrix[o_end, pred_shifted_idx] = 1

        # Relation-Entity (Object)
        head_matrix[pred_shifted_idx, o_start+1] = 1
        tail_matrix[pred_shifted_idx, o_end] = 1
        span_matrix[pred_shifted_idx, o_start+1] = 1
        span_matrix[pred_shifted_idx, o_end] = 1
        span_matrix[pred_shifted_idx, s_start+1] = 1
        span_matrix[pred_shifted_idx, s_end] = 1

        e2e.add((s_start, o_start))
        e2e.add((o_start, s_start))
        h2r.add((s_start, pred_shifted_idx))
        t2r.add((o_start, pred_shifted_idx))
    
    labels["head_matrices"].append(head_matrix)
    labels["tail_matrices"].append(tail_matrix)
    labels["span_matrices"].append(span_matrix)
    labels["spo_span"].append(spo_span)
    labels["spo_text"].append(spo_text)


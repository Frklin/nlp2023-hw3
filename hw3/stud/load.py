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
    

class RelationDataset(Dataset):

    def __init__(self, data_path, tokenizer='bert-base-uncased'): 
        self.data_path = data_path
        self.tokens = []
        self.relations = []
        self.tokenizer = tokenizer
        self.rel_inputs = tokenizer.encode_plus(config.rel_abrv, add_special_tokens=False)
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.tokens.append(data['tokens'])
                for relation in data['relations']:
                    subject = Entity(relation["subject"]['start_idx'], relation["subject"]['end_idx'],relation["subject"]['text'])
                    object = Entity(relation["object"]['start_idx'], relation["object"]['end_idx'],relation["object"]['text'])
                    self.relations.append(Relation(
                        subject=subject,
                        relation=relation['relation'],
                        object=object
                    ))

    def __len__(self):
        return len(self.tokens)


    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        inputs = self.tokenizer.encode_plus(tokens, max_length=config.MAX_LEN, truncation=True, padding='max_length')
        sep_idx = inputs["input_ids"].index(self.tokenizer.sep_token_id)
        
        input_ids = inputs["input_ids"] + self.rel_inputs['input_ids'] 
        input_ids[sep_idx] = self.tokenizer.sep_token_id

        attention_mask = inputs["attention_mask"] + [1] * config.REL_NUM
        attention_mask[sep_idx] = 0

        token_type_ids = inputs["token_type_ids"] + [1] * config.REL_NUM

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "relations": self.relations[idx]
        }



def parseRelation2Id(data_path):
    with open(data_path, 'r') as f:
        relation2Id = json.load(f)
    return relation2Id
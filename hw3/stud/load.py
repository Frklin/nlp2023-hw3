from torch.utils.data import Dataset
import torch
import config
import json


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
                if l == 180:
                    break
                l+=1
                data = json.loads(line)
                self.tokens.append(data['tokens'])
                self.relations.append(data["relations"])


    def __len__(self):
        return len(self.tokens)


    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        inputs = self.tokenizer(" ".join(tokens), max_length=config.MAX_LEN+2, truncation=True, padding='max_length')
        sep_idx = inputs["input_ids"].index(self.tokenizer.sep_token_id)
        
        input_ids = inputs["input_ids"] + self.encoded_preds['input_ids'] 
        input_ids[sep_idx] = self.tokenizer.pad_token_id

        attention_mask = inputs["attention_mask"] + [1] * config.REL_NUM
        attention_mask[sep_idx] = 0

        token_type_ids = inputs["token_type_ids"] + [1] * config.REL_NUM

        position_ids = inputs.word_ids()
        set_position_shift(idx, position_ids)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        return idx, input_ids, attention_mask, token_type_ids, position_ids, self.relations[idx]


def parseRelation2Id(data_path):
    with open(data_path, 'r') as f:
        relation2Id = json.load(f)
        relation2Id = {k: v for k, v in sorted(relation2Id.items(), key=lambda item: item[1])}
    return relation2Id

def parseRel2Pred(data_path):
    with open(data_path, 'r') as f:
        rel2pred = json.load(f)
    return rel2pred

def set_position_shift(idx, position_ids):
    if idx not in config.index_shift:
        config.index_shift["idx"] = position_ids#.append({"idx": idx, "pos_ids": position_ids})
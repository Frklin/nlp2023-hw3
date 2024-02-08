from torch.utils.data import Dataset, IterableDataset
import torch
import config
import json


class RelationDataset(IterableDataset):

    def __init__(self, data_path, tokenizer): 
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
        config.encoded_preds = torch.tensor(self.encoded_preds['input_ids'])
        with open(self.data_path, 'r') as f:
            l = 0
            for line in f:
                # if l == 10:
                #     break
                # l+=1
                data = json.loads(line)
                self.tokens.append(data['tokens'])
                self.relations.append(data["relations"])

    def __iter__(self):
        for idx in range(len(self.tokens)):
            tokens = ['[CLS]']
            tokens.extend(self.tokens[idx])

            input_ids = []
            position_ids = [-1]
            for i in range(len(tokens)):
                encoded_token = self.tokenizer.encode(tokens[i], add_special_tokens=False)
                input_ids.extend(encoded_token)
                position_ids.extend([i] * len(encoded_token))
            
            set_position_shift(idx, position_ids)

            input_ids = torch.tensor(input_ids)
            position_ids = torch.tensor(position_ids)

            yield idx, input_ids, position_ids, self.relations[idx]



    def __len__(self):
        return len(self.tokens)


    # def __getitem__(self, idx):
    #     tokens = ['[CLS]']
    #     tokens.extend(self.tokens[idx])

    #     input_ids = []
    #     position_ids = [None]
    #     for i in range(len(tokens)):
    #         encoded_token = self.tokenizer.encode(tokens[i], add_special_tokens=False)
    #         input_ids.extend(encoded_token)
    #         position_ids.extend([i] * len(encoded_token))

    #     tokens_ids_len = len(input_ids)
        
    #     attention_mask = [1] * tokens_ids_len + [0] * (config.MAX_LEN - tokens_ids_len) + [1] * config.REL_NUM
    #     token_type_ids = [0] * config.MAX_LEN + [1] * config.REL_NUM

    #     input_ids = input_ids + [0] * (config.MAX_LEN - tokens_ids_len) + self.encoded_preds['input_ids']


    #     set_position_shift(idx, position_ids)

    #     input_ids = torch.tensor(input_ids)
    #     attention_mask = torch.tensor(attention_mask)
    #     token_type_ids = torch.tensor(token_type_ids)

    #     return idx, input_ids, attention_mask, token_type_ids, position_ids, self.relations[idx]


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
import sys
sys.path.append("../")
sys.path.append("hw3")
sys.path.append("hw3/stud")

from typing import List, Dict
from transformers import BertTokenizerFast
from UniRel import UniRE
from transformers import BertConfig
from torch.utils.data import DataLoader
from model import Model
from load import set_position_shift
import config
import torch


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel()


class Baseline(Model):

    preds = [{"subject": {"start_idx": 24,
                          "end_idx": 25},
              "relation": "/location/neighborhood/neighborhood_of",
              "object": {"start_idx": 1,
                         "end_idx": 2}},
             {"subject": {"start_idx": 1,
                          "end_idx": 2},
              "relation": "/location/location/contains",
              "object": {"start_idx": 24,
                         "end_idx": 25}},
             {"subject": {"start_idx": 4,
                          "end_idx": 6},
              "relation": "/people/person/place_lived",
              "object": {"start_idx": 8,
                         "end_idx": 9}}
             ]

    def predict(self, tokens: List[List[str]]) -> List[List[Dict]]:
        return [self.preds for _ in tokens]


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(config.PRETRAINED_MODEL, do_basic_tokenize=True)
        self.rel2pred_str = " ".join([rel for rel in config.rel2pred.values()])
        self.encoded_preds = self.tokenizer.encode_plus(self.rel2pred_str, add_special_tokens=False)
        self.bert_config = BertConfig.from_pretrained(config.PRETRAINED_MODEL, finetuning_task="UniRel")
        self.bert_config.num_rels = config.REL_NUM
        self.bert_config.num_labels = config.REL_NUM
        self.bert_config.threshold = config.THRESHOLD
        # self.is_additional_att = False
        # self.is_separate_ablation = False
        # self.test_data_type = False
        self.model = UniRE.load_from_checkpoint(config.LOAD_MODEL_PATH, config=self.bert_config)
        self.model.eval()
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model.to(self.device)
        pass
    
    def predict(self, tokens: List[List[str]]) -> List[List[Dict]]:
        # STUDENT: implement here your predict function

        indices = []
        input_ids = []
        position_ids = []
        attention_masks = []
        token_type_ids = []

        max_len = max([len(self.tokenizer.encode(" ".join(sentence))) for sentence in tokens])

        for idx, sentence in enumerate(tokens):
            sentence_input_ids = [101]
            sentence_position_ids = [-1]
            for i in range(len(sentence)):
                encoded_token = self.tokenizer.encode(sentence[i], add_special_tokens=False)
                sentence_input_ids.extend(encoded_token)
                sentence_position_ids.extend([i] * len(encoded_token))

            set_position_shift(idx, sentence_position_ids)

            # PADDING
            sentence_attention_masks = [1] * len(sentence_input_ids) + [0] * (max_len - len(sentence_input_ids)) + [1] * config.REL_NUM
            sentence_token_type_ids = [0] * len(sentence_input_ids) + [0] * (max_len - len(sentence_input_ids)) + [1] * config.REL_NUM
            sentence_input_ids += [0] * (max_len - len(sentence_input_ids)) + self.encoded_preds['input_ids']
            sentence_position_ids += [-1] * (len(sentence_input_ids) - len(sentence_position_ids))

            indices.append(idx)
            input_ids.append(torch.tensor(sentence_input_ids))
            position_ids.append(torch.tensor(sentence_position_ids))
            attention_masks.append(torch.tensor(sentence_attention_masks))
            token_type_ids.append(torch.tensor(sentence_token_type_ids))

        input_ids = torch.stack(input_ids)
        position_ids = torch.stack(position_ids)
        attention_masks = torch.stack(attention_masks)
        token_type_ids = torch.stack(token_type_ids)
        
        input_data = [indices, input_ids, attention_masks, token_type_ids]

        with torch.no_grad():
            preds = self.model.predict(input_data)

        return preds
    


if __name__ == "__main__":
    # You can test your function here
    first_sentence = ["In", "baseball", ",", "if", "Tom", "Glavine", "shuts", "out", "Atlanta", "for", "the", "Mets", "on", "Monday", "and", "the", "Braves", "come", "back", "the", "next", "day", "and", "knock", "John", "Maine", "out", "of", "the", "game", "in", "the", "first", "inning", "and", "go", "on", "to", "rout", "the", "Mets", ",", "Glavine", "'s", "shutout", "is", "n't", "wiped", "out", "."]
    second_sentence = ["But", "lawmakers", "in", "those", "districts", ",", "including", "Representative", "Roy", "Blunt", "of", "Missouri", ",", "the", "third-ranking", "Republican", "in", "the", "House", ",", "were", "not", "told", "about", "the", "poll", "and", "were", "caught", "off", "guard", "by", "it", "."]

    first_sentence_labels = [{"subject": {"start_idx": 4, "end_idx": 6, "entity_type": "PERSON", "text": "Tom Glavine"}, "relation": "/people/person/place_lived", "object": {"start_idx": 8, "end_idx": 9, "entity_type": "LOCATION", "text": "Atlanta"}}]
    second_sentence_labels =[{"subject": {"start_idx": 8, "end_idx": 10, "entity_type": "PERSON", "text": "Roy Blunt"}, "relation": "/people/person/place_lived", "object": {"start_idx": 11, "end_idx": 12, "entity_type": "LOCATION", "text": "Missouri"}}]

    tokens = [first_sentence, second_sentence]

    model = build_model("cpu")

    print(model.predict(tokens))



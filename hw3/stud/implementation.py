import sys
sys.path.append("../")
sys.path.append("hw3")
sys.path.append("hw3/stud")

from typing import List, Dict
from transformers import BertTokenizerFast
from UniRel import UniRE
from transformers import BertConfig
from model import Model
from load import set_position_shift
import config
import torch

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel().to(device)


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


    def __init__(self):
        # Load the tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(config.PRETRAINED_MODEL, do_basic_tokenize=True)
        self.rel2pred_str = " ".join([rel for rel in config.rel2pred.values()])
        self.encoded_preds = self.tokenizer.encode_plus(self.rel2pred_str, add_special_tokens=False)
        self.bert_config = BertConfig.from_pretrained(config.PRETRAINED_MODEL, finetuning_task="UniRel")
        self.bert_config.num_rels = config.REL_NUM
        self.bert_config.num_labels = config.REL_NUM
        self.bert_config.threshold = config.THRESHOLD
        
        # Load the model
        self.model = UniRE.load_from_checkpoint(config.LOAD_MODEL_PATH, config=self.bert_config)
        self.model.eval()

    def predict(self, tokens: List[List[str]]) -> List[List[Dict]]:
        
        # Get the maximum length of the tokens for the padding
        max_len = max([len(self.tokenizer.encode(" ".join(sentence))) for sentence in tokens])

        # Get the input_ids, attention_masks, token_type_ids
        indices, input_ids, attention_masks, token_type_ids = self.get_input_data(tokens, max_len)       

        input_data = [indices, input_ids, attention_masks, token_type_ids]

        # Get the predictions
        with torch.no_grad():
            preds = self.model.predict(input_data, max_len)

        return preds

    def get_input_data(self, tokens: List[List[str]], max_len: int) -> List[torch.Tensor]:
        """
        Get the input_ids, attention_masks, token_type_ids
        
        Args:
            tokens: List of tokens
            max_len: Maximum length of the tokens for the padding

        Returns:
            indices: List of indices
            input_ids: List of input_ids
            attention_masks: List of attention_masks
            token_type_ids: List of token_type_ids
        """

        indices = []
        input_ids = []
        attention_masks = []
        token_type_ids = []
        config.index_shift = {}

        for idx, sentence in enumerate(tokens):
            sentence_input_ids = [self.tokenizer.cls_token_id]
            sentence_position_ids = [-1]

            # Ecode the tokens and keep track of the position ids
            for i in range(len(sentence)):
                encoded_token = self.tokenizer.encode(sentence[i], add_special_tokens=False)
                sentence_input_ids.extend(encoded_token)
                sentence_position_ids.extend([i] * len(encoded_token))

            # Pad the input_ids, attention_masks, token_type_ids
            sentence_attention_masks = [1] * len(sentence_input_ids) + [0] * (max_len - len(sentence_input_ids)) + [1] * config.REL_NUM
            sentence_token_type_ids = [0] * len(sentence_input_ids) + [0] * (max_len - len(sentence_input_ids)) + [1] * config.REL_NUM
            sentence_input_ids += [0] * (max_len - len(sentence_input_ids)) + self.encoded_preds['input_ids']
            sentence_position_ids += [-1] * (len(sentence_input_ids) - len(sentence_position_ids))

            # Set the position shift to keep track of the position of the tokens
            set_position_shift(idx, sentence_position_ids)

            indices.append(idx)
            input_ids.append(torch.tensor(sentence_input_ids))
            attention_masks.append(torch.tensor(sentence_attention_masks))
            token_type_ids.append(torch.tensor(sentence_token_type_ids))

        # Turn the lists into tensors
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        token_type_ids = torch.stack(token_type_ids)

        return indices, input_ids, attention_masks, token_type_ids

    def to(self, device: str):
        """
        Move the model to the device
        """
        self.model.to(device)
        return self


from torch.utils.data import Dataset, IterableDataset
import torch
import config
import json


class RelationDataset(IterableDataset):
    """
    Class for loading the data of the relation extraction
    
    Args:
        data_path: Path to the data
        tokenizer: Tokenizer for encoding the tokens
    """

    def __init__(self, data_path, tokenizer): 
        self.data_path = data_path
        self.tokens = []
        self.relations = []

        self.tokenizer = tokenizer

        # Load the relations and encoded the predicates
        self.rel2pred_str = " ".join([rel for rel in config.rel2pred.values()])
        self.encoded_preds = tokenizer.encode_plus(self.rel2pred_str, add_special_tokens=False)

        self.data = self.load_data()

    def load_data(self):
        """
        Load the data from the data_path
        """
        config.encoded_preds = torch.tensor(self.encoded_preds['input_ids'])
        with open(self.data_path, 'r') as f:
            l = 0
            for line in f:
                if config.DEBUG and l == 100:
                    break
                l+=1
                data = json.loads(line)
                self.tokens.append(data['tokens'])
                self.relations.append(data["relations"])

    def __iter__(self):
        for idx in range(len(self.tokens)):
            tokens = self.tokens[idx]

            # initialize the input_ids and position_ids with [CLS] token and -1 position
            input_ids = [self.tokenizer.cls_token_id]
            position_ids = [-1]

            # Encode the tokens and keep track of the position ids
            for i in range(len(tokens)):
                encoded_token = self.tokenizer.encode(tokens[i], add_special_tokens=False)
                input_ids.extend(encoded_token)
                position_ids.extend([i] * len(encoded_token))

            # set the position shift to keep track of the position of the tokens
            set_position_shift(idx, position_ids)

            # Turn the input_ids and position_ids into tensor
            input_ids = torch.tensor(input_ids)
            position_ids = torch.tensor(position_ids)

            yield idx, input_ids, position_ids, self.relations[idx]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def parseRelation2Id(data_path: str) -> dict:
    """
    Parse the relation2Id from the data_path

    Args:
        data_path: Path to the relation2Id

    Returns:
        relation2Id: Dictionary of the relation2Id
    """

    with open(data_path, 'r') as f:
        relation2Id = json.load(f)
        relation2Id = {k: v for k, v in sorted(relation2Id.items(), key=lambda item: item[1])}
    return relation2Id

def parseRel2Pred(data_path: str) -> dict:
    """
    Parse the rel2pred from the data_path

    Args:
        data_path: Path to the rel2pred

    Returns:
        rel2pred: Dictionary of the rel2pred with new representation of the predicates
    """
    with open(data_path, 'r') as f:
        rel2pred = json.load(f)
    return rel2pred

def set_position_shift(idx: int, position_ids: list):
    """
    Set the position shift for the tokens to keep track of the position of the tokens (for reconstruction of the sentence later on)

    Args:
        idx: Index of the sentence
        position_ids: List of position_ids
    """

    if idx not in config.index_shift:
        config.index_shift[idx] = position_ids
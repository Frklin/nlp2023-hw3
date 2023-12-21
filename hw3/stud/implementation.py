from typing import List, Dict

from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return Baseline()


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
        # Load your models/tokenizer/etc. that only needs to be loaded once when doing inference
        pass

    def predict(self, tokens: List[List[str]]) -> List[List[Dict]]:
        # STUDENT: implement here your predict function
        pass

from typing import List, Dict


class Model:
    def predict(self, tokens: List[List[str]]) -> List[List[Dict]]:
        """
        A simple wrapper for your model

        Args: tokens: list of list of strings. The outer list represents the sentences, the inner one the tokens
        contained within it. Ex: [ ['Spirit', 'Lake', ',', 'Idaho', '.'], ['Burial', 'was', 'in', 'Queens', ',',
        'New', 'York', '.'] ]

        Returns: list of list of predictions, where a single prediction is a dict containing info about the spans and
        the type of the relation. Ex: [
                                        [
                                            {'subject':
                                                {'start_idx': 3,
                                                 'end_idx': 4},
                                             'relation': '/location/location/contains',
                                             'object':
                                                {'start_idx': 0,
                                                 'end_idx': 2}
                                            }
                                        ],
                                        [
                                            {'subject':
                                                {'start_idx': 5,
                                                 'end_idx': 7},
                                             'relation': '/location/location/contains',
                                             'object':
                                                {'start_idx': 3,
                                                 'end_idx': 4}
                                            }
                                        ]
                                      ]

        """
        raise NotImplementedError

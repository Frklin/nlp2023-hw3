import json
import logging
from rich.progress import track

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

import argparse
import requests
import time

from requests.exceptions import ConnectionError
from typing import Tuple, List, Any, Dict


def flat_relations_types(l: List[List[Dict]]) -> List[str]:
    return [relation_dict["relation"] for sentence_rel_dicts in l for relation_dict in sentence_rel_dicts]


def count(l: List[Any]) -> Dict[Any, int]:
    d = {}
    for e in l:
        d[e] = 1 + d.get(e, 0)
    return d


def read_dataset(path: str) -> Tuple[List[List[str]], List[List[Dict]]]:
    tokens_s, relations_s = [], []

    with open(path) as f:
        for line in f:
            data = json.loads(line)
            tokens_s.append(data["tokens"])
            relations_s.append(data["relations"])

    assert len(tokens_s) == len(relations_s)

    return tokens_s, relations_s


def score(targets: List[List[Dict]], predictions: List[List[Dict]]) -> Tuple[float, float, float]:
    true_positives = 0
    num_golds = 0
    num_preds = 0

    def get_tupled_labels(rel_dicts: List[Dict]) -> List[Tuple[Tuple[int, int], str, Tuple[int, int]]]:
        tupled_labels = []
        for rel_dict in rel_dicts:
            subject_dict, object_dict = rel_dict["subject"], rel_dict["object"]
            tupled_labels.append(
                (
                    (subject_dict["start_idx"], subject_dict["end_idx"]),
                    rel_dict["relation"],
                    (object_dict["start_idx"], object_dict["end_idx"])
                )
            )
        return tupled_labels

    for sent_targets, sent_predictions in zip(targets, predictions):
        sent_targets, sent_predictions = get_tupled_labels(sent_targets), get_tupled_labels(sent_predictions)
        true_positives += len(set(sent_targets).intersection(set(sent_predictions)))
        num_golds += len(sent_targets)
        num_preds += len(sent_predictions)
    precision = true_positives / num_preds if num_preds > 0 else 0.0
    recall = true_positives / num_golds if num_golds > 0 else 0.0

    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return f1_score, precision, recall


def main(test_path: str, endpoint: str, batch_size=32):
    try:
        tokens_s, relations_s = read_dataset(test_path)
    except FileNotFoundError as e:
        logging.error(f"Evaluation crashed because {test_path} does not exist")
        exit(1)
    except Exception as e:
        logging.error(
            f"Evaluation crashed. Most likely, the file you gave is not in the correct format"
        )
        logging.error(f"Printing error found")
        logging.error(e, exc_info=True)
        exit(1)

    max_try = 10
    iterator = iter(range(max_try))

    while True:

        try:
            i = next(iterator)
        except StopIteration:
            logging.error(
                f"Impossible to establish a connection to the server even after 10 tries"
            )
            logging.error(
                "The server is not booting and, most likely, you have some error in build_model or StudentClass"
            )
            logging.error(
                "You can find more information inside logs/. Checkout both server.stdout and, most importantly, "
                "server.stderr"
            )
            exit(1)

        logging.info(f"Waiting 10 second for server to go up: trial {i}/{max_try}")
        time.sleep(10)

        try:
            response = requests.post(
                endpoint, json={"tokens_s": [
                    ["Frodo",
                     "lives",
                     "in",
                     "The",
                     "Shire",
                     ",",
                     "an",
                     "inland",
                     "area",
                     "settled",
                     "by",
                     "Hobbits",
                     "in",
                     "a",
                     "region",
                     "of",
                     "Middle-earth",
                     "."]]}
            ).json()
            response["predictions_s"]
            logging.info("Connection succeded")
            break
        except ConnectionError as e:
            continue
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            logging.error(f"Response was: {response}")
            logging.error(e, exc_info=True)
            exit(1)

    predictions_s = []

    for i in track(range(0, len(tokens_s), batch_size), description="Evaluating"):
        batch = tokens_s[i: i + batch_size]
        try:
            response = requests.post(endpoint, json={"tokens_s": batch}).json()
            predictions_s += response["predictions_s"]
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            logging.error(f"Response was: {response}")
            logging.error(e, exc_info=True)
            exit(1)

    flat_gold_relations_types = flat_relations_types(relations_s)
    flat_pred_relations_types = flat_relations_types(predictions_s)

    gold_relations_distribution = count(flat_gold_relations_types)
    pred_relations_distribution = count(flat_pred_relations_types)

    print(f"# gold relations: {len(flat_gold_relations_types)}")
    print(f"# pred relations: {len(flat_pred_relations_types)}")

    keys = set(gold_relations_distribution.keys()) | set(pred_relations_distribution.keys())
    for k in keys:
        print(
            f"\t# {k}: ({gold_relations_distribution.get(k, 0)}, {pred_relations_distribution.get(k, 0)})"
        )

    f1_score, precision, recall = score(relations_s, predictions_s)

    print(f"# f1: {f1_score:.4f}")
    print(f"# precision: {precision:.4f}")
    print(f"# recall: {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=str, help="File containing data you want to evaluate upon"
    )
    args = parser.parse_args()

    main(test_path=args.file, endpoint="http://127.0.0.1:12345")

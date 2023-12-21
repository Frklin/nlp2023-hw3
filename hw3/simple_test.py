from pprint import pp
from typing import List

from stud.implementation import build_model


def main(sentences: List[List[str]]):

    model = build_model("cpu")
    predicted_sentences = model.predict(sentences)

    for sentence, tagged_sentence in zip(sentences, predicted_sentences):
        print(f"# sentence = {sentence}")
        for relation_dict in tagged_sentence:
            pp(relation_dict)
            print()
        print()


if __name__ == "__main__":
    main([["Frodo",
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
           "."]])

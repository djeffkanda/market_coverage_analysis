import argparse

import pandas as pd
import torch

from util import PATHS, generate_dataset, split_by_stop, set_device
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from tqdm.auto import tqdm


def argument_parser():
    """
        A parser ....
    """
    parser = argparse.ArgumentParser(
        usage="\n python ner.py"
              "-pl_url [url] -lp [log-path]"

    )

    parser.add_argument(
        '-pfx',
        '--prefix',
        type=str,
        default='yh',
        required=True
    )

    # parser.add_argument('--filter_video', dest='filter_video', action='store_true')
    # parser.add_argument('--no-filter_video', dest='filter_video', action='store_false')
    # parser.set_defaults(filter_video=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    prefix = args.prefix
    path = PATHS[prefix]
    file_dates, file_txts = generate_dataset(path, prefix=prefix)
    # split text by stops
    new_f_dates, new_f_txts = split_by_stop(file_dates, file_txts)
    # NER
    device = set_device()
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    ner_model = pipeline("ner", model=model, tokenizer=tokenizer, device=device, grouped_entities=True)

    # Create data frame of NER
    ner_df = pd.DataFrame()

    for dt, text in tqdm(zip(new_f_dates, new_f_txts), total=len(new_f_txts)):
        ner = ner_model(text)
        if len(ner) < 1:
            continue
        ner = pd.DataFrame(ner)
        ner['date'] = dt

        ner_df = ner_df.append(ner, ignore_index=True)

    ner_df.to_csv(f'{prefix}_ner.csv', encoding='utf-8')

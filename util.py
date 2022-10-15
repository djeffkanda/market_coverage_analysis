from datetime import datetime, timedelta
import pandas as pd
# import bertopic
import os
import re

import torch

PATHS = dict(
    blw="./data/blw/",
    bsm="./data/bsm/",
    yh="./data/yh/"
)


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def generate_dataset(path, prefix='blw'):
    file_dates = []
    file_txts = []

    os.chdir(path)

    # iterate through all file
    for file in os.listdir():

        # Check whether file is in text format or not
        if file.endswith(".txt") and file.startswith(prefix):

            file_path = f"{path}{file}"

            # call read text file function

            # remove file extention .txt and the starting datetime of the generated text
            new_file_name = re.sub(r'_\d{14}\.txt|.txt', "", file)
            # print(new_file_name)
            fname_split = new_file_name.split('_')
            # print(fnsplit[1])
            f_date = datetime.strptime(fname_split[1], '%Y%m%d')
            if f_date.year < 2020:
                continue

            if len(fname_split) > 2:
                f_date += timedelta(seconds=int(fname_split[2]))

            file_dates.append(f_date)
            file_txts.append(read_text_file(file))

    return file_dates, file_txts


def filter_text_tm(txts: list):
    tmp_txts = []
    for txt in txts:
        # print(len(txt))
        txt = " ".join(re.sub("[^a-zA-Z]+", " ", txt).split())

        tmp_txts.append(txt)
        # print(len(txt), "after")
    return tmp_txts


def split_by_stop(f_dates, f_txts):
    new_dates, new_txts = [], []

    CHUNK_LENTH = 500

    for dt, txt in zip(f_dates, f_txts):

        txt_spit = txt.split('.')
        chunk_txt = ""
        for chunk_item in txt_spit:
            if len(chunk_txt) + len(chunk_item) <= CHUNK_LENTH:
                chunk_txt += " " + chunk_item
            else:
                new_txts.append(chunk_txt)
                new_dates.append(dt)
                chunk_txt = chunk_item

        new_txts.append(chunk_txt)
        new_dates.append(dt)

        # new_dates.extend([dt] * len(txt_spit))
        # new_txts.extend(txt_spit)

    return new_dates, new_txts


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

from datetime import datetime, timedelta
import pandas as pd
import bertopic
import os
import re

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

            if len(fname_split) > 2:
                f_date += timedelta(seconds=int(fname_split[2]))

            file_dates.append(f_date)
            file_txts.append(read_text_file(file_path))

    return file_dates, file_txts


def filter_text_tm(txts : list):
  tmp_txts = []
  for txt in txts:
    # print(len(txt))
    txt = " ".join(re.sub("[^a-zA-Z]+", " ", txt).split())

    tmp_txts.append(txt)
    # print(len(txt), "after")
  return tmp_txts
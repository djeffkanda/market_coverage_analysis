import argparse
import datetime
import os, re
from pytube import YouTube, Playlist

import torch
import whisper
from whisper.utils import write_vtt
import numpy as np
from pathlib import Path
from collections import defaultdict


def argument_parser():
    """
        A parser ....
    """
    parser = argparse.ArgumentParser(
        usage="\n python extract.py"
              "-pl_url [url] -lp [log-path]"

    )
    parser.add_argument(
        '--pl_url',
        type=str,
        help='url to the youtube playlist',
        required=True
    )

    parser.add_argument(
        '-pfx',
        '--prefix',
        type=str,
        default='yh',
        required=True
    )

    parser.add_argument('--filter_video', dest='filter_video', action='store_true')
    parser.add_argument('--no-filter_video', dest='filter_video', action='store_false')
    parser.set_defaults(filter_video=False)

    return parser.parse_args()


def to_snake_case(name):
    return name.lower().replace(" ", "_").replace(":", "_").replace("__", "_")


def download_youtube_audio(url, out_fname=None, out_dir=".", best_quality=True):
    "Download the audio from a YouTube video"
    yt = YouTube(url)
    if out_fname is None:
        out_fname = os.path.join(out_dir, to_snake_case(yt.title) + ".mp4")
    yt = (yt.streams
          .filter(only_audio=True, file_extension="mp4")
          .order_by("abr"))
    if best_quality:
        yt = yt.desc()
    else:
        yt = yt.asc()
    return yt.first().download(filename=out_fname)


def transcribe_file(model, file, language="en"):
    "Run whisper on an audio file and save results"
    print(f"Transcribing file: {file}")
    file = Path(file)
    result = model.transcribe(str(file), verbose=False, language=language)

    # save TXT
    with open(file.with_suffix(".txt"), "w", encoding="utf-8") as txt:
        print(result["text"], file=txt)


def read_urls_from_log(file):
    urls = []
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
            urls = [line.split(' -- ')[0] for line in lines]
    except Exception as e:
        pass

    return urls


if __name__ == "__main__":
    args = argument_parser()

    ## CUDA is strongly recommended
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    ## load the model
    model = whisper.load_model("base.en").to(DEVICE)

    _URL = args.pl_url  # "https://youtu.be/m5XVb1F9o7k?list=PLGaYlBJIOoa9ubuZWkdlPKN-XGVvWGlBe"
    yt_playlist = Playlist(_URL)

    keyword = 'Coverage'
    video_ind = defaultdict(int)
    PREFIX = args.prefix
    LOG_FILE_NAME = f"log_{PREFIX}_finance.txt"

    dt_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    processed_urls = read_urls_from_log(LOG_FILE_NAME)

    with open(LOG_FILE_NAME, 'a') as f:
        for url, video in zip(yt_playlist.video_urls, yt_playlist.videos):
            title = video.title

            if (keyword in title or not args.filter_video) and not url in processed_urls:
                date = str(video.publish_date)
                date = date.replace('-', '')
                date = date.replace(' 00:00:00', '')

                video_ind[date] += 1
                # print(f"{date} ==> {video_ind[date]}")

                date = date + ("_" + str(video_ind[date] - 1) if video_ind[date] > 1 else "") + f"_{dt_str}"
                print(url + " -- " + title + " " + date)

                try:
                    file = download_youtube_audio(url, out_fname=f"{PREFIX}_{date}.mp4")
                    _ = transcribe_file(model, file)
                    # print(f"we are here {file}")
                    f.write(f"{url} -- {file}\n")
                    os.remove(file)
                except Exception as e:
                    print(f'skipped sentence due to :: {e}')
                    pass

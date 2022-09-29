import argparse


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
        '-lp',
        '--log_path',
        type=str,
        help='Path to the log of processed videos',
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

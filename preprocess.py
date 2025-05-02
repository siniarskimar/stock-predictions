import sys
from argparse import ArgumentParser

import pandas as pd


DATETIME_FORMAT = "%d-%m-%Y"
CSV_QUOTE = '"'
CSV_SEP = ','
CSV_DECIMALPOINT = '.'


def argument_parser():
    p = ArgumentParser(prog="preprocess.py")
    p.add_argument('raw_csv', help="Filepath to CSV to be preprocessed")
    p.add_argument('-t', '--training',
                   help="Filepath to training file destination")

    p.add_argument('-i', '--inferrence',
                   help="Filepath to inferrence file destination")

    p.add_argument('-s', '--split',
                   help="Percentage split given to testing",
                   type=int, default=70)
    return p


def main():
    argparser = argument_parser()
    if len(sys.argv) == 1:
        argparser.print_help()
        return

    args = argparser.parse_args()
    if args.training is None and args.inferrence is None:
        print('err: no destination file given')
        exit(1)

    if not 0 <= args.split <= 100:
        print(f'data split is defined outside of range [0; 100], got {args.split}')
        exit(1)

    training_split = float(args.split) / 100
    input_csv = pd.read_csv(
        sys.argv[1],
        sep=CSV_SEP,
        quotechar=CSV_QUOTE,
        decimal=CSV_DECIMALPOINT,
        date_format=DATETIME_FORMAT
    )

    input_csv['Date'] = pd.to_datetime(
        input_csv['Date'], format=DATETIME_FORMAT)

    input_csv = input_csv.sort_values('Date')
    row_count = len(input_csv)
    training_row_count = int(row_count * training_split)

    if args.training is not None:
        training_data = input_csv.iloc[:training_row_count]
        training_data.to_csv(args.training,
                             sep=',',
                             quotechar='"',
                             decimal='.',
                             date_format=DATETIME_FORMAT,
                             index=False)

    if args.inferrence is not None:
        inferrence_data = input_csv.iloc[training_row_count:]
        inferrence_data.to_csv(args.inferrence,
                               sep=',',
                               quotechar='"',
                               decimal='.',
                               date_format=DATETIME_FORMAT,
                               index=False)


if __name__ == "__main__":
    main()

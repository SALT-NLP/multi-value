import argparse
import csv
import json
import os
import pickle as pkl
import sys
from glob import glob

import pandas as pd
from tqdm import tqdm

from .Dialects import AfricanAmericanVernacular, IndianDialect


csv.field_size_limit(min(sys.maxsize, 2147483646))


def build_df_row_by_row(fn, sep="\t"):
    with open(fn, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
        df = pd.DataFrame(columns=[x.strip() for x in lines[0].split(sep)])
        for i, line in enumerate(lines[1:]):
            df.loc[i + 1] = [x.strip() for x in line.split(sep)]
    return df


def convert(args, D, directory):

    def write_header(row, keys, outfile):
        out = []
        mod = []
        for key in row:
            if (("sentence" in key) or ("question" in key)) and ("parse" not in key):
                mod.extend(["%s-%s" % (key, which) for which in sorted(keys)])
                out.append("%s-glue" % key)
                if args.html:
                    out.append("%s-glue-html" % key)
            out.append(str(key))
        outfile.write("\t".join(out + mod) + "\n")

    for in_fn in sorted(glob(os.path.join(args.GLUE, directory) + "/*.tsv")):
        print(in_fn)
        basename = os.path.basename(in_fn)

        out_dir = os.path.join(args.VALUE, directory)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_fn = os.path.join(out_dir, basename)

        if os.path.exists(out_fn):
            print("already wrote", out_fn)
            continue

        print("starting %s" % out_fn)
        with open(out_fn, "w") as outfile:
            wrote_header = False
            header = []
            total_length = len(
                list(csv.reader(open(in_fn, "r", encoding="utf-8-sig"), delimiter="\t", quotechar=None))
            )
            with open(in_fn, "r", encoding="utf-8-sig") as infile:
                read_tsv = csv.reader(infile, delimiter="\t", quotechar=None)
                for row in tqdm(read_tsv, total=total_length):
                    if not wrote_header:
                        header = row
                        write_header(row, D.modification_counter.keys(), outfile)
                        wrote_header = True
                    else:
                        modifications = {}
                        out = []
                        if len(header) != len(row):
                            print("failed", row)
                        for i, key in enumerate(header):
                            if (("sentence" in key) or ("question" in key)) and ("parse" not in key) and i < len(row):
                                revised = D.convert_sae_to_dialect(row[i])
                                modifications[key] = D.modification_counter
                                out.append(row[i])
                                if args.html:
                                    out.append(D.highlight_modifications_html())
                                out.append(revised)
                            elif i < len(row):
                                out.append(str(row[i]))
                            else:
                                print("failed", row)

                        outfile.write(
                            "\t".join(
                                out
                                + [
                                    str(modifications[which][key])
                                    for which in sorted(modifications.keys())
                                    for key in sorted(modifications[which].keys())
                                ]
                            )
                            + "\n"
                        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GLUE", default="data/GLUE", help="the directory where GLUE datasets are found")
    parser.add_argument("--VALUE", default="data/VALUE", help="the directory where VALUE datasets will be saved")
    parser.add_argument("--dialect", default="aave", help="the directory where VALUE datasets will be saved")
    parser.add_argument(
        "--lexical_mapping",
        default="NONE",
        help="a pickle file containing the lexical mapping from sae to the dialect",
    )
    parser.add_argument(
        "--morphosyntax", action="store_true", help="set this flag to include morphosyntactic transformations"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="set this flag to add a column for HTML modification highlighting and tagging",
    )
    parser.add_argument("--all", action="store_true", help="set this flag to build WNLI")
    parser.add_argument("--CoLA", action="store_true", help="set this flag to build COLA")
    parser.add_argument("--MNLI", action="store_true", help="set this flag to build MNLI")
    parser.add_argument("--QNLI", action="store_true", help="set this flag to build QNLI")
    parser.add_argument("--RTE", action="store_true", help="set this flag to build RTE")
    parser.add_argument("--SST", action="store_true", help="set this flag to build SST-2")
    parser.add_argument("--STS", action="store_true", help="set this flag to build STS-B")
    parser.add_argument("--SNLI", action="store_true", help="set this flag to build SNLI")
    parser.add_argument("--QQP", action="store_true", help="set this flag to build QQP")
    parser.add_argument("--WNLI", action="store_true", help="set this flag to build WNLI")
    args = parser.parse_args()

    mapping = {}
    if os.path.exists(args.lexical_mapping):
        with open(args.lexical_mapping, "rb") as infile:
            mapping = pkl.load(infile)
    dialect_choice = args.dialect.lower()
    if dialect_choice == "aave":
        D = AfricanAmericanVernacular(mapping, morphosyntax=args.morphosyntax)
    elif dialect_choice == "indian":
        D = IndianDialect(mapping, morphosyntax=args.morphosyntax)
    else:
        print("Dialect {} Unimplemented".format(dialect))
        sys.exit()

    D.clear()
    if args.all:
        for fn in sorted(glob(args.GLUE + "/*")):
            dir_name = os.path.basename(fn)
            print(fn, dir_name)
            convert(args, D, dir_name)

    if args.CoLA:
        convert(args, D, "CoLA")
    if args.MNLI:
        convert(args, D, "MNLI")
    if args.QNLI:
        convert(args, D, "QNLI")
    if args.QQP:
        convert(args, D, "QQP")
    if args.RTE:
        convert(args, D, "RTE")
    if args.SNLI:
        convert(args, D, "snli_1.0")
    if args.SST:
        convert(args, D, "SST-2")
    if args.STS:
        convert(args, D, "STS-B")
    if args.WNLI:
        convert(args, D, "WNLI")


if __name__ == "__main__":
    main()

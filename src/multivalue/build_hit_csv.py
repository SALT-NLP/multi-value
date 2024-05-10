import glob, os
import pandas as pd
from Dialects import AfricanAmericanVernacular
import csv


def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding="utf-8-sig") as f:
        data = list(csv.reader(f, delimiter="\t", quotechar=quotechar))
        return pd.DataFrame(data[1:], columns=data[0])


D = AfricanAmericanVernacular({})

collected = []
for fn in glob.glob("../data/VALUE_no_ass_or_lex/*"):
    #     if 'STS-B' in fn:
    #         print('skipping STS-B')
    #         continue

    bn = os.path.basename(fn)
    print(bn)
    df = pd.concat([read_tsv(fn) for fn in glob.glob("../data/VALUE_no_ass_or_lex/%s/*.tsv" % bn)])

    if "acceptability" in df.columns:
        print("considering only acceptable instances for", bn)
        df = df[df["acceptability"] == 1].copy()

    for consider_cat in D.modification_counter.keys():
        print("\t", consider_cat)

        consider_cols = ["-".join(col.split("-")[:-1]) for col in df.columns if consider_cat in col]

        for consider_col in consider_cols:
            print("\t\t", consider_col)
            consider = df[df["%s-%s" % (consider_col, consider_cat)].astype(float) > 0]
            N = int(30 / len(consider_cols))
            if N > len(consider):
                N = len(consider)
                print("cut", consider_cat, consider_col, "short to", N, "samples")
            consider = consider.sample(n=N, random_state=50).copy()
            renamed_columns = {
                "%s-%s" % (consider_col, ccat): ccat.replace("/", "_") for ccat in D.modification_counter.keys()
            }
            renamed_columns[consider_col + "-glue"] = "sae"
            renamed_columns[consider_col + "-glue-html"] = "sae_highlight"
            renamed_columns[consider_col] = "aave"
            consider.rename(columns=renamed_columns, inplace=True)
            consider = consider[
                ["sae", "sae_highlight", "aave"]
                + list(set(renamed_columns.values()).difference({"sae", "sae_highlight", "aave"}))
            ].copy()
            consider["dataset"] = bn
            collected.append(consider)
            df.drop(consider.index, inplace=True)

collected = pd.concat(collected)
collected.drop_duplicates(subset="sae", inplace=True)
collected = collected.sample(frac=1, random_state=7)
for i in range(int(len(collected) / 100) + 1):
    start = 100 * i
    end = min(start + 100, len(collected))
    if not os.path.exists("../hit/HIT_input/"):
        os.makedirs("../hit/HIT_input/")
    collected.iloc[start:end].to_csv("../hit/HIT_input/%s_%s.csv" % (start, end), index=False)

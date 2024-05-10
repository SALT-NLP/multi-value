import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--A", default=1.0, help="The probability of executing a feature with attestation level A")
    parser.add_argument("--B", default=0.6, help="The probability of executing a feature with attestation level B")
    parser.add_argument("--C", default=0.3, help="The probability of executing a feature with attestation level C")
    args = parser.parse_args()

    ewave = pd.read_csv("resources/ewave.csv")
    language_vectors = {key: np.zeros(max(ewave["parameter_pk"]) + 1) for key in sorted(set(ewave["language_name"]))}
    for _, row in ewave.iterrows():
        pct = 0.0
        if row["attestation"] == "A":
            pct = args.A
        elif row["attestation"] == "B":
            pct = args.B
        elif row["attestation"] == "C":
            pct = args.C
        language_vectors[row["language_name"]][row["parameter_pk"]] = pct
    df = pd.DataFrame(language_vectors)
    df.index.rename("feature_id", inplace=True)
    df.to_csv("resources/attestation_vectors.csv")

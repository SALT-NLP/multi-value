import sys
import os
import pickle as pkl

import argparse
from src.Dialects import AfricanAmericanVernacular


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A REPL for the VALUE Transformations")
    parser.add_argument("--transform", type=str, default=None, required=False)
    args = parser.parse_args()

    if args.transform == "aave_like":
        mapping = {}
        converter = AfricanAmericanVernacular(mapping, morphosyntax=True)
    else:
        print("Please select a dialect with a supported set of transformations!")
        sys.exit()

    while True:
        example = input("Enter an input in SAE: ")
        print(
            "After "
            + converter.dialect_name
            + ": "
            + converter.convert_sae_to_dialect(example)
        )
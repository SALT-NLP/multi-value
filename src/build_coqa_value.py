import argparse
import os, sys
from tqdm import tqdm
from .Dialects import AfricanAmericanVernacular, IndianDialect, ColloquialSingaporeDialect, ChicanoDialect, AppalachianDialect, MultiDialect
import pandas as pd
import numpy as np
import pickle as pkl
from glob import glob
import json
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/CoQA')
    parser.add_argument('--dialect', default='indian')
    parser.add_argument('--lexical_mapping', default='NONE', help='a pickle file containing the lexical mapping from sae to the dialect')
    parser.add_argument('--morphosyntax', action='store_true', help='set this flag to include morphosyntactic transformations')
    parser.add_argument('--html', action='store_true', help='set this flag to add a column for HTML modification highlighting and tagging')
    args = parser.parse_args()
    
    mapping = {}
    if os.path.exists(args.lexical_mapping):
        with open(args.lexical_mapping, 'rb') as infile:
            mapping = pkl.load(infile)
    dialect_choice = args.dialect.lower()
    if dialect_choice == "aave":
        D = AfricanAmericanVernacular(mapping, morphosyntax=args.morphosyntax)
    elif dialect_choice == "indian":
        D = IndianDialect(mapping, morphosyntax=args.morphosyntax)
    elif dialect_choice == "singapore":
        D = ColloquialSingaporeDialect(mapping, morphosyntax=args.morphosyntax)
    elif dialect_choice == "chicano":
        D = ChicanoDialect(mapping, morphosyntax=args.morphosyntax)
    elif dialect_choice == "appalachian":
        D = AppalachianDialect(mapping, morphosyntax=args.morphosyntax)
    elif dialect_choice == "multi":
        D = MultiDialect(mapping, morphosyntax=args.morphosyntax)
    else:
        print("Dialect {} Unimplemented".format(dialect))
        sys.exit()
        
    D.clear()
    for fn in sorted(glob(os.path.join(args.input,  '*.json'))):
        bn = os.path.basename(fn)
        print(fn, bn)

        with open(fn, 'r') as infile:
            data = json.load(infile)
            
            for datapoint in tqdm(data['data']):
                count = 0
                for q in datapoint['questions']:
                    q['sae_input_text'] = q['input_text']
                    q['input_text'] = D.convert_sae_to_dialect(q['input_text'])
                for a in datapoint['answers']:
                    a['dialect_input_text'] = D.convert_sae_to_dialect(a['input_text'])
                datapoint['dialect_story'] = D.convert_sae_to_dialect(datapoint['story'])

            output = f"data/CoQA/{D.dialect_code}_CoQA"
                
            if not os.path.exists(output):
                os.makedirs(output)
                
            with open( os.path.join(output, bn), 'w') as outfile:
                json.dump(data, outfile)
            
if __name__ == "__main__":  
    main()
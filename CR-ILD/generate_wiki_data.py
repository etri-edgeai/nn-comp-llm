from tqdm.auto import tqdm
from datasets import load_dataset
import nltk
from nltk import tokenize

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os

def main():
    parser = ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./wikipedia")
    parser.add_argument("--corpus_dir", type=str, default="./wikipedia")
    parser.add_argument("--data_dir", type=str, default="./glue_data")
    parser.add_argument("--num_docs", type=int, default=1000)
    parser.add_argument("--num_data", type=int, default=100000)
    args = parser.parse_args()
    
    nltk.download('punkt')
    os.makedirs(args.corpus_dir, exist_ok=True)
    
    idx = 0
    wiki = load_dataset('wikipedia', '20200501.en', split='train', cache_dir=args.cache_dir)
    with open(os.path.join(args.corpus_dir, "corpus.txt"), "w", encoding="utf-8") as fp:
        for document in tqdm(wiki):
            if np.random.random() < 0.7:
                document = document["text"].replace("\n", " ")
                document = tokenize.sent_tokenize(document)
                for sentence in document:
                    fp.write(sentence)
                    fp.write("\n")
                fp.write("\n")
                idx += 1
                
            if idx == max(args.num_docs - 1, args.num_data / 5):
                break
               
    # Generate Dataset for Fine-tuning 
    file = open(os.path.join(args.corpus_dir, "corpus.txt"), 'r')
    lines = file.readlines()
    
    dataset = []
    data = []
    for line in lines:
        if line != '\n':
            if len(line.split()) > 10:
                data.append(line[:-1])
            if len(data) == 2:
                dataset.append(data)
                data = []
        
        if len(dataset) == args.num_data:
            break
            
    os.makedirs(os.path.join(args.data_dir, 'WIKI/'), exist_ok=True)
    pd.DataFrame(dataset).to_csv(os.path.join(args.data_dir, 'WIKI/train.tsv'),
                                 sep='\t', index=False)
            
if __name__ == "__main__":
    main()
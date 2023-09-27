"""
    Create paraphrasing data using GPT API.
"""
import argparse
import os
import openai
import torch
from torch.utils.data import DataLoader, Subset
import datasets
from tqdm import tqdm
import pickle

PROMPT = 'Please rephrase the following document: \n'
NUM_PARAPHRASING = 4
TEST_SIZE = 300
SEED = 42
TEMP = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_ai_api_key", type=str, help="OpenAI API key")
    parser.add_argument("--subset", type=bool, default=False,
                        help="Weather to run on a subset of the dataset or on all test set.")
    args = parser.parse_args()
    return args


def create_paraphrase_on_dataset(open_ai_api_key, subset=False):
    openai.api_key = open_ai_api_key
    # Note the default temperature is 1.

    dataset = datasets.load_dataset('xsum', split='test', cache_dir='./data')
    if subset:
        results_filename = f'paraphrased_gpt_seed{SEED}_temp{TEMP}'
        sub_dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)['test']
    else:
        results_filename = f'paraphrased_gpt_temp{TEMP}'
        sub_dataset = dataset

    paraphrased_docs = {}

    for count, sample in tqdm(enumerate(sub_dataset), desc='Building rephrase test subset using GPT', ncols=100,
                              total=len(sub_dataset)):
        doc, summary, idx = sample.values()
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant. Specialized in diverse document rephrasing."},
                    {"role": "user", "content": f"{PROMPT}{doc}"}],
                temperature=TEMP,
                n=NUM_PARAPHRASING
            )
            # I want to write {idx: paraphrasing} which will be a dict, with lists values.
            paraphrasing = [doc] + [response['choices'][i]['message']['content'] for i in range(NUM_PARAPHRASING)]
            paraphrased_docs[idx] = paraphrasing
            if count % 100 == 0:
                with open(f'{results_filename}.pkl', "wb") as pickle_file:
                    pickle.dump(paraphrased_docs, pickle_file)

        except Exception as e:
            print(f"On doc {idx}, exception: {e}")

    with open(f'{results_filename}.pkl', "wb") as pickle_file:
        pickle.dump(paraphrased_docs, pickle_file)

    return paraphrased_docs


if __name__ == '__main__':
    args = parse_args()
    paraphrased_docs = create_paraphrase_on_dataset(args.open_ai_api_key, args.subset)




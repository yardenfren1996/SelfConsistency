import argparse

import datasets
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from llms import Llama_2, Paraphraser

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use.")
    parser.add_argument("--dataset", type=str, default='xsum', help="The dataset to use.")
    parser.add_argument("--results_filename", type=str, default='ours', help="Desired name for the output file.")
    parser.add_argument("--from_index", type=int, default=0, help="Start index of the dataset.")
    parser.add_argument("--to_index", type=int, default=-1, help="End index of the dataset.")
    parser.add_argument("--no_generations", type=int, default=10, help="number of generations per sentence.")
    parser.add_argument("--no_seeds", type=int, default=5, help="number of seeds per document.")
    args = parser.parse_args()
    return args


CONFIG = {'llama_2': {'cls': Llama_2, 'config': {'model_name': "meta-llama/Llama-2-7b-chat-hf"}},
          'paraphraser': {'cls': Paraphraser, 'config': {'model_name': "humarin/chatgpt_paraphraser_on_T5_base"}}}

BATCH_SIZE = 1

if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = datasets.load_dataset(args.dataset, split='test', cache_dir='./data')
    subset_indices = list(range(args.from_index, args.to_index))
    sub_dataset = Subset(dataset, subset_indices)
    del subset_indices

    # Load model
    cls, config = CONFIG[args.model].values()
    config['device'] = device
    LLM = cls(**config)

    data_loader = DataLoader(sub_dataset, batch_size=BATCH_SIZE)

    results = {}
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                doc, sum, id = batch.values()
                paragraphs = doc[0].strip().split('\n')
                results[id[0]] = [LLM.paraphrase(sen, num_return_sequences=args.no_generations) for sen in paragraphs]
                if i % 25 == 0:
                    utils.save_results(results, args.results_filename)

    except Exception as e:
        utils.save_results(results, args.results_filename)
        raise type(e)(f'failed to paraphrase doc {id[0]}, due to: {e}')

    utils.save_results(results, args.results_filename)

import argparse

import datasets
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import utils
from llms import Pegasus, Llama_2, Flan_T5

CONFIG = {'pegasus': {'cls': Pegasus, 'config': {'model_name': "google/pegasus-xsum"}},
          'llama_2': {'cls': Llama_2, 'config': {'model_name': "meta-llama/Llama-2-7b-chat-hf"}},
          'flan_t5': {'cls': Flan_T5, 'config': {'model_name': "google/flan-t5-xl"}}}


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


BATCH_SIZE = 1
if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = datasets.load_dataset(args.dataset, split='test', cache_dir='./data')

    subset_indices = list(range(args.from_index, args.to_index)) if args.to_index > 0 else list(
        range(args.from_index, len(dataset)))
    sub_dataset = Subset(dataset, subset_indices)
    del subset_indices

    # Load model
    cls, config = CONFIG[args.model].values()
    config['device'] = device
    LLM = cls(**config)

    data_loader = DataLoader(sub_dataset, batch_size=BATCH_SIZE)

    print(f'Running self consistency on {len(sub_dataset)} samples.')
    print(f'From index: {args.from_index}, To index: {args.to_index}')
    # results = utils.load_results(f'results/{args.results_filename}.pkl')
    results = {}
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                doc, sum, id = batch.values()
                # if id[0] in results:
                #     continue
                # if id[0] in ['12620805', '26075127', '27667106', '19226042', '34790102', '33548728','24145324','33240318']:
                #     continue
                outputs = LLM.summarize(document=doc, no_generations=args.no_generations, no_seeds=args.no_seeds)
                data = {'outputs': outputs}
                results[id[0]] = data
                if i % 25 == 0:
                    utils.save_results(results, args.results_filename)
    except Exception as e:
        utils.save_results(results, args.results_filename)
        raise type(e)(f'failed to self consistency with id {id[0]}, due to: {e}')

    utils.save_results(results, args.results_filename)

import argparse
import pickle

import datasets
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xsum', help="The dataset to use.")
    parser.add_argument("--results_filename", type=str, default='ours', help="Desired name for the output file.")
    parser.add_argument("--from_index", type=int, default=0, help="Start index of the dataset.")
    parser.add_argument("--to_index", type=int, default=-1, help="End index of the dataset.")
    args = parser.parse_args()
    return args


def rephrase(sentence, T=1):
    prompt = f'Paraphrase the following: "{sentence}" \nParaphrased:'
    with torch.no_grad():
        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**model_inputs, do_sample=True,
                                top_k=10, num_return_sequences=T)

        res = tokenizer.decode(output[0], skip_special_tokens=True).split('Paraphrased:')[1].strip()
    return res


MODEL = "meta-llama/Llama-2-7b-chat-hf"
NUM_GENERATIONS = 5
BATCH_SIZE = 1

if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = datasets.load_dataset(args.dataset, split='test', cache_dir='./data')
    subset_indices = list(range(args.from_index, args.to_index))
    sub_dataset = Subset(dataset, subset_indices)
    del subset_indices

    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir='./models')
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", cache_dir='./models')

    data_loader = DataLoader(sub_dataset, batch_size=BATCH_SIZE)

    results = {}
    try:
        with torch.no_grad():
            for batch in tqdm(data_loader):
                doc, sum, id = batch.values()
                paragraphs = doc[0].strip().split('\n')
                res = []
                for seed in range(NUM_GENERATIONS):
                    torch.manual_seed(seed)
                    res.append('\n'.join([rephrase(sen) for sen in paragraphs]))
                    results[id[0]] = res


    except Exception as e:
        with open(f'results/distributed/{args.results_filename}.pkl', "wb") as pickle_file:
            pickle.dump(results, pickle_file)
        raise type(e)(f'failed to paraphrase doc {id[0]}, due to: {e}')

    with open(f'results/distributed/{args.results_filename}.pkl', "wb") as pickle_file:
        pickle.dump(results, pickle_file)
import argparse
import pickle

import datasets
from rouge_score import rouge_scorer
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoTokenizer, AutoModel

from simmilarity import get_centroid, compute_rouge_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xsum', help="The dataset to use.")
    parser.add_argument("--results_filename", type=str, default='ours', help="Desired name for the output file.")
    parser.add_argument("--from_index", type=int, default=0, help="Start index of the dataset.")
    parser.add_argument("--to_index", type=int, default=-1, help="End index of the dataset.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    NUMBER_OF_GENERATIONS = 25
    BATCH_SIZE = 1

    dataset = datasets.load_dataset(args.dataset, split='test', cache_dir='./data')
    subset_indices = list(range(args.from_index, args.to_index))
    sub_dataset = Subset(dataset, subset_indices)
    del subset_indices

    model_name = "google/pegasus-xsum"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir='./models')
    model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir='./models').to(device)

    bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens',
                                                   cache_dir='./models')
    bert_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', cache_dir='./models')
    scorer = rouge_scorer.RougeScorer(['rouge2'])
    data_loader = DataLoader(sub_dataset, batch_size=BATCH_SIZE)

    results = {}
    with torch.no_grad():
        for batch in tqdm(data_loader):
            doc, sum, id = batch.values()
            batch = tokenizer(doc, truncation=True, padding="longest", return_tensors="pt").to(device)
            translated = model.generate(**batch, num_return_sequences=NUMBER_OF_GENERATIONS,
                                        num_beams=NUMBER_OF_GENERATIONS,
                                        do_sample=True)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
            result = get_centroid(bert_model, bert_tokenizer, tgt_text)
            rogue_2_score = compute_rouge_score(scorer, result, sum[0])
            results[id[0]] = rogue_2_score

    with open(f'results/distributed/{args.results_filename}.pkl', "wb") as pickle_file:
        pickle.dump(results, pickle_file)

import argparse

import datasets
import numpy as np
import torch
import nltk
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from SelfConsistency import utils

nltk.download('punkt')


class BERTSimilarityScore:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir='./models')
        self.model = AutoModel.from_pretrained(self.model_name, cache_dir='./models').to(device)

    def get_similarity_matrix(self, generations):
        try:
            token = {'input_ids': [], 'attention_mask': []}
            for sentence in generations:
                new_token = self.tokenizer.encode_plus(sentence, max_length=128,
                                                       truncation=True, padding='max_length',
                                                       return_tensors='pt')
                token['input_ids'].append(new_token['input_ids'][0])
                token['attention_mask'].append(new_token['attention_mask'][0])

            token['input_ids'] = torch.stack(token['input_ids']).to(self.device)
            token['attention_mask'] = torch.stack(token['attention_mask']).to(self.device)
            # Process tokens through model:
            embeddings = self.model(**token).last_hidden_state
            mask = token['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
            mask_embeddings = embeddings * mask
            summed = torch.sum(mask_embeddings, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / summed_mask
            mean_pooled = mean_pooled.cpu().detach().numpy()
            return cosine_similarity(mean_pooled)
        except Exception as e:
            raise type(e)(f'failed to get_similarity_matrix, due to: {e}')

    def get_centroid(self, generations):
        try:
            out = self.get_similarity_matrix(generations)
            centroid_idx = np.argmax(np.sum(out, axis=1))
            return generations[centroid_idx]
        except Exception as e:
            raise type(e)(f'failed to get_centroid, due to: {e}')

    def compute_score(self, output, reference):
        try:
            pred = self.get_similarity_matrix([output, reference])
            return pred[0, 1]
        except Exception as e:
            raise type(e)(f'failed to compute_score, due to: {e}')


class GTESimilarityScore:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(self.model_name, cache_folder='./models').to(device)

    def get_similarity_matrix(self, generations):
        try:
            embeddings = self.model.encode(generations)
            return cos_sim(embeddings, embeddings)
        except Exception as e:
            raise type(e)(f'failed to get_similarity_matrix, due to: {e}')

    def get_centroid(self, generations):
        try:
            out = self.get_similarity_matrix(generations)
            centroid_idx = torch.argmax(torch.sum(out, dim=1))
            return generations[centroid_idx]
        except Exception as e:
            raise type(e)(f'failed to get_centroid, due to: {e}')

    def compute_score(self, output, reference):
        try:
            pred = self.get_similarity_matrix([output, reference])
            return pred[0, 1].item()
        except Exception as e:
            raise type(e)(f'failed to compute_score, due to: {e}')


class NgramSimilarityScore:
    def __init__(self, N, **kwargs):
        self.N = N

    def get_ngrams(self, text):
        try:
            tokenizer = get_tokenizer('basic_english')
            tokens = tokenizer(text)
            if self.N <= 0 or self.N > len(tokens):
                raise ValueError("Invalid n-gram size")
            ngrams = []

            for i in range(len(tokens) - self.N + 1):
                ngram = tuple(tokens[i:i + self.N])
                ngrams.append(ngram)

            return set(ngrams)
        except Exception as e:
            raise type(e)(f'failed to get_ngrams, due to: {e}')

    def ngram_consistency_score(self, gen_i, gen_j):
        try:
            V_i, V_j = self.get_ngrams(gen_i), self.get_ngrams(gen_j)
            return len(V_i.intersection(V_j)) / (len(V_i.union(V_j)) + 1e-9)
        except Exception as e:
            raise type(e)(f'failed to ngram_consistency_score, due to: {e}')

    def get_similarity_matrix(self, generations):
        try:
            sim_matrix = np.ones((len(generations), len(generations)))
            for i in range(len(generations)):
                for j in range(i + 1, len(generations)):
                    score = self.ngram_consistency_score(generations[i], generations[j])
                    sim_matrix[i, j] = sim_matrix[j, i] = score
            return sim_matrix
        except Exception as e:
            raise type(e)(f'failed to get_similarity_matrix, due to: {e}')

    def get_centroid(self, generations):
        try:
            out = self.get_similarity_matrix(generations)
            centroid_idx = np.argmax(np.sum(out, axis=1))
            return generations[centroid_idx]
        except Exception as e:
            raise type(e)(f'failed to get_centroid, due to: {e}')

    def compute_score(self, output, reference):
        try:
            pred = self.get_similarity_matrix([output, reference])
            return pred[0, 1]
        except Exception as e:
            raise type(e)(f'failed to compute_score, due to: {e}')


class NounNgramSimilarityScore(NgramSimilarityScore):
    def __init__(self, N, **kwargs):
        super().__init__(N)

    def get_ngrams(self, text):
        try:
            tokenizer = get_tokenizer('basic_english')
            text = self.filter_non_nouns(text)
            tokens = tokenizer(text)
            ngrams = []

            for i in range(len(tokens) - self.N + 1):
                ngram = tuple(tokens[i:i + self.N])
                ngrams.append(ngram)

            return set(ngrams)
        except Exception as e:
            raise type(e)(f'failed to get_ngrams, due to: {e}')

    @staticmethod
    def filter_non_nouns(text):
        try:
            tokens = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)

            def is_noun(tag):
                return tag.startswith('N')

            nouns = [token for token, tag in pos_tags if is_noun(tag)]

            filtered_text = ' '.join(nouns)

            return filtered_text

        except Exception as e:
            raise type(e)(f'failed to filter_non_nouns, due to: {e}')


# def GSC(simmialrity_func, generation, set_of_generations):
#     score = 0.0
#     for generation_j in set_of_generations:
#         score += simmialrity_func(generation, generation_j)
#     return (1 / (len(set_of_generations) - 1)) * score
#
#
# # Todo implement
# def weighted_ngram_consistency_score(gen_i, gen_j, n):
#     pass


def compute_rouge_score(scorer, predictions, references, rouge_type='rouge2'):
    return scorer.score(predictions, references)[rouge_type].precision


CONFIG = {
    'bert': {'cls': BERTSimilarityScore, 'config': {'model_name': 'sentence-transformers/bert-base-nli-mean-tokens'}},
    'gte': {'cls': GTESimilarityScore, 'config': {'model_name': 'thenlper/gte-large'}},
    'ngram': {'cls': NgramSimilarityScore, 'config': {'N': 1}},
    'noun_ngram': {'cls': NounNgramSimilarityScore, 'config': {'N': 1}}}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use.")
    parser.add_argument("--dataset", type=str, default='xsum', help="The dataset to use.")
    parser.add_argument("--data", type=str, help="Results pkl file to use.")
    parser.add_argument("--results_filename", type=str, default='ours', help="Desired name for the output file.")
    args = parser.parse_args()
    return args


BATCH_SIZE = 1

if __name__ == '__main__':
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = datasets.load_dataset(args.dataset, split='test', cache_dir='./data')
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Load model
    cls, config = CONFIG[args.model].values()
    config['device'] = device
    score_model = cls(**config)

    # Load summarization results data
    res_data = utils.load_results(args.data)

    scorer = rouge_scorer.RougeScorer(['rouge2'])
    try:
        with torch.no_grad():
            for batch in tqdm(data_loader):
                doc, sum, id = batch.values()
                sum_model_outputs = res_data[id[0]]['outputs']
                centroid = score_model.get_centroid(sum_model_outputs) if len(sum_model_outputs) > 1 else \
                    sum_model_outputs[0]
                score = score_model.compute_score(centroid, sum[0])
                rouge_score = compute_rouge_score(scorer, centroid, sum[0])

                res_data[id[0]][f'{args.model}_centroid'] = centroid
                res_data[id[0]][f'{args.model}_score_centroid'] = score
                res_data[id[0]][f'rouge2_{args.model}_centroid'] = rouge_score
    except Exception as e:
        raise type(e)(f'failed to simmilarity with id {id[0]}, due to: {e}')

    utils.save_results(res_data, args.results_filename)

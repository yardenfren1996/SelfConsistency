import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torchtext.data.utils import get_tokenizer


def GSC(simmialrity_func, generation, set_of_generations):
    score = 0.0
    for generation_j in set_of_generations:
        score += simmialrity_func(generation, generation_j)
    return (1 / (len(set_of_generations) - 1)) * score


def get_ngrams(text, n):
    # Tokenize the input text using PyTorch's get_tokenizer
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(text)
    if n <= 0 or n > len(tokens):
        raise ValueError("Invalid n-gram size")
    ngrams = []

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)

    return set(ngrams)


def ngram_consistency_score(gen_i, gen_j, n):
    V_i, V_j = get_ngrams(gen_i, n), get_ngrams(gen_j, n)
    return (1 / len(V_i.union(V_j))) * len(V_i.intersection(V_j))


# Todo implement
def weighted_ngram_consistency_score(gen_i, gen_j, n):
    pass


def compute_rouge_score(scorer, predictions, references, rouge_type='rouge2'):
    return scorer.score(predictions, references)[rouge_type].precision


def BERT_simmilarity_score(bert_model, bert_tokenizer, generations):
    token = {'input_ids': [], 'attention_mask': []}
    for sentence in generations:
        new_token = bert_tokenizer.encode_plus(sentence, max_length=128,
                                               truncation=True, padding='max_length',
                                               return_tensors='pt')
        token['input_ids'].append(new_token['input_ids'][0])
        token['attention_mask'].append(new_token['attention_mask'][0])

    token['input_ids'] = torch.stack(token['input_ids'])
    token['attention_mask'] = torch.stack(token['attention_mask'])

    # Process tokens through model:
    embeddings = bert_model(**token).last_hidden_state
    mask = token['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    mask_embeddings = embeddings * mask
    summed = torch.sum(mask_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().numpy()
    return cosine_similarity(mean_pooled)


def get_centroid(bert_model, bert_tokenizer, generations):
    out = BERT_simmilarity_score(bert_model, bert_tokenizer, generations)
    centroid_idx = np.argmax(np.sum(out, axis=1))
    return generations[centroid_idx]


def compare_with_gt(bert_model, bert_tokenizer, generations):
    out = BERT_simmilarity_score(bert_model, bert_tokenizer, generations)
    return out[0, 1]

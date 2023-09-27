import argparse
import torch
from tqdm import tqdm
import pickle

from SelfConsistency.constants import SUMMARY_MODEL_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use.")
    parser.add_argument("--results_filename", type=str, default='ours', help="Desired name for the output file.")
    parser.add_argument("--no_generations", type=int, default=10, help="number of generations per sentence.")
    parser.add_argument("--no_seeds", type=int, default=1, help="number of seeds per document.")
    parser.add_argument("--paraphrased_docs_filename", type=str, default=5,
                        help="Path/filename of the paraphrased documents to load and use for summarization.")

    args = parser.parse_args()
    return args


def summarise_using_paraphrased_docs(paraphrased_docs, results_filename, model='pegasus', no_seeds=1, no_generations=10):
    """
    Create different versions (no_seeds x no_generation x no_of_paraphrased_docs) of summarization per document,
    using the summary model.
    Where no_of_paraphrased_docs is the length of the list (paraphrased_docs value).
    Save results to file and return them.

    :param paraphrased_docs:    (dictionary) Keys are doc ids, Values are list of document paraphrased. It is
                                recommended that the first item in the list is the original document.
    :param results_filename:    (str) path to save the results
    :param model:               (str) model to use for summarization
    :param no_seeds:            (int) number of seeds to create the different summarise
    :param no_generations:      (int) number of generation per seed.
    :return:                    (Dict) Keys are doc ids, value is
                                {'outputs': list of different summarization for this doc}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    cls, config = SUMMARY_MODEL_CONFIG[model].values()
    config['device'] = device
    LLM = cls(**config)

    # Summarise on different paraphrased docs:
    results = {}
    for i, (doc_idx, doc_list) in enumerate(tqdm(paraphrased_docs.items())):
        try:
            outputs = []
            for doc in doc_list:
                res = LLM.summarize(document=doc, no_generations=no_generations, no_seeds=no_seeds)
                outputs.extend(res)
            results[doc_idx] = {'outputs': outputs}
            if i % 25 == 0:
                with open(f'{results_filename}.pkl', "wb") as pickle_file:
                    pickle.dump(results, pickle_file)
        except Exception as e:
            print(f"On doc {doc_idx}, exception: {e}")

    with open(f'{results_filename}.pkl', "wb") as pickle_file:
        pickle.dump(results, pickle_file)

    return results


if __name__ == '__main__':
    args = parse_args()

    paraphrased_docs_filename = args.paraphrased_docs_filename
    # Load paraphrased docs:
    with open(f'{paraphrased_docs_filename}.pkl', 'rb') as file:
        paraphrased_docs_ = pickle.load(file)

    results_ = summarise_using_paraphrased_docs(paraphrased_docs_, args.results_filename, model=args.model,
                                                no_seeds=args.args, no_generations=args.no_generations)

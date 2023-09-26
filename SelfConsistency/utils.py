import pickle


def save_results(res, filename):
    try:
        with open(f'results/{filename}.pkl', "wb") as pickle_file:
            pickle.dump(res, pickle_file)
    except Exception as e:
        raise type(e)(f'failed to save_results, due to: {e}')


def load_results(filename):
    try:
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        return data
    except Exception as e:
        raise type(e)(f'failed to load_results, due to: {e}')


def split_dataset(dataset_size, num_chunks):
    if num_chunks <= 0:
        raise ValueError("Number of chunks must be greater than 0.")

    chunk_size = dataset_size // num_chunks
    remainder = dataset_size % num_chunks

    chunks = []
    start = 0

    for i in range(num_chunks):
        end = start + chunk_size
        if i < remainder:
            end += 1  # Distribute the remainder elements evenly

        chunks.append((start, end))
        start = end

    return chunks

import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_prefix", type=str, help="File prefix to merge.")
    parser.add_argument("--number_of_files", type=int, default=0, help="Number of files to merge.")
    parser.add_argument("--output_path", type=str, default='',
                        help="Path different from the default location for the output file.")
    args = parser.parse_args()
    return args


def merge_pickels(file_prefix, number_of_files, output_path):
    try:
        res = {}
        for i in range(number_of_files):
            with open(f'distributed/{file_prefix}_{i}.pkl', 'rb') as f:
                res.update(pickle.load(f))
        with open(f'{output_path}{file_prefix}.pkl', 'wb') as f:
            pickle.dump(res, f)
        print(f'successfully merged {number_of_files} files into {file_prefix}.pkl')
    except Exception as e:
        raise type(e)(f'failed to merge_pickels, due to: {e}')


if __name__ == '__main__':
    args = parse_args()
    merge_pickels(args.file_prefix, args.number_of_files, args.output_path)

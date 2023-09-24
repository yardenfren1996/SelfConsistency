import argparse
import subprocess

import datasets

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The model to use.")
    parser.add_argument("--dataset", type=str, default='xsum', help="The dataset to use.")
    parser.add_argument("--number_of_jobs", type=int, default=1, help="Number of jobs to submit.")
    parser.add_argument("--job_name", type=str, default="SC_inf", help="Job name for the sbatch script.")
    parser.add_argument("--results_filename", type=str, default='inference_res',
                        help="Desired name for the output inference file.")
    parser.add_argument("--no_generations", type=int, default=10, help="number of generations per sentence.")
    parser.add_argument("--no_seeds", type=int, default=5, help="number of seeds per document.")
    parser.add_argument("--method_type", type=str, default='self_consistency',
                        help="can be one of self_consistency or paraphrasing")
    parser.add_argument("--partition", type=str, default='studentkillable',
                        help="can be one of studentkillable or studentbatch")

    args = parser.parse_args()
    return args


# TODO: Add whether I run self_consistency or paraphrasing !! in the args !

def main():
    try:
        args = parse_args()
        num_gpus = 8 if args.model == 'flan_t5' else 1
        dataset = datasets.load_dataset(args.dataset, split='test', cache_dir='./data')
        chunk_ranges = utils.split_dataset(dataset_size=len(dataset), num_chunks=args.number_of_jobs)
        # Loop through the job parameters and submit sbatch commands
        for batch_num, (start_idx, end_idx) in enumerate(chunk_ranges):
            # if batch_num not in [2, 4, 6]:
            #     continue
            if args.method_type == 'self_consistency':
                ex_line = 'python self_consistency.py --model {args.model} --dataset {args.dataset} --from_index {start_idx} --to_index {end_idx} --results_filename "distributed/{args.results_filename}_{batch_num}" --no_generations {args.no_generations} --no_seeds {args.no_seeds}'
            elif args.method_type == 'paraphrasing':
                ex_line = 'python paraphrasing.py --model {args.model} --dataset {args.dataset} --from_index {start_idx} --to_index {end_idx} --results_filename "distributed/{args.results_filename}_{batch_num}" --no_generations {args.no_generations}'
            else:
                raise ValueError("Method type is not supported, can be one of the following: self consistency / paraphrasing.")
            sbatch_script = f"""#! /bin/sh
#SBATCH --job-name={args.job_name}_{batch_num}
#SBATCH --output=logs/{args.job_name}_{batch_num}.out
#SBATCH --error=logs/{args.job_name}_{batch_num}.err
#SBATCH --partition={args.partition}

{ex_line}        
"""
            with open('evaluate.slurm', 'w') as f:
                f.write(sbatch_script)

            subprocess.run(
                ["sbatch", "--time", "1400", "--nodes", "1", "--ntasks", "1", "--mem", "12000", "--cpus-per-task", "4",
                 "--gpus",
                 f"{num_gpus}", "./evaluate.slurm"])

        print("All jobs submitted successfully!")

    except Exception as e:
        raise type(e)(f'failed to main, due to: {e}')


if __name__ == '__main__':
    main()

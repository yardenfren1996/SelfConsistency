import argparse
import subprocess
import datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xsum', help="The dataset to use.")
    parser.add_argument("--number_of_jobs", type=int, default=1, help="Number of jobs to submit.")
    parser.add_argument("--job_name", type=str, default="SC_inf", help="Job name for the sbatch script.")
    parser.add_argument("--results_filename", type=str, default='inference_res',
                        help="Desired name for the output inference file.")
    args = parser.parse_args()
    return args


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


def main():
    try:
        args = parse_args()
        dataset = datasets.load_dataset(args.dataset, split='test', cache_dir='./data')
        chunk_ranges = split_dataset(dataset_size=len(dataset), num_chunks=args.number_of_jobs)
        # Loop through the job parameters and submit sbatch commands
        for batch_num, (start_idx, end_idx) in enumerate(chunk_ranges):
            sbatch_script = f"""#! /bin/sh
#SBATCH --job-name={args.job_name}_{batch_num}
#SBATCH --output=logs/{args.job_name}_{batch_num}.out
#SBATCH --error=logs/{args.job_name}_{batch_num}.err
#SBATCH --partition=studentkillable
        
python self_consistency.py --dataset {args.dataset} --from_index {start_idx} --to_index {end_idx} --results_filename {args.results_filename}_{batch_num}
"""
            with open('evaluate.slurm', 'w') as f:
                f.write(sbatch_script)

            subprocess.run(
                ["sbatch", "--time", "1000", "--nodes", "1", "--ntasks", "1", "--mem", "12000", "--cpus-per-task", "4",
                 "--gpus",
                 "1", "./evaluate.slurm"])

        print("All jobs submitted successfully!")

    except Exception as e:
        raise type(e)(f'failed to main, due to: {e}')


if __name__ == '__main__':
    main()

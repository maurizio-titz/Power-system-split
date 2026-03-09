from matplotlib import pyplot as plt
import numpy as np
import tqdm
import sys
import pickle
import gzip
from multiprocessing import Pool
import functools


sys.path.append("./")

from utils.config import path_to_cascade_results_sclopf
from utils.data_handling import get_co2_levels
import os
import pickle
import gzip

from scripts.run_cascade_code import collect_batch_results

co2l = 0.6
jureca_files_path = "./jureca_res/"
batch_files_path = jureca_files_path + "batched"
new_res_path = jureca_files_path + "/full/"


def compare_batched_to_old_results(batched_dir, old_res_dir, co2l):
    """this is a check function to compare the results of the new batched cascade code to the old full cascade code, to make sure we get the same results. It compares the number of cascades found at each timestamp, and prints the difference."""

    print(f"Comparing results for CO2 level {co2l}")
    casc_number_diffs = []
    casc_numbers_old = []
    casc_numbers_batched = []

    old_results_file_path = old_res_dir + f"system_splits_Co2L{co2l}_n600.pklz"
    with gzip.open(old_results_file_path, "rb") as f:
        old_res = pickle.load(f)

    batched_file_path = batched_dir + f"system_splits_Co2L{co2l}_n600.pklz"

    with gzip.open(batched_file_path, "rb") as f:
        batched_results = pickle.load(f)

    # extract the highest volt failures from full results
    for timestamp in tqdm.tqdm(batched_results.keys(), leave=False):
        batched_cascade_at_timestamp = batched_results[timestamp]
        old_results_at_timestamp = old_res[timestamp]
        batched_cascade_at_timestamp_ = {
            (k[0][0], k[1][0]): {} for k, v in batched_cascade_at_timestamp.items()
        }
        for k, v in batched_cascade_at_timestamp.items():
            batched_cascade_at_timestamp_[(k[0][0], k[1][0])][(k[0][1], k[1][1])] = v
        for fail_idxs, d in batched_cascade_at_timestamp_.items():
            if len(d) == 1:
                batched_cascade_at_timestamp_[fail_idxs] = list(d.values())[0]
            else:
                num_par_tuples = sorted(list(d.keys()))
                batched_cascade_at_timestamp_[fail_idxs] = (
                    batched_cascade_at_timestamp_[fail_idxs][num_par_tuples[-1]]
                )

        keys_to_remove = []
        for k, v in batched_cascade_at_timestamp_.items():
            if not v[1]:
                keys_to_remove.append(k)
                raise ValueError("non splitting cascade!")
            else:
                batched_cascade_at_timestamp_[k] = v[0]

        for k in keys_to_remove:
            batched_cascade_at_timestamp_.pop(k)

        casc_number_diff = len(old_results_at_timestamp.keys()) - len(
            batched_cascade_at_timestamp_.keys()
        )
        casc_numbers_old.append(len(old_results_at_timestamp.keys()))
        casc_numbers_batched.append(len(batched_cascade_at_timestamp_.keys()))

        print(f"Timestamp {timestamp}: Cascade number difference: {casc_number_diff}")
        casc_number_diffs.append(casc_number_diff)

    return casc_numbers_old, casc_numbers_batched


def process_co2_level(co2l, batch_files_path, new_res_path):
    """Process a single CO2 level - wrapper function for parallel execution"""
    print(f"Processing CO2 level: {co2l}")
    collect_batch_results(
        batch_files_path,
        co2l=co2l,
        n_nodes=600,
        exclude_str=["collected", " ", "check"],
        save_dir=new_res_path,
        all_cascades=False,
    )
    print(f"Completed CO2 level: {co2l}")


if __name__ == "__main__":
    co2_levels = get_co2_levels(n_nodes=600)

    # Create a partial function with fixed arguments
    process_func = functools.partial(
        process_co2_level, batch_files_path=batch_files_path, new_res_path=new_res_path
    )

    # Use multiprocessing to run in parallel
    with Pool() as pool:
        pool.map(process_func, co2_levels)

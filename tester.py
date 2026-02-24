"""
File for random testing
"""

import json
import pyft
import pysam
import torch
import pyBigWig

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *

#--------------------------------------------------------------------------------------------------

def process_stats(stats_raw, save_file_name):

    stats_n = {}
    for key in stats_raw:
        stats_n[f"{key}_mean"] = np.mean(np.array(stats_raw[key]))
        stats_n[f"{key}_std"] = np.std(np.array(stats_raw[key]))

    with open(save_file_name, "w") as f:
        json.dump(stats_n, f, indent=4)

# Load your data here (replace with your actual lists/arrays)
# df = pd.DataFrame({'val': num_m6a_list, 'cell': 'GM12878'}) ...

def plot_fiber_stats(df):
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparative Fiber-seq Statistics: GM12878 vs K562', fontsize=18)

    stats_to_plot = [
        ('m6a', 'm6A Counts per Fiber'),
        ('cpg', 'CpG Counts per Fiber'),
        ('nuc', 'Nucleosome Counts per Fiber'),
        ('msp', 'MSP Counts per Fiber')
    ]

    for i, (col, title) in enumerate(stats_to_plot):
        ax = axes.flat[i]
        sns.histplot(data=df, x=col, hue='CellType', kde=True,
                     element="step", common_norm=False, ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('fiber_stats_distributions_chr_19.png')

# Example usage:
# plot_fiber_stats(your_combined_dataframe)

def wtd(cram, bam, chrom, save_file_name):

    stats = {
        "m6a" : [],
        "cpg" : [],
        "msp" : [],
        "nuc" : [],
        "total" : 0
    }

    step_size = 500000

    total_fibers = 0
    start = 0
    while True:
        end = start + step_size
        chunk_fibers = 0

        with suppress_stdout_stderr():
            for fiber in cram.fetch(chrom, start, end):
                stats["m6a"].append(len(fiber.m6a.starts))
                stats["cpg"].append(len(fiber.cpg.starts))
                stats["msp"].append(len(fiber.msp.starts))
                stats["nuc"].append(len(fiber.nuc.starts))
                chunk_fibers += 1
        if chunk_fibers == 0 and end > bam.get_reference_length(chrom):
            break
        total_fibers += chunk_fibers
        start = end

    stats["total"] = total_fibers

    with open(save_file_name, "w") as f:
        json.dump(stats, f, indent=4)

    return total_fibers, stats

def smt():

    GM_fiber_bam = pyft.Fiberbam("/home/azr/projects/def-maxwl/azr/data/727/complete/GM12878/GM12878-fire-v0.1-filtered.cram")
    GM_pysam = pysam.AlignmentFile("/home/azr/projects/def-maxwl/azr/data/727/complete/GM12878/GM12878-fire-v0.1-filtered.cram")
    K5_fiber_bam = pyft.Fiberbam("/home/azr/projects/def-maxwl/azr/data/727/complete/K562/K562-fire-v0.1-filtered.cram")
    K5_pysam = pysam.AlignmentFile("/home/azr/projects/def-maxwl/azr/data/727/complete/K562/K562-fire-v0.1-filtered.cram")

    chrom = "chr19"

    gm_total, gm_stats = wtd(GM_fiber_bam, GM_pysam, chrom, "GM_stats_raw_chr_19.json")
    k5_total, k5_stats = wtd(K5_fiber_bam, K5_pysam, chrom, "K5_stats_raw_chr_19.json")

    gm_stats = json.load(open("GM_stats_raw_chr_19.json"))
    k5_stats = json.load(open("K5_stats_raw_chr_19.json"))

    process_stats(gm_stats, "GM_stats_pro_chr_19.json")
    process_stats(k5_stats, "K5_stats_pro_chr_19.json")

    # gm_stats = {
    #     'num_m6a_per_fiber':gm_stats_l["m6a"],
    #     'num_cpg_per_fiber':gm_stats_l["cpg"],
    #     'num_nucleosomes_per_fiber':gm_stats_l["nuc"],
    #     'num_msps_per_fiber':gm_stats_l["msp"],
    # }

    # 1. Convert dictionaries to DataFrames
    df_gm = pd.DataFrame(gm_stats)
    df_k562 = pd.DataFrame(k5_stats)

    # 2. Add a column to identify the cell type
    df_gm['CellType'] = 'GM12878'
    df_k562['CellType'] = 'K562'

    # 3. Concatenate (stack) them into one DataFrame
    combined_df = pd.concat([df_gm, df_k562], ignore_index=True)

    # Now you can pass combined_df to the plotting function
    plot_fiber_stats(combined_df)

    # for chr_n in range(15,23):

    #     step_size = 5000000

    #     n_GM = 0
    #     n_GM_p = 1
    #     i = 0
    #     while n_GM_p:
    #         with suppress_stdout_stderr():
    #             n_GM_p = len(list(GM_fiber_bam.fetch(f"chr{chr_n}", i, i+step_size)))
    #         i += step_size
    #         n_GM += n_GM_p

    #     n_K5 = 0
    #     n_K5_p = 1
    #     i = 0
    #     while n_K5_p:
    #         with suppress_stdout_stderr():
    #             n_K5_p = len(list(K5_fiber_bam.fetch(f"chr{chr_n}", i, i+step_size)))
    #         i += step_size
    #         n_K5 += n_K5_p

    #     print(f"In chr{chr_n}, found {n_GM} in GM, {n_K5} in K5")

    print("All Done smt")

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    smt()
    print("All Done")

if __name__=="__main__":
    tester()

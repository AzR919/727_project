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

def get_raw_stats(cram, chrom, start, end, save_file_name, step_size=500000):

    if os.path.exists(save_file_name):
        return json.load(save_file_name)

    stats = {
        "m6a" : [],
        "cpg" : [],
        "msp" : [],
        "nuc" : [],
        "total" : 0
    }

    total_fibers = 0
    start_step = start

    while start_step < end:
        end_step = min(start_step + step_size, end)

        with suppress_stdout_stderr():
            for fiber in cram.fetch(chrom, start_step, end_step):
                if start_step <= fiber.start < end_step:
                    stats["m6a"].append(len(fiber.m6a.starts))
                    stats["cpg"].append(len(fiber.cpg.starts))
                    stats["msp"].append(len(fiber.msp.starts))
                    stats["nuc"].append(len(fiber.nuc.starts))
                    total_fibers += 1

        start_step = end_step

    stats["total"] = total_fibers

    with open(save_file_name, "w") as f:
        json.dump(stats, f, indent=4)
    return stats

def process_stats(stats_raw, save_file_name):

    stats_n = {}
    for key in stats_raw:
        stats_n[f"{key}_mean"] = np.mean(np.array(stats_raw[key]))
        stats_n[f"{key}_std"] = np.std(np.array(stats_raw[key]))

    with open(save_file_name, "w") as f:
        json.dump(stats_n, f, indent=4)

    return stats_n

def plot_fiber_stats(gm_stats, k5_stats, save_file_name):

    # 1. Convert dictionaries to DataFrames
    df_gm = pd.DataFrame(gm_stats)
    df_k562 = pd.DataFrame(k5_stats)

    # 2. Add a column to identify the cell type
    df_gm['CellType'] = 'GM12878'
    df_k562['CellType'] = 'K562'

    # 3. Concatenate (stack) them into one DataFrame
    combined_df = pd.concat([df_gm, df_k562], ignore_index=True)

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
        sns.histplot(data=combined_df, x=col, hue='CellType', kde=True,
                     element="step", common_norm=False, ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_file_name)

def get_coverage_range(crams, chrom, start, end, ccre_path, save_file_name, context_length=2048):

    # Load BED file (columns: chrom, start, end, name, score, strand, type...)
    df = pd.read_csv(ccre_path, sep='\t', header=None, usecols=[0, 1, 2])
    df.columns = ['chrom', 'start', 'end']
    # # Filter for chromosomes present in your chr_sizes to avoid errors
    # ccre_list = df[df['chrom'].isin([chrom])].values

    ccre_list = df[
        (df['chrom'] == chrom) &
        (df['start'] >= start) &
        (df['end'] <= end)
    ].values

    coverage_list = [[],[]]

    for ccre in ccre_list:

        total_fibers = [0,0]
        # 1. Pick a random cCRE
        ccre_start, ccre_end = ccre[1], ccre[2]

        # 2. Calculate the "true" center of the cCRE
        true_center = (ccre_start + ccre_end) // 2

        focal_point = true_center

        # 4. Create the window around the focal point
        half_window = context_length // 2
        true_start = focal_point - half_window
        true_end = true_start + context_length

        with suppress_stdout_stderr():
            for cram_i, cram in enumerate(crams):
                for fiber in cram.fetch(chrom, true_start, true_end):
                    total_fibers[cram_i] += 1

        coverage_list[0].append(total_fibers[0])
        coverage_list[1].append(total_fibers[1])

    coverage_avg = [sum(coverage_list_i)/len(coverage_list_i) for coverage_list_i in coverage_list]
    coverage_min = min([a+b for a, b in zip(coverage_list[0], coverage_list[1])])

    print(save_file_name, len(ccre_list), coverage_avg, coverage_min)

    return

def milestone():

    save_dir = "./milestone"

    GM_fiber_bam = pyft.Fiberbam("/home/azr/projects/def-maxwl/azr/data/727/complete/GM12878/GM12878-fire-v0.1-filtered.cram")
    GM_pysam = pysam.AlignmentFile("/home/azr/projects/def-maxwl/azr/data/727/complete/GM12878/GM12878-fire-v0.1-filtered.cram")
    K5_fiber_bam = pyft.Fiberbam("/home/azr/projects/def-maxwl/azr/data/727/complete/K562/K562-fire-v0.1-filtered.cram")
    K5_pysam = pysam.AlignmentFile("/home/azr/projects/def-maxwl/azr/data/727/complete/K562/K562-fire-v0.1-filtered.cram")
    ccre_path = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/gm12878_ccres.bed"

    chrom = "chr21"
    chrom_len = GM_pysam.get_reference_length(chrom)

    datasets = ["test", "val", "train", "full"]
    bps = [[int (0.9*chrom_len), chrom_len], [int (0.8*chrom_len), int (0.9*chrom_len)], [0,int (0.8*chrom_len)], [0,chrom_len]]

    # for idx, dataset in enumerate(datasets):
    #     start = bps[idx][0]
    #     end = bps[idx][1]

    #     gm_raw_stats = get_raw_stats(GM_fiber_bam, chrom, start, end, f"{save_dir}/GM_stats_raw_{dataset}_{chrom}.json")
    #     k5_raw_stats = get_raw_stats(K5_fiber_bam, chrom, start, end, f"{save_dir}/K5_stats_raw_{dataset}_{chrom}.json")

    #     process_stats(gm_raw_stats, f"{save_dir}/GM_stats_pro_{dataset}_{chrom}.json")
    #     process_stats(k5_raw_stats, f"{save_dir}/K5_stats_pro_{dataset}_{chrom}.json")

    #     plot_fiber_stats(gm_raw_stats, k5_raw_stats, f"{save_dir}/fiber_stats_distributions_{dataset}_{chrom}.png")

    for idx, dataset in enumerate(datasets[:3]):
        start = bps[idx][0]
        end = bps[idx][1]
        save_file_name = dataset
        get_coverage_range([GM_fiber_bam, K5_fiber_bam], chrom, start, end, ccre_path, save_file_name)

#--------------------------------------------------------------------------------------------------

def process_stats_old(stats_raw, save_file_name):

    stats_n = {}
    for key in stats_raw:
        stats_n[f"{key}_mean"] = np.mean(np.array(stats_raw[key]))
        stats_n[f"{key}_std"] = np.std(np.array(stats_raw[key]))

    with open(save_file_name, "w") as f:
        json.dump(stats_n, f, indent=4)

def plot_fiber_stats_old(df):
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

    process_stats_old(gm_stats, "GM_stats_pro_chr_19.json")
    process_stats_old(k5_stats, "K5_stats_pro_chr_19.json")

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
    plot_fiber_stats_old(combined_df)

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
    milestone()
    print("All Done")

if __name__=="__main__":
    tester()

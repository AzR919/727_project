"""
Data processing file
"""

import os
import pyft
import pysam
import torch
import random
import pyBigWig
import numpy as np
import pandas as pd

# from w_redirect import stdout_redirected
from torch.utils.data import IterableDataset

from utils import *

class fiber_data_iterator(IterableDataset):

    def __init__(self, bam_list,
                 fibers_per_entry, context_length,
                 iters_per_epoch, fasta_path,
                 mode, input_flags,
                 ccre_path, chr_sizes_file=None):

        self.fiber_bam_list = [pyft.Fiberbam(fiber_data_path) for fiber_data_path in bam_list]

        self.fibers_per_entry = fibers_per_entry
        self.context_length = context_length
        self.iters_per_epoch = iters_per_epoch
        self.mode = mode

        self.load_fasta(fasta_path)
        self.load_genomic_coords(bam_list[0], mode=mode)

        self.load_ccres(ccre_path)

        self.num_iters = self.iters_per_epoch if self.mode=="train" else len(self.ccre_list)

        self.input_flags = input_flags
        # Map bit positions to functions
        feature_map = [
            self.get_m6a,
            self.get_cpg,
            self.get_msp,
            self.get_nuc,
            self.get_fire_msp,
        ]
        # Pre-determine which functions to run once at startup
        self.input_features = [
            feature_map[i] for i in range(5) if self.input_flags[i]
        ]

    def load_fasta(self, fasta_path):

        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")
        if not os.path.exists(fasta_path + ".fai"):
            pysam.faidx(fasta_path)               # build index if needed
        self.fasta = pysam.FastaFile(fasta_path)

    def onehot_for_locus(self, locus):
        """
        Helper to fetch DNA and convert to one-hot for a given locus [chrom, start, end].
        Returns a tensor [context_length, 4].

        """
        def get_DNA_sequence(chrom, start, end):
            """
            Retrieve the sequence for a given chromosome and coordinate range from a fasta file.

            """
            # Ensure coordinates are within the valid range
            if start < 0 or end <= start:
                raise ValueError("Invalid start or end position")

            return self.fasta.fetch(chrom, start, end)

        def dna_to_onehot(sequence):
            # Create a mapping from nucleotide to index
            mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}

            # Convert the sequence to indices
            indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)

            # Create one-hot encoding
            one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

            # Remove the fifth column which corresponds to 'N'
            one_hot = one_hot[:, :4]

            return one_hot.to(torch.float32)

        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        seq = get_DNA_sequence(chrom, start, end)
        return dna_to_onehot(seq).permute(1,0)

    def load_genomic_coords(self, bam_file_path, mode="train"):
        start = 0 if mode=="train" else 0.8 if mode=="val" else 0.9
        end = 0.8 if mode=="train" else 0.9 if mode=="val" else 1

        samfile = pysam.AlignmentFile(bam_file_path, "rb")
        possible_chr_sizes = dict(zip(samfile.references, samfile.lengths))
        # main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        main_chrs = ["chr20", "chr21"]
        # main_chrs = ["chr20"] if mode=="train" else ["chr21"]

        self.chr_sizes = {k: (int(start*possible_chr_sizes[k]), int(end*possible_chr_sizes[k])) for k in main_chrs if k in possible_chr_sizes}
        # self.chr_sizes = {k: (0, possible_chr_sizes[k]) for k in main_chrs if k in possible_chr_sizes}

    def load_ccres(self, bed_path):
        # Load BED file (columns: chrom, start, end, name, score, strand, type...)
        df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2])
        df.columns = ['chrom', 'start', 'end']

        # 1. Initialize an empty list to store valid chunks
        filtered_chunks = []

        # 2. Iterate through your chromosome dictionary
        for chrom, (locus_start, locus_end) in self.chr_sizes.items():
            # Filter for the specific chromosome AND the coordinate range
            mask = (
                (df['chrom'] == chrom) &
                (df['start'] >= locus_start) &
                (df['end'] <= locus_end)
            )

            filtered_chunks.append(df[mask])

        # 3. Combine them back into one DataFrame and cast to ndarray
        self.ccre_list = pd.concat(filtered_chunks, ignore_index=True).values

    def gen_loci(self):

        raise NotImplementedError

        random_chr = random.choice(list(self.chr_sizes.keys()))

        random_start = random.randint(0, self.chr_sizes[random_chr])
        random_end = random_start + self.context_length

        return random_chr, random_start, random_end

    def gen_ccre_loci(self, jitter_range=200, i=None):
        """
        Generates a genomic window centered around a random cCRE with optional jitter.

        @args:
            jitter_range (int): The maximum number of base pairs to shift the center.
                                e.g., 200 means a shift between -200 and +200 bp.
            i (int): retrieve a specific coordinate
        """

        # 1. Pick a random cCRE
        if i is not None:
            ccre_chrom, ccre_start, ccre_end = self.ccre_list[i]
        else:
            ccre_chrom, ccre_start, ccre_end = random.choice(self.ccre_list)

        # 2. Calculate the "true" center of the cCRE
        true_center = (ccre_start + ccre_end) // 2

        # 3. Apply Jitter
        # This shifts the focus point slightly so the cCRE isn't always perfectly centered
        jitter = random.randint(-jitter_range, jitter_range)
        focal_point = true_center + jitter

        # 4. Create the window around the focal point
        half_window = self.context_length // 2
        random_start = focal_point - half_window
        random_end = random_start + self.context_length

        # 5. Boundary Check (Crucial to prevent index errors)
        min_size = self.chr_sizes[ccre_chrom][0]
        max_size = self.chr_sizes[ccre_chrom][1]

        if random_start < min_size:
            random_start = min_size
            random_end = min_size+self.context_length
        elif random_end > max_size:
            random_end = max_size
            random_start = max_size - self.context_length

        return ccre_chrom, int(random_start), int(random_end)

    def get_m6a(self, fiber, start, end, Q_THRESHOLD=200):

        m6a_data = np.zeros((self.context_length), dtype=np.float32)

        # for ref_pos, aq in zip(fiber.m6a.reference_starts, fiber.m6a.ml):
        #     if ref_pos is None: continue
        #     if start <= ref_pos < end and aq >= Q_THRESHOLD:
        #         m6a_data[ref_pos-start:ref_pos-start+len] = 1

        # 1. Convert lists to numpy arrays
        ref_starts = np.array(fiber.m6a.reference_starts, dtype=np.float32)
        qualities = np.array(fiber.m6a.ml, dtype=np.float32)

        # 2. Create a boolean mask for everything that passes the filters
        # - Within the genomic window
        # - Above the quality threshold
        # - Not None (numpy handles this well if converted correctly)
        mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)

        # 3. Extract the passing positions and calculate their relative offsets
        valid_positions = (ref_starts[mask] - start).astype(np.int32)

        # 4. Use "Fancy Indexing" to set all 1s at once
        m6a_data[valid_positions] = 1

        return m6a_data

    def get_cpg(self, fiber, start, end, Q_THRESHOLD=200):

        cpg_data = np.zeros((self.context_length), dtype=np.float32)

        # 1. Convert lists to numpy arrays
        ref_starts = np.array(fiber.cpg.reference_starts, dtype=np.float32)
        qualities = np.array(fiber.cpg.ml, dtype=np.float32)

        # 2. Create a boolean mask for everything that passes the filters
        # - Within the genomic window
        # - Above the quality threshold
        # - Not None (numpy handles this well if converted correctly)
        mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)

        # 3. Extract the passing positions and calculate their relative offsets
        valid_positions = (ref_starts[mask] - start).astype(np.int32)

        # 4. Use "Fancy Indexing" to set all 1s at once
        cpg_data[valid_positions] = 1

        return cpg_data

    def get_msp(self, fiber, start, end, Q_THRESHOLD=0):

        msp_data = np.zeros((self.context_length), dtype=np.float32)

        for ref_pos, len, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
            if ref_pos is None: continue
            if start <= ref_pos < end and aq >= Q_THRESHOLD:
                msp_data[ref_pos-start:ref_pos-start+len] = 1

        return msp_data

    def get_nuc(self, fiber, start, end, Q_THRESHOLD=0):

        nuc_data = np.zeros((self.context_length), dtype=np.float32)

        for ref_pos, len, aq in zip(fiber.nuc.reference_starts, fiber.nuc.reference_lengths, fiber.nuc.qual):
            if ref_pos is None: continue
            if start <= ref_pos < end and aq >= Q_THRESHOLD:
                nuc_data[ref_pos-start:ref_pos-start+len] = 1

        return nuc_data

    def get_fire_msp(self, fiber, start, end, Q_THRESHOLD=200):

        fire_msp_data = np.zeros((self.context_length), dtype=np.float32)

        for ref_pos, len, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
            if ref_pos is None: continue
            if start <= ref_pos < end and aq >= Q_THRESHOLD:
                fire_msp_data[ref_pos-start:ref_pos-start+len] = 1

        return fire_msp_data

    def get_fiber_data(self, chrom, start, end, num_fibers_per_bam):

        AQ_THRESHOLD = 200

        fibers_bam_list = []

        for i, bam in enumerate(self.fiber_bam_list):
            fibers_tensor_per_bam = []
            n_needed = num_fibers_per_bam[i] # How many do we actually want?

            with suppress_stdout_stderr():
                possible_fibers = bam.fetch(chrom, start, end)

            for fiber in possible_fibers:

                single_fiber_data = np.array([func(fiber, start, end) for func in self.input_features])
                fibers_tensor_per_bam.append(single_fiber_data)

            # ONLY fail if we actually needed fibers from this BAM and it was empty
            if n_needed > 0 and len(fibers_tensor_per_bam) == 0:
                return None

            # If it's empty but we didn't need any (n_needed == 0),
            # append an empty 3D array so shapes match for concatenate later
            if len(fibers_tensor_per_bam) == 0:
                num_feats = len(self.input_features)
                seq_len = end - start
                fibers_bam_list.append(np.empty((0, num_feats, seq_len), dtype=np.float32))
            else:
                fibers_bam_list.append(np.array(fibers_tensor_per_bam, dtype=np.float32))

        final_fiber_list = []

        for n_to_pick, data in zip(num_fibers_per_bam, fibers_bam_list):

            if n_to_pick > data.shape[0]: return None

            # 1. Generate n random unique indices from the range [0, N)
            # replace=False ensures you don't pick the same index twice
            indices = np.random.choice(data.shape[0], size=n_to_pick, replace=False)

            # 2. Use fancy indexing to extract the sub-array
            # This will result in a shape of (n_to_pick, C, L)
            final_fiber_list.append(data[indices])

        combined_fiber_data = np.concatenate(final_fiber_list, axis=0)
        shuffled_fiber_data = np.random.permutation(combined_fiber_data)

        return torch.from_numpy(shuffled_fiber_data).permute(1,2,0)

    def gen_random_perc(self):

        # # 1. Generate n random floats
        # raw_weights = [np.random.rand() for _ in self.fiber_bam_list]

        # # 2. Calculate the sum to use for normalization
        # total = sum(raw_weights)

        # # 3. Scale each to 100
        # # We use round() to get clean integers
        # percentages = [(np.round((w / total) * 10))*10 for w in raw_weights]

        # # 4. Correct for rounding errors (ensuring the sum is exactly 100)
        # diff = 100 - sum(percentages)
        # percentages[0] += diff

        possible_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Uniformly pick the percentage for the first cell type
        p1 = random.choice(possible_values)

        return [p1, 100-p1]

    def get_fibers_per_bam(self, percentages):

        total_num_fibers = self.fibers_per_entry
        num_fibers = [round((w / 100) * total_num_fibers) for w in percentages]

        # Correct for rounding errors (ensuring the sum is exactly total_num_fibers)
        diff = total_num_fibers - sum(num_fibers)
        num_fibers[0] += diff

        # calculate true percentages
        true_percentages = [np.round((w / total_num_fibers), decimals=3) for w in num_fibers]
        true_percentages[0] += (1.0 - true_percentages.sum())

        return num_fibers, torch.from_numpy(np.array(true_percentages, dtype=np.float32))

    # def __iter__(self):

    #     jitter_range = 200 if self.mode=="train" else 0

    #     for i in range(self.num_iters):

    #         i = None if self.mode=="train" else i
    #         found_possible_locus = False

    #         while not found_possible_locus:

    #             random_locus = self.gen_ccre_loci(jitter_range=jitter_range, i=i)
    #             random_perc = self.gen_random_perc()
    #             fibers_per_bam, true_percentages = self.get_fibers_per_bam(random_perc)

    #             fiber_tensor = self.get_fiber_data(*random_locus, fibers_per_bam)
    #             if fiber_tensor is None and self.mode!="train": break
    #             if fiber_tensor is None : continue

    #             dna = self.onehot_for_locus(random_locus)
    #             found_possible_locus = True

    #         if fiber_tensor is None: continue

    #         yield fiber_tensor, dna, true_percentages, random_locus

    def __iter__(self):
        jitter_range = 200 if self.mode == "train" else 0

        for i in range(self.num_iters):
            curr_idx = None if self.mode == "train" else i

            # For training, try multiple times to find a good locus.
            # For val/test, try once and move on if it's empty.
            max_retries = 20 if self.mode == "train" else 1

            for _ in range(max_retries):
                random_locus = self.gen_ccre_loci(jitter_range=jitter_range, i=curr_idx)
                random_perc = self.gen_random_perc()
                fibers_per_bam, true_percentages = self.get_fibers_per_bam(random_perc)

                fiber_tensor = self.get_fiber_data(*random_locus, fibers_per_bam)

                if fiber_tensor is not None:
                    # Everything is valid, get DNA and yield
                    dna = self.onehot_for_locus(random_locus)
                    yield fiber_tensor, dna, true_percentages, random_locus
                    break # Success! Exit the retry loop and move to next i

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    fiber_data_path = ["/home/azr/projects/def-maxwl/azr/data/727/complete/GM12878/GM12878-fire-v0.1-filtered.cram", "/home/azr/projects/def-maxwl/azr/data/727/complete/K562/K562-fire-v0.1-filtered.cram"]
    data_iterator = fiber_data_iterator(fiber_data_path,
            fibers_per_entry=50, context_length=2048,
            iters_per_epoch=2, fasta_path="/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa",
            input_flags=[1, 0, 1, 1, 0],
            ccre_path="/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/gm12878_ccres.bed")

    for data in data_iterator:
        pass
    pass

if __name__=="__main__":
    tester()

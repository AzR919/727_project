"""
Main file

"""

import os
import sys

from args import get_args
from data_utils import fiber_data_iterator
from trainer import Trainer
from models import model_selector
from utils import *

#--------------------------------------------------------------------------------------------------
# Main

def main():

    args = get_args()
    set_seed(args.seed)

    save_str = create_save_str(args)
    res_dir = os.path.join(args.res_dir, save_str)

    data_iterator = fiber_data_iterator(args.fiber_data_paths,
            fibers_per_entry=args.fibers_per_entry, context_length=args.context_length,
            iters_per_epoch=args.iters_per_epoch, fasta_path="/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa",
            input_flags=args.input_flags,
            ccre_path="/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/gm12878_ccres.bed")

    model = model_selector(args.model, args)

    trainer = Trainer(model, data_iterator, epochs=args.epochs, batch_size=args.batch_size, run_name=args.name_suffix, config=args)

    trainer.train(save_dir=res_dir)

if __name__=="__main__":
    main()

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
    os.makedirs(res_dir, exist_ok=True)

    # Common parameters for all iterators
    data_kwargs = {
        "bam_list": args.fiber_data_paths,
        "fibers_per_entry": args.fibers_per_entry,
        "context_length": args.context_length,
        "fasta_path": "/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa",
        "input_flags": args.input_flags,
        "ccre_path": "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/gm12878_ccres.bed"
    }

    # 1. Training Iterator
    train_iterator = fiber_data_iterator(**data_kwargs, mode="train", iters_per_epoch=args.iters_per_epoch)

    # 2. Validation Iterator
    val_iterator = fiber_data_iterator(**data_kwargs, mode="val", iters_per_epoch=args.iters_per_epoch)

    # 3. Test Iterator
    test_iterator = fiber_data_iterator(**data_kwargs, mode="test", iters_per_epoch=args.iters_per_epoch)

    # data_iterator = fiber_data_iterator(args.fiber_data_paths,
    #         fibers_per_entry=args.fibers_per_entry, context_length=args.context_length,
    #         iters_per_epoch=args.iters_per_epoch, fasta_path="/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa",
    #         input_flags=args.input_flags,
    #         ccre_path="/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/gm12878_ccres.bed")

    model = model_selector(args.model, args)

    # trainer = Trainer(model, data_iterator, epochs=args.epochs, batch_size=args.batch_size, run_name=args.name_suffix, config=args)
    trainer = Trainer(
        model,
        train_iterator,
        val_dataset=val_iterator,
        test_dataset=test_iterator,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_name=args.name_suffix,
        config=args,
        save_dir=res_dir
    )

    trainer.train()

if __name__=="__main__":
    main()

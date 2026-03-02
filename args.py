"""
File for argparsing
"""

import argparse

def get_args():

    """Argument parser"""
    parser = argparse.ArgumentParser(
        description="Fiber seq test runs",
    )

    # Data
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--fiber_data_paths", type=str, nargs="+",
                           default=["/home/azr"])
    data_group.add_argument("--context_length", type=int,
                           default=2048)
    data_group.add_argument("--fibers_per_entry", type=int,
                             default=32)
    data_group.add_argument("--input_flags", type=int, nargs="+", default=[1,1,1,1,1],
                            help="binary indicators of what to use as input." \
                            "0: m6a, 1: cpg, 2: msp, 3: nuc, 4: fire_msp")
    data_group.add_argument("--num_input_features", type=int, default=0,
                            help="computed at runtime")
    data_group.add_argument("--cat_dna", action="store_true",
                            help="concatenate DNA as input")

    # Model
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--model", type=str,
                             default="base")
    model_group.add_argument("--d_model", type=int,
                           default=64)

    # Train
    trainer_group = parser.add_argument_group("Model Architecture")
    trainer_group.add_argument("--epochs", type=int,
                             default=2)
    trainer_group.add_argument("--iters_per_epoch", type=int,
                             default=1024)
    trainer_group.add_argument("--batch_size", type=int,
                             default=32)


    # I/O
    io_group = parser.add_argument_group("Model I/O")
    io_group.add_argument("--res_dir", type=str,
                          default="./results",
                          help="Directory to save trained models")
    io_group.add_argument("--name_suffix", type=str, default="",
                         help="Suffix to append to auto-generated model name")

    # misc
    misc_group = parser.add_argument_group("miscellaneous arguments")
    misc_group.add_argument("--debug", "-D", action="store_true",
                            help="Enable debug mode with extra logging")
    misc_group.add_argument("--seed", type=int, default=919)

    parsed_args = parser.parse_args()
    parsed_args.num_input_features = sum(parsed_args.input_flags)

    return parsed_args


#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()

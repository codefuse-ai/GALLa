import os
from transformers import set_seed
from arguments import prepare_args
from data.filter import filter

# get args
os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = prepare_args()


def main():

    set_seed(args.seed)

    filter(args)


if __name__ == "__main__":
    main()
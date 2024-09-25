
import os
import argparse

import mlflow

from assets.tokenizer import CharacterTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alphabet', required=True)
    parser.add_argument('--max_length', type=int, required=True)
    parser.add_argument('--default_char', type=str, default=' ')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--experiment_name')

    return parser.parse_args()


def main():

    args = parse_args()

    experiment_name = args.experiment_name
    del args.experiment_name
    experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="create_tokenizer"
        ):

        mlflow.log_params(args.__dict__)

        tokenizer = CharacterTokenizer(
            alphabet = args.alphabet,
            max_length = args.max_length,
            default_char = args.default_char,
        )

        tokenizer.save_config(args.save_dir)


if __name__ == "__main__":
    main()


import os
import ast
import argparse
import mlflow

from assets.unet_2d_condition import  UNet2DCharConditionModel
from assets.utils import TupleAction

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument('--down_block_types', action=TupleAction, required=True)
    parser.add_argument('--up_block_types', action=TupleAction, required=True)
    parser.add_argument('--block_out_channels', action=TupleAction)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--layers_per_block', type=int, default=2)
    parser.add_argument('--encoder_hid_dim', type=int)
    parser.add_argument('--char_embed_dim', type=int)
    parser.add_argument('--num_unique_chars', type=int, default=None)
    parser.add_argument('--num_class_embeds', type=int)
    parser.add_argument('--attention_head_dim', type=int, default=8)
    parser.add_argument('--cross_attention_dim', type=int, default=128)
    parser.add_argument('--addition_embed_type', default=None)
    parser.add_argument("--time_embedding_type", default="positional")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--experiment_name")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    experiment_name = args.experiment_name
    del args.experiment_name
    experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="create_model"
        ):

        mlflow.log_params(args.__dict__)

        save_dir = args.save_dir
        del args.save_dir

        model = UNet2DCharConditionModel(
            **args.__dict__
        )

        model.save_pretrained(save_dir, safe_serialization=False)



if __name__ == "__main__":
    main()



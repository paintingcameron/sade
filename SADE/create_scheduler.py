
import os
import argparse

import mlflow

from diffusers.schedulers import DDPMScheduler, DDIMScheduler

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--scheduler_type', default='ddpm')
    parser.add_argument('--num_training_timesteps', type=int, default=1000)
    parser.add_argument('--variance_type', default='fixed_small')
    parser.add_argument('--beta_schedule', default='linear')
    parser.add_argument('--timestep_spacing', default='leading')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument("--experiment_name")
    parser.add_argument("--prediction_type", default="epsilon", choices=['epsilon', 'v_prediction'])

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    experiment_name = args.experiment_name
    del args.experiment_name
    experiment = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="create_scheduler"
        ):

        mlflow.log_params(args.__dict__)

        if args.scheduler_type.lower() == "ddpm":
            scheduler = DDPMScheduler(
                num_train_timesteps=args.num_training_timesteps,
                variance_type=args.variance_type,
                beta_schedule=args.beta_schedule,
                timestep_spacing=args.timestep_spacing,
                prediction_type=args.prediction_type,
            )
        elif args.scheduler_type.lower() == "ddim":
            scheduler = DDIMScheduler(
                num_train_timesteps=args.num_training_timesteps,
                beta_schedule=args.beta_schedule,
                timestep_spacing=args.timestep_spacing,
                prediction_type=args.prediction_type,
            )

        else:
            raise NotImplementedError(f"Unknown scheudler: {args.scheduler_type}")

        scheduler.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
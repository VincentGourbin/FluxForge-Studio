"""
Standalone training script for LoRA adapters using DreamBooth.
This script provides a command-line interface for training custom LoRA models.
"""

from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.error.exceptions import StopTrainingException
from mflux.ui.cli.parsers import CommandLineParser


def main():
    """
    Main function for standalone LoRA training.
    Parses command line arguments and initiates the DreamBooth training process.
    
    Usage:
        python train.py --train-config path/to/config.json [--train-checkpoint path/to/checkpoint]
    """
    parser = CommandLineParser(description="Finetune a LoRA adapter")
    parser.add_model_arguments(require_model_arg=False)
    parser.add_training_arguments()
    args = parser.parse_args()

    # Initialize the training components from configuration
    flux, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
        config_path=args.train_config,
        checkpoint_path=args.train_checkpoint
    )  # fmt: off

    try:
        # Start the DreamBooth training process
        DreamBooth.train(
            flux=flux,
            runtime_config=runtime_config,
            training_spec=training_spec,
            training_state=training_state
        )  # fmt: off
    except StopTrainingException as stop_exc:
        # Save training state if training is stopped prematurely
        training_state.save(training_spec)
        print(stop_exc)


if __name__ == "__main__":
    main()

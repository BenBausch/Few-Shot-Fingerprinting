import argparse
import logging
import os
import random

import numpy as np
import torch

from inference.Embedder import Embedder
from inference.Visualizer import Visualizer
from Tester import Tester
from Trainer import Trainer
from utils.utils import get_dataloaders, setup

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    config_file = args.config

    # set random seeds
    seed = 42
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    (
        config,
        checkpoint,
        model,
        optimizer,
        scheduler,
        loss_function,
        batch_size,
        number_of_epochs,
        device,
        result_path,
    ) = setup(config_file)

    logging.info(
        f"Initializing {'Trainer' if config['MODE'] == 'TRAIN' else 'Tester'} with the following parameters:"
    )
    logging.info(f"Model: {config['MODEL']['NAME']}")
    logging.info(f"Optimizer: {config['TRAINING']['OPTIMIZER']['NAME']}")
    logging.info(f"Optimizer settings: {config['TRAINING']['OPTIMIZER']['SETTINGS']}")
    logging.info(f"Loss Function: {config['TRAINING']['LOSS']['NAME']}")
    logging.info(f"Loss Function settings: {config['TRAINING']['LOSS']['SETTINGS']}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Result path: {result_path}")
    logging.info(f"Number of Epochs: {number_of_epochs}")
    logging.info(f"Device: {device}")
    logging.info("----------------------------------------------------------")

    if config["MODE"] == "TRAIN":
        training_loader, validation_loader = get_dataloaders(
            config=config, batch_size=batch_size
        )

        validation_tester = Tester(
            config=config,
            model=model,
            test_loader=validation_loader,
            device=device,
            result_path=config["VALIDATION"]["RESULT_PATH"],
            n_way=config["VALIDATION"]["N_WAY"],
            k_shot=config["VALIDATION"]["K_SHOT"],
            q_queries=config["VALIDATION"]["Q_QUERIES"],
            top_k=config["VALIDATION"]["TOP_K"],
            max_batch_size=100,
            similarity_metric=config["TRAINING"]["LOSS"]["SETTINGS"]["distance_metric"],
            is_val_tester=True,
        )

        trainer = Trainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_function,
            device=device,
            result_path=result_path,
            checkpoint=checkpoint,
            train_loader=training_loader,
            validation_loader=validation_loader,
            num_epochs=number_of_epochs,
            validation_tester=validation_tester,
        )

        model, epoch = trainer.fit()

        config["MODE"] = "TEST"
        test_loader = get_dataloaders(config=config, batch_size=batch_size)
        result_path = config["TESTING"]["RESULT_PATH"]
        result_path = os.path.join(result_path, f"model_checkpoint_{epoch}")

        tester = Tester(
            config=config,
            model=model,
            test_loader=test_loader,
            device=device,
            result_path=config["TESTING"]["RESULT_PATH"],
            n_way=config["TESTING"]["N_WAY"],
            k_shot=config["TESTING"]["K_SHOT"],
            q_queries=config["TESTING"]["Q_QUERIES"],
            top_k=config["TESTING"]["TOP_K"],
            max_batch_size=100,
            similarity_metric=config["TRAINING"]["LOSS"]["SETTINGS"]["distance_metric"],
            wandb_log=True,
        )

        tester.test()

    elif config["MODE"] == "TEST":

        test_loader = get_dataloaders(config=config, batch_size=batch_size)

        tester = Tester(
            config=config,
            model=model,
            test_loader=test_loader,
            device=device,
            result_path=config["TESTING"]["RESULT_PATH"],
            n_way=config["TESTING"]["N_WAY"],
            k_shot=config["TESTING"]["K_SHOT"],
            q_queries=config["TESTING"]["Q_QUERIES"],
            top_k=config["TESTING"]["TOP_K"],
            max_batch_size=1000,
            similarity_metric=config["TRAINING"]["LOSS"]["SETTINGS"]["distance_metric"],
            wandb_log=True,
        )

        tester.test()

    elif config["MODE"] == "EMBEDDER":
        embed_loader = get_dataloaders(config=config, batch_size=batch_size)
        embedder = Embedder(
            config=config,
            model=model,
            data_loader=embed_loader,
            device=device,
            result_path=config["EMBEDDER"]["RESULT_PATH"],
            max_batch_size=200,
        )
        embedder.embed_data()
        embedder.plot_embedded_data(
            pickle_path=config["EMBEDDER"]["RESULT_PATH"],
        )

    elif config["MODE"] == "VISUALIZER":
        loader = get_dataloaders(config=config, batch_size=batch_size)
        visualizer = Visualizer(
            config=config,
            model=model,
            data_loader=loader,
            device=device,
            result_path=config["VISUALIZER"]["RESULT_PATH"],
            max_batch_size=1,
        )
        visualizer.visualize_data()


if __name__ == "__main__":
    main()

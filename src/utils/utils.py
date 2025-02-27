import json
import logging
import os
from datetime import UTC, datetime

import git
import torch

import wandb
from datasets.APDataset import APDataset
from datasets.NwayKshotDataset import NwayKshotDataset
from datasets.SingleFullPatientDataset import SingleFullSubjectDataset
from losses import get_loss
from models import get_model
from samplers.RandomSamplerWithSpecificLength import RandomSamplerWithSpecificLength
from transforms import get_transforms


def setup(config_file):
    with open(config_file, "r") as f:
        config = json.loads(f.read())

    # Save the config file within the run log directory for reproducibility
    os.makedirs(config["LOG_PATH"], exist_ok=True)
    config_path = os.path.join(config["LOG_PATH"], "config.json")
    with open(config_path, "w") as f:
        config["GIT_COMMIT"] = git.Repo(
            search_parent_directories=True
        ).head.object.hexsha
        f.write(json.dumps(config, indent=4))

    setup_wandb(config)
    setup_logger(config)

    if config["MODEL"]["CHECKPOINT"] is not None:
        checkpoint = torch.load(config["MODEL"]["CHECKPOINT"])
    else:
        checkpoint = None
    device = torch.device(device=config["DEVICE"])
    model = get_model(config=config, checkpoint=checkpoint)
    model.to(device=device)

    optimizer = get_optimizer(
        optimizer_name=config["TRAINING"]["OPTIMIZER"]["NAME"],
        checkpoint=checkpoint,
        model_params=model.parameters(),
        **config["TRAINING"]["OPTIMIZER"]["SETTINGS"],
    )

    scheduler_rate = (
        config["TRAINING"]["WARM_RESTART_EVERY_N_SAMPLES"]
        // config["TRAINING"]["BATCH_SIZE"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=scheduler_rate
    )
    loss_function = get_loss(
        loss_name=config["TRAINING"]["LOSS"]["NAME"],
        **config["TRAINING"]["LOSS"]["SETTINGS"],
    )
    batch_size = config["TRAINING"]["BATCH_SIZE"]
    number_of_epochs = config["TRAINING"]["NUM_EPOCHS"]
    result_path = config["SAVE_PATH"]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return (
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
    )


def setup_wandb(config):
    """
    Setup wandb for logging
    """
    wandb.init(
        # set the wandb project where this run will be logged
        entity=config["PROJECT"]["ENTITY"],
        project=config["PROJECT"]["NAME"],
        notes=config["PROJECT"]["NOTES"],
        name=f'{config["TRAINING"]["LOSS"]["NAME"]}_{datetime.now(UTC).strftime("%Y-%m-%d")}',
        tags=config["PROJECT"]["TAGS"],
    )


def setup_logger(config):
    """
    Setup logger for logging
    """
    log_path = config["LOG_PATH"]

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file_name = os.path.join(
        log_path, f'{datetime.now(UTC).strftime("%Y-%m-%d")}.log'
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
    )


def dict_to_device(dictionary: dict, device: str):
    for key, values in dictionary.items():
        if isinstance(values, torch.Tensor):
            dictionary[key] = values.to(device)
        elif isinstance(values, (list, tuple)):
            dictionary[key] = [tensor.to(device) for tensor in values]
        else:
            pass
    return dictionary


def get_optimizer(optimizer_name, checkpoint, model_params, *args, **kwargs):
    """
    Get optimizer based on the optimizer name defined in the config file

    Args:
        optimizer_name (str): Name of the optimizer as defined in the config file

    Returns:
        func: Optimizer function
    """
    if optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(params=model_params, *args, **kwargs)
    if optimizer_name == "ADAMW":
        optimizer = torch.optim.AdamW(params=model_params, *args, **kwargs)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return optimizer


def get_dataloaders(config, batch_size):

    anchor_trans, positive_trans = get_transforms(config)

    if config["MODE"] == "TRAIN":
        training_dataset = APDataset(
            dataset_name=config["DATASET"]["DATASET_NAME"],
            dataset_path=config["DATASET"]["DATASET_PATH"],
            split_file_path=config["DATASET"]["TRAINING_SET"],
            meta_data_path=None,
            file_endings=config["DATASET"]["FILE_TYPE"],
            anchor_modalities=config["DATASET"]["ANCHOR_MODALITIES"],
            positive_modalities=config["DATASET"]["POSITIVE_MODALITIES"],
            anchor_transform=anchor_trans,
            positive_transform=positive_trans,
            num_positives=config["DATASET"]["TRAINING_NUM_POS"],
            min_images_per_modality=config["DATASET"]["NUMBER_OF_IMAGES_PER_PATIENT"],
        )
        train_sampler = RandomSamplerWithSpecificLength(
            training_dataset, config["TRAINING"]["SAMPLES_PER_EPOCH"]
        )
        training_loader = torch.utils.data.DataLoader(
            dataset=training_dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=40,
        )

        validation_dataset = NwayKshotDataset(
            dataset_name=config["DATASET"]["DATASET_NAME"],
            dataset_path=config["DATASET"]["DATASET_PATH"],
            split_file_path=config["DATASET"]["VALIDATION_SET"],
            meta_data_path=None,
            n_way=config["VALIDATION"]["N_WAY"],
            k_shot=config["VALIDATION"]["K_SHOT"],
            k_modalities=config["DATASET"]["K_MODALITIES"],
            q_queries=config["VALIDATION"]["Q_QUERIES"],
            q_modalities=config["DATASET"]["Q_MODALITIES"],
            iterations=config["VALIDATION"]["ITERATIONS"],
            file_endings=config["DATASET"]["FILE_TYPE"],
            min_images_per_modality=config["DATASET"]["NUMBER_OF_IMAGES_PER_PATIENT"],
            support_transform=anchor_trans,
            query_transform=anchor_trans,
        )

        validation_loader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=10,
        )

        return training_loader, validation_loader

    elif config["MODE"] == "TEST":

        support_trans, query_trans = get_transforms(config)

        test_loader = torch.utils.data.DataLoader(
            NwayKshotDataset(
                dataset_name=config["DATASET"]["DATASET_NAME"],
                dataset_path=config["DATASET"]["DATASET_PATH_TEST"],
                split_file_path=config["DATASET"]["TEST_SET"],
                meta_data_path=config["DATASET"]["DATASET_META_PATH"],
                n_way=config["TESTING"]["N_WAY"],
                k_shot=config["TESTING"]["K_SHOT"],
                k_modalities=config["DATASET"]["K_MODALITIES"],
                q_queries=config["TESTING"]["Q_QUERIES"],
                q_modalities=config["DATASET"]["Q_MODALITIES"],
                iterations=config["TESTING"]["ITERATIONS"],
                file_endings=config["DATASET"]["FILE_TYPE"],
                min_images_per_modality=config["DATASET"][
                    "NUMBER_OF_IMAGES_PER_PATIENT"
                ],
                support_transform=support_trans,
                query_transform=query_trans,
            ),
            sampler=None,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=30,
        )

        return test_loader

    elif config["MODE"] == "EMBEDDER" or config["MODE"] == "VISUALIZER":

        transforms, _ = get_transforms(config)

        dataset = SingleFullSubjectDataset(
            dataset_name=config["DATASET"]["DATASET_NAME"],
            dataset_path=config["DATASET"]["DATASET_PATH"],
            split_file_path=config[config["MODE"]]["DATA_SPLIT"],
            meta_data_path=config["DATASET"]["DATASET_META_PATH"],
            file_endings=config["DATASET"]["FILE_TYPE"],
            min_images_per_modality=None,
            transforms=transforms,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=10,
        )

        return data_loader

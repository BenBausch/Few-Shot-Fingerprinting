from transforms.mri_transforms import get_mri_test_transforms, get_mri_train_transforms
from transforms.omniglot_transforms import (
    get_omniglot_test_transforms,
    get_omniglot_train_transforms,
)
from transforms.xray_transforms import (
    get_xray_test_transforms,
    get_xray_train_transforms,
)


def get_transforms(config):
    if config["DATASET"]["TRANSFORMS"] == "OMNIGLOT":
        if config["MODE"] == "TRAIN":
            return get_omniglot_train_transforms(resize=config["DATASET"]["RESIZE"])
        elif (
            (config["MODE"] == "TEST")
            or (config["MODE"] == "EMBEDDER")
            or (config["MODE"] == "VISUALIZER")
        ):
            return get_omniglot_test_transforms(resize=config["DATASET"]["RESIZE"])
    elif config["DATASET"]["TRANSFORMS"] == "CHESTXRAY14":
        if config["MODE"] == "TRAIN":
            return get_xray_train_transforms(resize=config["DATASET"]["RESIZE"])
        elif (
            (config["MODE"] == "TEST")
            or (config["MODE"] == "EMBEDDER")
            or (config["MODE"] == "VISUALIZER")
        ):
            return get_xray_test_transforms(resize=config["DATASET"]["RESIZE"])
    elif config["DATASET"]["TRANSFORMS"] == "BRATS2021":
        if config["MODE"] == "TRAIN":
            return get_mri_train_transforms(resize=config["DATASET"]["RESIZE"])
        elif (
            (config["MODE"] == "TEST")
            or (config["MODE"] == "EMBEDDER")
            or (config["MODE"] == "VISUALIZER")
        ):
            return get_mri_test_transforms(resize=config["DATASET"]["RESIZE"])

    return config["DATASET"]["RESIZE"]

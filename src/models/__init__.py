from typing import List

from torchvision.models.resnet import BasicBlock
from vit_pytorch import ViT
from vit_pytorch.vit_3d import ViT as ViT3D
from monai.networks.nets import ResNet

from models.CustomResNet import CustomResNet


def get_resnet_layers(config: dict) -> List[int]:
    if config["MODEL"]["NAME"] == "ResNet18":
        layers = [2, 2, 2, 2]
    elif config["MODEL"]["NAME"] == "ResNet34":
        layers = [3, 4, 6, 3]
    elif config["MODEL"]["NAME"] == "ResNet50":
        layers = [3, 4, 6, 3]
    elif config["MODEL"]["NAME"] == "ResNet101":
        layers = [3, 4, 23, 3]
    else:
        raise ValueError("Model not implemented.")

    return layers


def get_model(config, checkpoint=None):

    if "ResNet" in config["MODEL"]["NAME"]:
        if config["MODEL"]["SETTINGS"]["spatial_dims"] == 2:
            layers = get_resnet_layers(config)
            model = CustomResNet(
                in_channels=config["MODEL"]["SETTINGS"]["in_channels"],
                spatial_dims=config["MODEL"]["SETTINGS"]["spatial_dims"],
                block=BasicBlock,
                layers=layers,
                num_classes=1000,
                embedding_size=config["MODEL"]["SETTINGS"]["embedding_size"],
            )
        if config["MODEL"]["SETTINGS"]["spatial_dims"] == 3:
            layers = get_resnet_layers(config)
            model = ResNet(
                n_input_channels=config["MODEL"]["SETTINGS"]["in_channels"],
                spatial_dims=config["MODEL"]["SETTINGS"]["spatial_dims"],
                block="basic",
                layers=layers,
                block_inplanes=config["MODEL"]["SETTINGS"]["block_inplanes"],
                num_classes=config["MODEL"]["SETTINGS"]["embedding_size"],
            )
    elif "ViT" in config["MODEL"]["NAME"]:
        if config["MODEL"]["SETTINGS"]["spatial_dims"] == 2:
            model = ViT(
                channels=config["MODEL"]["SETTINGS"]["in_channels"],
                image_size=config["DATASET"]["RESIZE"][0],
                patch_size=config["MODEL"]["SETTINGS"]["patch_size"],
                num_classes=config["MODEL"]["SETTINGS"]["embedding_size"],
                pool="cls",
                depth=config["MODEL"]["SETTINGS"]["num_layers"],
                heads=config["MODEL"]["SETTINGS"]["num_heads"],
                dim=config["MODEL"]["SETTINGS"]["hidden_size"],
                mlp_dim=config["MODEL"]["SETTINGS"]["hidden_size"] * 4,
            )
        if config["MODEL"]["SETTINGS"]["spatial_dims"] == 3:
            model = ViT3D(
                channels=config["MODEL"]["SETTINGS"]["in_channels"],
                image_size=config["DATASET"]["RESIZE"][1],
                image_patch_size=config["MODEL"]["SETTINGS"]["patch_size"],
                frames=config["DATASET"]["RESIZE"][0],
                frame_patch_size=config["MODEL"]["SETTINGS"]["patch_size"],
                num_classes=config["MODEL"]["SETTINGS"]["embedding_size"],
                dim=config["MODEL"]["SETTINGS"]["hidden_size"],
                depth=config["MODEL"]["SETTINGS"]["num_layers"],
                heads=config["MODEL"]["SETTINGS"]["num_heads"],
                mlp_dim=config["MODEL"]["SETTINGS"]["hidden_size"] * 4,
            )

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    return model

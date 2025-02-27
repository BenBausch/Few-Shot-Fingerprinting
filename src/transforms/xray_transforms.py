import torchvision.transforms as transforms

# resize = [512, 512]


def get_xray_train_transforms(resize):
    """
    Get the transformations for the X-ray dataset training splits.
    """
    anchor = transforms.Compose(
        [
            transforms.Resize(
                resize, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomAffine(
                degrees=(-5, 5),
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0,
            ),
        ]
    )

    positive = transforms.Compose(
        [
            transforms.Resize(
                resize, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomAffine(
                degrees=(-5, 5),
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0,
            ),
        ]
    )

    return anchor, positive


def get_xray_test_transforms(resize):
    """
    Get the transformations for the X-ray dataset test splits.
    Only resize and normalize for evaluation.
    """
    anchor = transforms.Compose(
        [
            transforms.Resize(
                resize, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    positive = transforms.Compose(
        [
            transforms.Resize(
                resize, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return anchor, positive

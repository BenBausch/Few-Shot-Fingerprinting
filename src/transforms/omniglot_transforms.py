import torchvision.transforms as transforms

# resize = [128, 128]


def get_omniglot_train_transforms(resize):
    """
    Get the transformations for the Omniglot dataset splits.
    """
    anchor = transforms.Compose(
        [
            transforms.Resize(
                resize, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomAffine(
                degrees=(-1, 1),
                translate=(0.05, 0.05),
                scale=(0.98, 1.02),
                shear=(-1, 1),
                fill=1,
            ),
        ]
    )

    positive = transforms.Compose(
        [
            transforms.Resize(
                resize, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomAffine(
                degrees=(-1, 1),
                translate=(0.05, 0.05),
                scale=(0.98, 1.02),
                shear=(-1, 1),
                fill=1,
            ),
        ]
    )

    return anchor, positive


def get_omniglot_test_transforms(resize):
    """
    Get the transformations for the Omniglot dataset splits.
    """
    support = transforms.Compose(
        [transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR)]
    )

    query = transforms.Compose(
        [transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR)]
    )

    return support, query

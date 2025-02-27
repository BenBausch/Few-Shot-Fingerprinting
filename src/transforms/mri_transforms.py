from monai.transforms import Compose, RandAffine, Resize

# resize is now a constant that is set here, and imported to the models init, to set the size of the image.


def get_mri_train_transforms(resize):

    anchor = Compose(
        [
            Resize(spatial_size=resize),
            RandAffine(
                prob=0.5,
                translate_range=(5, 5, 5),
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ]
    )
    positive = Compose(
        [
            Resize(spatial_size=resize),
            RandAffine(
                prob=0.5,
                translate_range=(5, 5, 5),
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ]
    )

    return anchor, positive


def get_mri_test_transforms(resize):
    anchor = Compose(
        [
            Resize(spatial_size=resize),
            RandAffine(
                prob=0.5,
                translate_range=(5, 5, 5),
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ]
    )
    positive = Compose(
        [
            Resize(spatial_size=resize),
            RandAffine(
                prob=0.5,
                translate_range=(5, 5, 5),
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
            ),
        ]
    )

    return anchor, positive

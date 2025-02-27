from losses.CustomTripletMarginLoss import CustomTripletMarginLoss


def get_loss(loss_name, *args, **kwargs):
    """
    Get loss function based on the loss name defined in the config file

    Args:
        loss_name (str): Name of the loss function as defined in the config file

    Returns:
        func: Loss function
    """
    if loss_name == "TRIPLET_LOSS":
        return CustomTripletMarginLoss(*args, **kwargs)

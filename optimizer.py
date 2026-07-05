from torch import optim
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau)


def create_optimizer(model, config):
    """
    Cria optimizer e scheduler.
    Opções suportadas:
    optimizer:
        - adam
        - adam_cosine
        - sgd
    Retorna:
        optimizer, scheduler
    """

    optimizer_name = config["optimizer"].lower()
    lr = config.get("lr")
    weight_decay = config.get("weight_decay")
    momentum = config.get("momentum")
    epochs = config.get("epochs")

    # ADAM

    if optimizer_name == "adam":

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = None

    # ADAM + COSINE ANNEALING

    elif optimizer_name == "adam_cosine":

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # SGD + MOMENTUM + WEIGHT DECAY

    elif optimizer_name == "sgd":

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        scheduler = None

    # SGD + REDUCE ON PLATEAU 

    elif optimizer_name == "sgd_plateau":

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    else:

        raise ValueError(f"Optimizer '{optimizer_name}' não suportado.")

    return optimizer, scheduler


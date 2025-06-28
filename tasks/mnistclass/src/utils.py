

import random
import numpy as np
import torch
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_wandb(cfg):#creates login thing on its own
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        tags=cfg.wandb.tags,
        config=dict(cfg),
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_optimizer(model, cfg):

    if cfg.optimizer.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=cfg.optimizer.betas,
        )
    elif cfg.optimizer.name == "sgd": #dont need it
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")


def log_model_info(model):
   
    param_count = count_parameters(model)
    print(f"model created with {param_count:,} parameters")
    
    if param_count >= 10000:
        print("WARNING: model has >= 10,000 parameters!")
    
    # Log to wandb
    wandb.log({"model_parameters": param_count})
    
    return param_count
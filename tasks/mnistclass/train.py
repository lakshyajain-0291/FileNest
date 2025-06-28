#main train file

import hydra
from omegaconf import DictConfig
import torch
import wandb

from src.model import create_model
from src.dataset import get_dataloaders
from src.trainer import Trainer
from src.utils import set_seed, init_wandb, log_model_info


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:#cfg MUST be DictConfig type. And func return NONE
    
    print("Starting mnist training...")
    print(f"Config: {cfg}")
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Initialize wandb
    init_wandb(cfg)
    
    # get dataloaders
    train_loader, test_loader = get_dataloaders(cfg)


    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("creating model...")
    model = create_model(cfg)
    param_count = log_model_info(model)
    
    # Verify parameter constraint
    if param_count >= 10000:#double check
        print(f"ERROR: Model has {param_count:,} parameters (>= 10,000)")
        return
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=1e-4,
    )
    
    # Watch model with wandb
    wandb.watch(model, log="all")
    
    # create trainer instance
    trainer = Trainer(model, train_loader, test_loader, optimizer, cfg)
    
    # Start training
    print("Starting training...")
    best_accuracy = trainer.train()
    
    # Log final results
    print(f"\nTraining completed!")

    
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Model parameters: {param_count:,}")
    
    # Final validation
    final_loss, final_acc = trainer.validate()
    wandb.log({
        "final_accuracy": final_acc,
        "final_loss": final_loss,
        "best_accuracy": best_accuracy,
        "parameter_count": param_count,
    })
    
    # Check if target met
    if final_acc >= 99:
        print("target accuracy achieved!")
    else:
        print(f"target accuracy not reached. Final: {final_acc:.2f}%")
    
    wandb.finish()


if __name__ == "__main__":
    main()
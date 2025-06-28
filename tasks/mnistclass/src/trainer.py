import os
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb



class Trainer:#define the training logic
    
    def __init__(self, model, train_loader, test_loader, optimizer, cfg):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Create save directory
        os.makedirs(cfg.training.save_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")#progress bar 
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100. * correct / total:.2f}%"
            })
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        best_accuracy = 0
        
        for epoch in range(self.cfg.training.epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.training.epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Step scheduler
            self.scheduler.step()#steps learning rate
            
            # Log to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            })
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"New best accuracy: {best_accuracy:.2f}%")
            
            # Early stopping if target accuracy reached
            if val_acc >= 99:
                print(f"Target accuracy reached: {val_acc:.2f}%")
                break
        
        print(f"\nTraining completed. Best accuracy: {best_accuracy:.2f}%")
        return best_accuracy
    
    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accuracy": accuracy,
            "config": self.cfg,
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.cfg.training.save_dir, "checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.cfg.training.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        #load saved model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["accuracy"]
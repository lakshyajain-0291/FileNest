#evaluate perfromance on test

import hydra
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.model import create_model
from src.dataset import get_dataloaders
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    
    print("Evaluatiion started")
    
    # Set seed
    set_seed(cfg.seed)
    
    # Get device
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    _, test_loader = get_dataloaders(cfg)
    
    # Create model
    model = create_model(cfg)
    model.to(device)
    
    # Load best model
    checkpoint_path = f"{cfg.training.save_dir}/best_model.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"loaded model from {checkpoint_path}")
        print(f"model trained for {checkpoint['epoch']} epochs")
        print(f"Best accuracy during training: {checkpoint['accuracy']:.2f}%")
    except FileNotFoundError:
        print(f"No checkpoint found at {checkpoint_path}")
        print("model not trained")
        return
    
    # evaluate model
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    test_loss = 0
    
    print("running evaluation.")
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / total
    avg_loss = test_loss / total
    
    print(f"\nTest Results:")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    
    # Check if target met
    # if accuracy >= 98.5:
    #     print("Target accuracy of 98.5% achieved!")
    # else:
    #     print(f"Target accuracy not reached. Current: {accuracy:.2f}%")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as confusion_matrix.png")
    
    # Sample predictions visualization
    print("\nSample predictions:")
    model.eval()
    with torch.no_grad():
        # Get first batch
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        
        # Show first 10 samples
        for i in range(min(10, len(data))):
            actual = target[i].item()
            predicted = pred[i].item()
            confidence = F.softmax(output[i], dim=0)[predicted].item()
            status = "✅" if actual == predicted else "❌"
            print(f"Sample {i+1}: Actual={actual}, Predicted={predicted}, "
                  f"Confidence={confidence:.3f} {status}")


if __name__ == "__main__":
    main()
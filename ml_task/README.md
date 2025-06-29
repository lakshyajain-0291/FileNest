MNIST MiniMobileNet with Hyperparameter Optimization
A lightweight MobileNet-inspired CNN for MNIST digit classification with automated hyperparameter tuning using Ray Tune and experiment tracking with Weights & Biases.

🎯 Objective
Achieve 98.5%+ accuracy on MNIST dataset with less than 10,000 parameters using an optimized MobileNet architecture.

🏗️ Architecture
OptimizedMiniMobileNet
Parameter Count: ~8,500-9,500 parameters (under 10k constraint)

Architecture: Depthwise separable convolutions with batch normalization

Layers:

Initial convolution: 1→12 channels

Depthwise separable block 1: 12→24 channels

Depthwise separable block 2: 24→32 channels (stride=2)

Depthwise separable block 3: 32→24 channels (stride=2)

Global average pooling + dropout + linear classifier

Key Features
Depthwise Separable Convolutions: Efficient parameter usage

Batch Normalization: Training stability and faster convergence

Strategic Downsampling: Spatial reduction with stride=2

Kaiming Weight Initialization: Improved training dynamics

Dropout Regularization: Prevents overfitting

🚀 Features
Automated Hyperparameter Optimization with Ray Tune

Experiment Tracking with Weights & Biases

Configuration Management with Hydra

Data Augmentation (rotation, translation)

Learning Rate Scheduling (StepLR)

GPU Acceleration support

📋 Requirements
bash
pip install torch torchvision
pip install ray[tune]
pip install wandb
pip install hydra-core omegaconf
pip install pandas
📁 Project Structure
text
.
├── train.py          # Main training script
├── config.yaml       # Hydra configuration file
├── README.md         # This file
└── dataset/          # MNIST data (auto-downloaded)
⚙️ Configuration
Create a config.yaml file:

text
model:
  num_classes: 10
  dropout: 0.3

training:
  num_epochs: 20

data:
  root: "dataset/"
  batch_sizes: [32, 64, 128]
  
augmentation:
  rotation_degrees: 10
  translate: [0.1, 0.1]
  normalize_mean: [0.1307]
  normalize_std: [0.3081]

hyperparameter_search:
  lr_min: 1e-4
  lr_max: 1e-2
  num_samples: 5
  metric: "mean_accuracy"
  mode: "max"

wandb:
  project: "mnist-minimobilenet"

ray_tune:
  scheduler:
    step_size: 7
    gamma: 0.1
🏃‍♂️ Usage
Basic Training
bash
python train.py
Override Configuration Parameters
bash
# Change number of epochs
python train.py training.num_epochs=30

# Modify dropout rate
python train.py model.dropout=0.5

# Change number of Ray Tune samples
python train.py hyperparameter_search.num_samples=10

# Multiple overrides
python train.py training.num_epochs=50 model.dropout=0.2 hyperparameter_search.num_samples=8
Custom Configuration File
bash
python train.py --config-name=custom_config
🔧 Hyperparameter Search
The system automatically optimizes:

Learning Rate: Log-uniform distribution (1e-4 to 1e-2)

Batch Size: Choice of

Fixed Epochs: 20 (configurable)

Uses ASHA Scheduler for efficient resource allocation and early stopping of poor-performing trials.

📊 Experiment Tracking
Weights & Biases Integration
Real-time loss and accuracy tracking

Hyperparameter comparison

Model performance visualization

Automatic experiment logging

Metrics Logged
Training loss per batch

Validation accuracy per epoch

Hyperparameter configurations

Model architecture details

🎯 Performance Optimizations
Data Augmentation
Random Rotation: ±10 degrees

Random Translation: ±10% in both directions

Normalization: MNIST-specific statistics

Training Enhancements
Learning Rate Scheduling: StepLR with decay

Proper Weight Initialization: Kaiming initialization

Batch Normalization: After each convolution

Test Set Validation: Accurate performance measurement

📈 Expected Results
Target Accuracy: 98.5%+

Parameter Count: <10,000

Training Time: ~5-10 minutes on GPU

Convergence: Typically within 15-20 epochs

🐛 Troubleshooting
Common Issues
Missing config.yaml

text
MissingConfigException: Cannot find primary config 'config'
Solution: Create the config.yaml file in the project directory

CUDA Out of Memory

text
RuntimeError: CUDA out of memory
Solution: Reduce batch size in configuration

Ray Tune Import Error

text
ModuleNotFoundError: No module named 'ray'
Solution: Install Ray with pip install ray[tune]

Wandb Authentication

text
wandb: ERROR Unable to authenticate
Solution: Run wandb login and enter your API key

🔬 Architecture Details
Parameter Distribution
Depthwise Convolutions: ~300 parameters

Pointwise Convolutions: ~7,500 parameters

Batch Normalization: ~200 parameters

Final Classifier: ~250 parameters

Total: ~8,250 parameters

Design Principles
Efficiency: Depthwise separable convolutions reduce parameters by 8-9x

Stability: Batch normalization enables faster, more stable training

Regularization: Dropout and data augmentation prevent overfitting

Scalability: Modular design allows easy architecture modifications

📝 License
This project is open source and available under the MIT License.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📚 References
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

Ray Tune Documentation

Hydra Configuration Framework

Weights & Biases
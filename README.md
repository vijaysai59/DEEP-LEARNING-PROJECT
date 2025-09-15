# DEEP-LEARNING-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: CHITTAMSETTY NAGA VIJAY SAI

*INTERN ID*: CT08DZ1376

*DOMAIN*: DATA SCIENCE

*DURATION*: 8 WEEKS

*MENTOR*:  NEELA SANTHOSH KUMAR 


*DESCRIPTION*:

This project implements a Convolutional Neural Network (CNN) to classify images using TensorFlow /Keras.
It was created as part of Task 2 of the Data-Science internship assignment.

Default dataset: CIFAR-10
 (10 classes of 32×32 colour images).

Optional dataset: MNIST (hand-written digits) for a quick demo run.

Goal: train a deep learning model, evaluate its accuracy, and save the best performing model and training history.

Key Features

CNN architecture with Conv2D → BatchNorm → MaxPool blocks.

EarlyStopping and ModelCheckpoint callbacks.

Automatic saving of:

outputs/best_model.h5 – best validation-accuracy weights.

outputs/final_model.h5 – final trained model.

outputs/acc.png / loss.png – accuracy & loss plots.

outputs/history.csv – epoch-wise metrics.

Command-line arguments to choose dataset, epochs, and batch size.

File Structure
task2_project/
│
├─ task2_deep_learning.py   # main training script
├─ outputs/                 # created after training
│   ├─ best_model.h5
│   ├─ final_model.h5
│   ├─ acc.png
│   ├─ loss.png
│   └─ history.csv
└─ README.md                # this file

Installation & Usage
# Clone repository
git clone https://github.com/<your-username>/task2_project.git
cd task2_project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow matplotlib scikit-learn

# Run training (CIFAR-10, 25 epochs)
python task2_deep_learning.py

# Quick test with MNIST, fewer epochs
python task2_deep_learning.py --dataset mnist --epochs 10

Results

On CIFAR-10, the model typically reaches ~70–75 % test accuracy after 25 epochs on a CPU-only machine.
Plots of training/validation accuracy and loss are saved in the outputs/ folder.

Requirements

Python 3.8 – 3.11

TensorFlow 2.x

matplotlib

scikit-learn

License

MIT License (or any license you prefer)

Tip for reviewers: If you just want to see a quick run, use the MNIST dataset:

python task2_deep_learning.py --dataset mnist --epochs 5

*OUTPUT*:

<img width="640" height="480" alt="acc" src="https://github.com/user-attachments/assets/d2e795b0-44f0-4217-a26b-b8c3a401cca9" />




<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/6d3dc098-bd19-4dfb-818e-92eb90b94c6e" />




<img width="676" height="378" alt="Screenshot 2025-09-15 213149" src="https://github.com/user-attachments/assets/330712ae-2b5f-4724-b208-fb6a0c58f525" />




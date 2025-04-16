# Dental X-ray Segmentation with U-Net++
This project focuses on **automatic tooth condition segmentation** using the [DENTEX dataset](https://huggingface.co/datasets/ibrahimhamamci/DENTEX). The goal is to identify dental issues such as **Impacted Teeth**, **Caries**, **Deep Caries**, and **Periapical Lesions** from X-ray images using a deep learning model based on the **U-Net++** architecture.

## 🧠 Model Architecture
- **Model:** U-Net++ (Nested U-Net)
- **Framework:** PyTorch
- **Loss Function:** Dice Loss
- **Input Size:** 256x256
- **Output Classes:** 4 (multi-class segmentation)

## 📁 Project Structure
my_project/
│
├── DENTEX/                       # Contains the dataset used for training and validation
│   ├── training_data/            # Training dataset (includes images and annotations)
│   │   ├── quadrant/             # Folder with X-ray images and associated JSON annotations for quadrant-based training
│   │   ├── quadrant-enumeration-disease/  # Folder with images and annotations for quadrant enumeration (includes disease annotations)
│   │   └── unlabelled/           # Folder with unlabelled images for model inference or semi-supervised learning
│   │
│   ├── validation_data/          # Validation dataset
│   │   ├── quadrant_enumeration_disease/  # Validation images for quadrant enumeration with disease annotations
│   │   └── validation_triple.json  # JSON file with validation annotations
│   │
│   └── disease/                  # Folder with images and labels for disease detection (e.g., Impacted, Caries)
│       ├── input/                # X-ray images for testing
│       └── label/                # JSON annotation labels for test images
│
├── .venv/                         # Virtual environment (Python dependencies)
├── datasets.py                    # Code for loading and preprocessing the dataset
├── train.py                       # Script for training the model
├── unet_plus_plus.py              # Definition of the U-Net++ model architecture
├── requirements.txt               # List of required Python packages
└── README.md                      # Project documentation (this file)



## 🚀 How to Run
**Clone the repo**:
```bash
git clone https://github.com/yourusername/model_prototype.git
cd model_prototype

pip install -r requirements.txt
python train.py

🧪 Dataset
We use the DENTEX dataset for training and validation, which contains labeled dental X-rays. Labels are stored in JSON format, and classes include:
Normal
Caries
Deep Caries
Impacted
Periapical Lesion

📊 Training Details
Epochs: 50 (can be adjusted)
Batch Size: 4 (adjustable)
Device: CPU or GPU (auto-detected)

📈 Sample Output (After Training)
Input X-ray	Ground Truth	Prediction

📸 Add real images from your visualizations after training!

🛠️ TODO
 Add validation metrics (IoU, accuracy)
 Add GUI for visualization

## 📄 License
This project is **not** open-source.

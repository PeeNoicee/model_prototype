# Dental X-ray Segmentation with U-Net++
This project focuses on **automatic tooth condition segmentation** using the [DENTEX dataset](https://huggingface.co/datasets/ibrahimhamamci/DENTEX). The goal is to identify dental issues such as **Impacted Teeth**, **Caries**, **Deep Caries**, and **Periapical Lesions** from X-ray images using a deep learning model based on the **U-Net++** architecture.

## 🧠 Model Architecture
- **Model:** U-Net++ (Nested U-Net)
- **Framework:** PyTorch
- **Loss Function:** Dice Loss
- **Input Size:** 256x256
- **Output Classes:** 4 (multi-class segmentation)

## 📁 Project Structure
![image](https://github.com/user-attachments/assets/1d34ffe3-d06e-41a2-8e31-df8f80722f5e)




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

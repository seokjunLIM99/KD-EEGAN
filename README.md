# KD-EEGAN

**KD-EEGAN** is a knowledge-distilled GAN-based image restoration model designed to enhance image quality under **low-light** and **motion-blur** conditions.  
It improves smoke detection performance in UAV-captured imagery by integrating a **teacher–student GAN framework** with detection-aware optimization.

---

## 🚀 How to Run

### 🔹 Training
1. Open the file `script.py`
2. Locate the variable `mode` and set it to **"train"**
3. Run the following command:
   ```bash
   python script.py
📝 This will start the training process using the predefined configurations in the script.

🔹 Testing
Open the file script.py

Change the mode variable to "test"

Run the following command:

bash
코드 복사
python script.py
🧪 This will perform model inference using the pretrained weights.

💾 Pretrained Weights & Configuration Files
You can download the following resources from Google Drive:

Pretrained Teacher weights

Proposed model weights

YOLO pretrained weights

YOLO configuration files (.cfg)

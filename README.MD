# GreenSort - Intelligent Waste Classification System

![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?logo=tensorflow&style=flat-square)

**GreenSort** is a computer vision-based waste classification system designed to facilitate B2B recycling transactions. Powered by deep learning, GreenSort classifies waste materials from images, estimates their market value, and calculates shipping costs based on quantity and distance.

The model is trained using the [RealWaste dataset](https://archive.ics.uci.edu/dataset/908/realwaste) and supports nine waste categories: **Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation**.

---

## 🚀 Key Features

- **Automated Waste Classification**: Classifies waste types from images with >90% accuracy using deep learning.
- **Price Estimation**: Calculates the value of recyclable materials based on type and weight.
- **Shipping Cost Calculation**: Estimates shipping costs based on distance and delivery method.
- **Interactive GUI**: `testing.py` script provides a user-friendly interface for testing.
- **Browser-Compatible Model**: Converted to TensorFlow.js for use in web applications.
- **Educational Add-On (Planned)**: Future integration of consultation and sustainability education features.

---

## 🗂️ Project Structure
```
GreenSort/
├── data/
│   └── realwaste/               # RealWaste dataset (download separately)
├── models/
│   └── greensort_model.h5       # Trained Keras model
│   └── tfjs_model/              # TensorFlow.js model for browser deployment
├── notebooks/
│   └── CapstoneDBS.ipynb # Jupyter notebook for training and evaluation
├── src/
│   └── utils.py                 # Helper functions for preprocessing and evaluation
├── testing.py                   # GUI tool for testing the model
├── requirements.txt             # Python dependencies
├── greensort_icon.ico           # Optional app icon
├── LICENSE                      # MIT License file
└── README.md                    # Project documentation
```

---

## ✅ Prerequisites

- **Python** 3.8 or higher
- **pip** for dependency management
- **RealWaste Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)
- **GPU (Optional)** for faster model training

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/greensort.git
   cd greensort
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```
   tensorflow>=2.12.0
   tensorflowjs>=4.0.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   pillow>=9.5.0
   scikit-learn>=1.2.0
   tk>=0.1.0           # For GUI (included with Python)
   ```

4. **Download the RealWaste dataset** and place it in `data/realwaste/`.

5. **Prepare the trained model**:
   - Place `greensort_model.h5` in the `models/` directory.
   - Or train the model yourself using `notebooks/CapstoneDBS.ipynb`.

---

## 🔧 Usage

### 1. Training the Model
- Open `notebooks/CapstoneDBS.ipynb` in Jupyter Notebook or JupyterLab.
- Follow the steps to:
  - Preprocess the dataset
  - Train the model (recommended: MobileNetV2)
  - Evaluate accuracy, precision, recall, and F1-score
- Save the model to `models/greensort_model.h5`
- Convert to TensorFlow.js:
  ```python
  import tensorflowjs as tfjs
  tfjs.converters.save_keras_model(model, 'models/tfjs_model')
  ```

### 2. Testing with the GUI
- Run the interface:
  ```bash
  python testing.py
  ```

- GUI features:
  - **Select Image**: Upload a waste image
  - **Set Quantity**: Input weight in kg
  - **Classify Waste**: View top prediction, confidence score, and estimated value
  - **Clear Results**: Reset the interface

- Example:
  - Image: plastic waste
  - Weight: 2.5 kg
  - Output: “Plastic” with 95% confidence → Estimated value: IDR 10,000 (based on IDR 4,000/kg)

### 3. Web Deployment
- Use the model in `models/tfjs_model/`
- Example JavaScript code:
  ```javascript
  import * as tf from '@tensorflow/tfjs';
  const model = await tf.loadLayersModel('path/to/tfjs_model/model.json');
  const prediction = model.predict(imageTensor);
  ```

---

## ♻️ Waste Categories & Pricing

| Category             | Price per kg (IDR) |
|----------------------|--------------------|
| Cardboard            | 2,000              |
| Food Organics        | 500                |
| Glass                | 1,500              |
| Metal                | 8,000              |
| Miscellaneous Trash  | 300                |
| Paper                | 2,500              |
| Plastic              | 4,000              |
| Textile Trash        | 1,000              |
| Vegetation           | 600                |

> *Note: Prices are for demonstration and may vary by region.*

---

## 📈 Model Performance

- **Architecture**: MobileNetV2 (fine-tuned)
- **Dataset**: RealWaste (9 classes)
- **Metrics**:
  - Accuracy: 91.2%
  - Precision (avg): 90.7%
  - Recall (avg): 89.8%
  - F1-Score (avg): 90.2%

**Challenges**:
- Visual similarity between classes (e.g., Paper vs. Cardboard)

**Suggestions**:
- Improve feature extraction
- Use more advanced data augmentation

---

## 🔮 Future Improvements

- **Multi-label classification** for multiple waste types in one image
- **Real-time shipping integration** via APIs (e.g., Google Maps)
- **Chatbot assistant** for education and user support
- **Model optimization** for edge devices (quantization, pruning)
- **Dataset expansion** to increase robustness

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m "Add feature"`)
4. Push to GitHub (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) and PEP 8 style guidelines.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE)

---

## 📬 Contact

- Email: [bayuardi30@outlook.com](mailto:bayuardi30@outlook.com)
- GitHub Issues: [Open an issue](https://github.com/yourusername/greensort/issues)

---

## 🙏 Acknowledgments

- [RealWaste Dataset](https://archive.ics.uci.edu/dataset/908/realwaste) — UCI Machine Learning Repository
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow.js](https://www.tensorflow.org/js)

---

**© 2025 GreenSort Team – Empowering sustainable waste management through technology.**
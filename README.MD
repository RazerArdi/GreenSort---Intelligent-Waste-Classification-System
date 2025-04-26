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
│   ├── greensort_model.h5       # Trained Keras model
│   ├── greensort_price_model_tf.h5  # Price prediction model
│   ├── preprocessing_metadata.json  # Metadata for preprocessing
│   └── tfjs_model/              # TensorFlow.js model for browser deployment(ComputerVision)
├── notebooks/
│   └── CapstoneDBS_ComputerVision.ipynb.
|   └── CapstoneDBS_PricingModel.ipynb
├── src/
│   └── utils.py                 # Helper functions for preprocessing and evaluation
├── testing.py                   # GUI tool for testing the model
├── requirements.txt             # Python dependencies
├── greensort_icon.ico           # Optional app icon
├── LICENSE                      # MIT License file
├── CHANGELOG.md                 # Version history and changes
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
   pandas>=2.0.0
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

### 2. Price Prediction Model
- The price prediction model uses a neural network to estimate waste prices based on:
  - Waste type
  - Weight (kg)
  - Shipping distance (km)
  - Market price (IDR/kg)
- Evaluation metrics:
  - Mean Absolute Error (MAE): ~Rp5,000
  - R² Score: 0.95

### 3. Testing with the GUI
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
  - Output: "Plastic" with 95% confidence → Estimated value: IDR 10,000 (based on IDR 4,000/kg)

### 4. Web Deployment
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
| Cardboard            | 2,500              |
| Food Organics        | 800                |
| Glass                | 2,000              |
| Metal                | 10,000             |
| Miscellaneous Trash  | 500                |
| Paper                | 3,000              |
| Plastic              | 5,000              |
| Textile Trash        | 1,500              |
| Vegetation           | 1,000              |

> *Note: Prices fluctuate based on market conditions. The model accounts for a ±20% price variation.*

---

## 📈 Model Performance

### Image Classification Model (MobileNetV2)
- **Architecture**: MobileNetV2 (fine-tuned)
- **Dataset**: RealWaste (9 classes)
- **Metrics**:
  - Accuracy: 91.2%
  - Precision (avg): 90.7%
  - Recall (avg): 89.8%
  - F1-Score (avg): 90.2%

### Price Prediction Model
- **Architecture**: Neural Network with Dropout
- **Dataset**: Synthetic (1000 samples)
- **Metrics**:
  - MAE: ~Rp5,000
  - R² Score: 0.95
  - Relative Error: 2-3% of average price

**Challenges**:
- Visual similarity between classes (e.g., Paper vs. Cardboard)
- Market price fluctuations affecting pricing model accuracy

**Suggestions**:
- Improve feature extraction
- Use more advanced data augmentation
- Incorporate real market data for price predictions

---

## 🔮 Future Improvements

- **Multi-label classification** for multiple waste types in one image
- **Real-time shipping integration** via APIs (e.g., Google Maps)
- **Chatbot assistant** for education and user support
- **Model optimization** for edge devices (quantization, pruning)
- **Dataset expansion** to increase robustness
- **Real market data integration** for more accurate price predictions

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

## 📦 CHANGELOG

All notable changes to this project are documented in the [CHANGELOG.md](CHANGELOG.md) file.

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
- Kementerian Lingkungan Hidup dan Kehutanan (KLHK) for waste management data

---

**© 2025 GreenSort Team – Empowering sustainable waste management through technology.**
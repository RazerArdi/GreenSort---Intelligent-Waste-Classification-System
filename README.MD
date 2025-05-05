# GreenSort - Intelligent Waste Classification System

![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?logo=tensorflow&style=flat-square)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-4.0+-FF6F00?logo=javascript&style=flat-square)

**GreenSort** is a computer vision-based waste classification system designed to facilitate B2B recycling transactions. Powered by deep learning, GreenSort classifies waste materials from images, estimates their market value, and calculates shipping costs based on quantity and distance. The system is deployable as a desktop GUI or a web application using TensorFlow.js.

The model is trained using the [RealWaste dataset](https://archive.ics.uci.edu/dataset/908/realwaste) and supports nine waste categories: **Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation**.

---

## 🚀 Key Features

- **Automated Waste Classification**: Classifies waste types from images with >90% accuracy using MobileNetV2.
- **Recommendation System**: Suggests waste handling actions based on type and weight using supervised learning.
- **Shipping Cost Calculation**: Estimates shipping costs based on distance.
- **Interactive GUI**: `testing.py` provides a user-friendly desktop interface for testing.
- **Web Deployment**: TensorFlow.js models enable browser-based classification and pricing predictions.
- **Educational Add-On (Planned)**: Future integration of consultation and sustainability education features.

---

## 🗂️ Project Structure
``` │, sudut ├── dan └──
GreenSort/
├── data/
│   └── realwaste-main/
│       ├── realwaste/                      # RealWaste dataset (download separately)
│       └── rekomendasi/                    # Recommendation dataset  
├── models/
│   ├── ComputerVision/
│   │   ├── greensort_model.h5              # Trained Keras classification model
│   │   └── tfjs_model/                     # TensorFlow.js models for browser deployment
│   ├── Sistemrekomendasi/
│       ├── models/                         # Trained Keras recommendation model
│       └── dataset.json                    # Dataset for recommendation
├── notebooks/
│   ├── CapstoneDBS_ComputerVision.ipynb    # Notebook for classification model training
│   └── CapstoneDBS_SistemRekomendasi.ipynb # Notebook for recommendation model training
├── src/
│   └── utils.py                            # Helper functions for preprocessing and evaluation
├── web/
│   ├── db/                                 # sql for database
│   ├── index.html                          # Web app entry point
│   ├── app.js                              # JavaScript for model inference
│   └── styles.css                          # Web app styling
├── testing.py                              # GUI tool for desktop testing
├── requirements.txt                        # Python dependencies
├── greensort_icon.ico                      # Optional app icon
├── LICENSE                                 # MIT License file
├── CHANGELOG.md                            # Version history and changes
└── README.md                               # Project documentation
```

---

## ✅ Prerequisites

- **Python** 3.8 or higher (for training and GUI)
- **pip** for Python dependency management
- **Node.js** 16 or higher (for web deployment)
- **RealWaste Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste)
- **GPU (Optional)** for faster model training
- **Web Browser** (e.g., Chrome, Firefox) for web deployment
- **Web Server** (e.g., `http-server` or `Live Server` extension in VS Code) for local web testing

---

## ⚙️ Installation

### For Desktop GUI
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

3. **Install Python dependencies**:
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

5. **Prepare the trained models**:
   - Place `greensort_model.h5` and `recycling_recommendation_model.h5` in `models/`.
   - Or train the models using `notebooks/CapstoneDBS_ComputerVision.ipynb` and `notebooks/CapstoneDBS_SistemRekomendasi.ipynb`.

### For Web Deployment
1. **Install Node.js** (if not already installed):
   - Download from [nodejs.org](https://nodejs.org/) or use a package manager:
     ```bash
     sudo apt install nodejs npm  # On Ubuntu/Debian
     ```

2. **Install TensorFlow.js**:
   ```bash
   npm install @tensorflow/tfjs
   ```

3. **Convert Keras models to TensorFlow.js** (if not already in `models/tfjs_model/`):
   ```bash
   pip install tensorflowjs
   tensorflowjs_converter --input_format keras models/greensort_model.h5 models/tfjs_model/classification
   tensorflowjs_converter --input_format keras models/recycling_recommendation_model.h5 models/tfjs_model/recommendation
   ```

4. **Set up a web server**:
   - Install a simple HTTP server:
     ```bash
     npm install -g http-server
     ```
   - Or use the `Live Server` extension in VS Code.

---

## 🔧 Usage

### 1. Training the Models
- Open `notebooks/CapstoneDBS_ComputerVision.ipynb` for the classification model and `notebooks/CapstoneDBS_SistemRekomendasi.ipynb` for the recommendation model in Jupyter Notebook or JupyterLab.
- Follow the steps to:
  - Preprocess the dataset
  - Train the models (MobileNetV2 for classification, neural network for pricing)
  - Evaluate metrics (accuracy, MAE, R²)
- Save the models to `../models/ComputerVision/greensort_model.h5` and `../models/Sistemrekomendasi/recycling_recommendation_model.h5`.
- Convert to TensorFlow.js:
  ```python
  import tensorflowjs as tfjs
  tfjs.converters.save_keras_model(classification_model, 'models/tfjs_model/classification')
  tfjs.converters.save_keras_model(recommendation_model, 'models/tfjs_model/recommendation')
  ```

### 2. Testing with the GUI
- Run the desktop interface:
  ```bash
  python testing.py
  ```

- GUI features:
  - **Select Image**: Upload a waste image.
  - **Set Quantity and Distance**: Input weight (kg) and shipping distance (km).
  - **Classify Waste**: View top prediction, confidence score, and estimated value.
  - **View Results**: Displays classification and pricing details.

- Example:
  - Image: Plastic waste
  - Weight: 2.5 kg
  - Distance: 10 km
  - Output: "Plastic" with 95% confidence → Estimated value: Rp 10,000 (based on Rp 4,000/kg)

### 3. Web Deployment
- Create a web app in the `web/` directory with the following files:
  - `index.html`: Web interface
  - `app.js`: JavaScript for model inference
  - `styles.css`: Styling

- **Sample `web/index.html`**:
  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>GreenSort - Waste Classification</title>
      <link rel="stylesheet" href="styles.css">
      <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.0.0/dist/tf.min.js"></script>
  </head>
  <body>
      <div class="container">
          <h1>GreenSort - Waste Classification</h1>
          <input type="file" id="imageInput" accept="image/*">
          <div>
              <label>Quantity (kg):</label>
              <input type="number" id="quantity" value="1.0" min="0.1" step="0.1">
          </div>
          <div>
              <label>Distance (km):</label>
              <input type="number" id="distance" value="10.0" min="0.1" step="0.1">
          </div>
          <button onclick="classifyImage()">Classify Waste</button>
          <div id="result"></div>
          <img id="preview" style="max-width: 400px; display: none;">
      </div>
      <script src="app.js"></script>
  </body>
  </html>
  ```

- **Sample `web/app.js`**:
  ```javascript
  async function loadModels() {
      const classificationModel = await tf.loadLayersModel('/models/tfjs_model/classification/model.json');
      const pricingModel = await tf.loadLayersModel('/models/tfjs_model/pricing/model.json');
      return { classificationModel, pricingModel };
  }

  const CATEGORIES = ['Cardboard', 'Food_Organics', 'Glass', 'Metal', 'Miscellaneous_Trash', 'Paper', 'Plastic', 'Textile_Trash', 'Vegetation'];
  const PRICE_PER_KG = {
      'Cardboard': 2000, 'Food_Organics': 500, 'Glass': 1500, 'Metal': 8000,
      'Miscellaneous_Trash': 300, 'Paper': 2500, 'Plastic': 4000, 'Textile_Trash': 1000, 'Vegetation': 600
  };

  async function classifyImage() {
      const imageInput = document.getElementById('imageInput');
      const quantity = parseFloat(document.getElementById('quantity').value);
      const distance = parseFloat(document.getElementById('distance').value);
      const resultDiv = document.getElementById('result');
      const preview = document.getElementById('preview');

      if (!imageInput.files[0]) {
          resultDiv.innerHTML = 'Please select an image.';
          return;
      }

      const { classificationModel, pricingModel } = await loadModels();

      // Load and preprocess image
      const img = new Image();
      img.src = URL.createObjectURL(imageInput.files[0]);
      img.onload = async () => {
          preview.src = img.src;
          preview.style.display = 'block';

          const tensor = tf.browser.fromPixels(img)
              .resizeNearestNeighbor([224, 224])
              .toFloat()
              .div(tf.scalar(255.0))
              .expandDims();

          // Classification
          const prediction = await classificationModel.predict(tensor).data();
          const predictedClassIndex = prediction.indexOf(Math.max(...prediction));
          const predictedClass = CATEGORIES[predictedClassIndex];
          const confidence = prediction[predictedClassIndex] * 100;

          // Pricing
          const pricePerUnit = PRICE_PER_KG[predictedClass];
          const pricingInput = tf.tensor2d([[...Array(9).fill(0).map((_, i) => i === predictedClassIndex ? 1 : 0), quantity, distance, pricePerUnit]]);
          const totalPrice = (await pricingModel.predict(pricingInput).data())[0];

          // Display results
          resultDiv.innerHTML = `
              <h3>Classification Result</h3>
              <p><strong>Type:</strong> ${predictedClass}</p>
              <p><strong>Confidence:</strong> ${confidence.toFixed(2)}%</p>
              <h3>Pricing Result</h3>
              <p><strong>Price per kg:</strong> Rp ${pricePerUnit.toLocaleString()}</p>
              <p><strong>Quantity:</strong> ${quantity.toFixed(2)} kg</p>
              <p><strong>Distance:</strong> ${distance.toFixed(2)} km</p>
              <p><strong>Total Value:</strong> Rp ${totalPrice.toLocaleString(undefined, {minimumFractionDigits: 0})}</p>
          `;
      };
  }
  ```

- **Sample `web/styles.css`**:
  ```css
  body {
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 20px;
      background-color: #f0f0f0;
  }
  .container {
      max-width: 800px;
      margin: 0 auto;
      background: white;
      padding: 20px;
      border-radius: 8px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
  }
  h1 {
      color: #27ae60;
  }
  input, button {
      margin: 10px 0;
      padding: 8px;
  }
  button {
      background-color: #27ae60;
      color: white;
      border: none;
      cursor: pointer;
  }
  button:hover {
      background-color: #219653;
  }
  #result {
      margin-top: 20px;
  }
  ```

- **Run the web app**:
  ```bash
  http-server web/ -p 8080
  ```
  - Open `http://localhost:8080` in a browser.
  - Upload an image, set quantity and distance, and click "Classify Waste" to see results.

### 4. Database
- Create a database `greensort.sql`
  ![ERD GreenSort](web/db/erd_greensort.png)

- **Sample** `db/greensort.sql`:
  ```sql  
  -- Table: waste_categories
  CREATE TABLE waste_categories (
    category_id INT PRIMARY KEY AUTO_INCREMENT,
    category_name VARCHAR(50) NOT NULL UNIQUE,
    base_market_price DECIMAL(10,2) NOT NULL
  );
     
  -- Table: users (optional)
  CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  -- Table: classifications
  CREATE TABLE classifications (
    classification_id INT PRIMARY KEY AUTO_INCREMENT,
    image_path VARCHAR(255) NOT NULL,
    category_id INT NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    classification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INT,
    FOREIGN KEY (category_id) REFERENCES waste_categories(category_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
  );

  -- Tabel referensi rekomendasi
  CREATE TABLE recommendation_rules (
    rule_id INT PRIMARY KEY AUTO_INCREMENT,
    category_id INT NOT NULL,
    weight_min_kg DECIMAL(10,2) NOT NULL,
    weight_max_kg DECIMAL(10,2) NOT NULL,
    recomamendation TEXT NOT NULL,
    FOREIGN KEY (category_id) REFERENCES waste_categories(category_id)
  );

  -- Tabel log hasil rekomendasi
  CREATE TABLE recommendations (
    recommendation_id INT PRIMARY KEY AUTO_INCREMENT,
    classification_id INT NOT NULL,
    estimated_weight DECIMAL(10,2) NOT NULL,
    generated_recommendation TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (classification_id) REFERENCES classifications(classification_id)
  );
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

> *Note: Prices fluctuate based on market conditions. The pricing model accounts for a ±20% price variation.*

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
- **Enhanced web UI** with real-time image previews and advanced visualizations

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
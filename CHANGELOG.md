# 📦 CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.6.1] - 2025-05-18
> Created by **Bayu Ardiyansyah** 

### 🎉 Added
- add webAPI (`API.JS`, `html.API` , `styleAPI.css`)

## 🛠️ Changed
- Replaced TensorFlow.js recommendation model with a rule-based system fetching recommendations from `dataset.json`, enabling category- and weight-based lookups.
- Updated `MODEL_URLS` to rename `recommendation` to `recommendationData`, pointing to `https://modelai14.s3.ap-southeast-2.amazonaws.com/dataset.json`.
- Modified `getRecommendation` to query `dataset.json` for recommendations, mapping English categories (e.g., "Cardboard") to Indonesian (e.g., "Kardus") for consistency.
- Adjusted image preprocessing in `preprocessImage` to include MobileNetV2-specific normalization ([-1, 1] range) for accurate classification.
- Add Testing_unit `tes.html`.

## 🐛 Fixed
- Resolved `Unknown regularizer: L2` error in classification model by registering a custom `L2Regularizer` class with `l2=0.0001`, matching the Keras model’s configuration.
- Fixed `Failed to load models: Recommendation model failed to load` error by removing invalid `tf.loadLayersModel` call for `dataset.json` and using `fetch` to load JSON data.
- Improved error handling for `dataset.json` fetch failures, adding HTTP status checks and user-friendly alerts.
- Ensured proper tensor disposal in `classifyWithModel` to prevent memory leaks during image classification.

## Issue
- UNIFX GUI `testingAPI` (yet)

---
## [6.5.6] - 2025-05-15
> Created by **Bayu Ardiyansyah** 

### 🛠️ Changed
- Revision for Web interface
- Revision for CapstoneDBS_ComputerVision.ipynb (MLFLOW)
- Revision for CapstoneDBS/Recommendation System.ipynb (MLFLOW)
- Retrain Model

---

## [6.5.2] - 2025-05-14
> Created by **Bayu Ardiyansyah** 

### 🎉 Added
- Initial release of the GreenSort web application.
- Implemented web interface with `index.html`, `style.css`, and `app.js`.
- Added tabbed navigation for "Classify Waste" and "About" sections.
- Integrated image upload and quantity input for waste classification.
- Added mock classification logic in `app.js` to simulate waste category predictions (9 categories).
- Implemented pricing calculations based on predefined rates (e.g., Cardboard: Rp 2,000/kg).
- Added mock recommendation system with static dataset for recycling suggestions.
- Included processed image modal to display 224x224 resized image.
- Added responsive design with CSS grid and media queries for mobile compatibility.
- Integrated Font Awesome (v6.2.0) for icons and Chart.js (v3.9.1) for potential future charting (not used in v1.0.0).
- Added demo mode warning to indicate simulated classifications.

### 🛠️ Changed
- Transitioned from Python/Tkinter GUI (using TensorFlow and Matplotlib) to a static web application.
- Replaced server-side model loading (`greensort_model.h5`, `recycling_recommendation_model.h5`) with client-side mock logic in `app.js`.
- Converted recommendation dataset from `dataset.json` to embedded JavaScript object in `app.js`.
- Updated UI to use modern web technologies (HTML5, CSS3, JavaScript ES6) instead of Tkinter widgets.
- Standardized category mappings (e.g., Cardboard → Kardus) for consistency with Python version.

### 🐛 Fixed
- Ensured file input works correctly when served via a local HTTP server (e.g., `http-server` or Python `http.server`).
- Fixed potential CORS issues by relying on CDN resources (Font Awesome, Chart.js) or local files.
- Corrected modal display logic to prevent overlapping content in the processed image view.

### 🗑️ Removed
- Removed dependency on Python, TensorFlow, Tkinter, Matplotlib, and Pandas from the Python GUI version.
- Eliminated server-side model loading and dataset file (`dataset.json`) in favor of client-side simulation.
- Removed Matplotlib-based processed image display in favor of HTML5 Canvas.

---

## [5.5.1] - 2025-05-01 
> Created by **Muhammad Rofi'ul Arham**

### 🎉 Added
-  _N/A_

### 🛠️ Changed
-  Add the dataset `dataset_rekomendasi_daur_ulang.json`

### 🐛 Fixed
- Fixed the integration of `testing.py` with the recommendation system (Issue resolved).
- Fixed the case index retrieval in `CapstoneDBS_SystemRecommendation.ipynb`.

### 🗑️ Removed
- _N/A_

---

## [5.5.0] - 2025-04-28 
> Created by **Bayu Ardiyansyah**

### 🎉 Added
-  System recomendation in `CapstoneDBS_SistemRekomendasi.ipynb` at testing 2 ()

### 🛠️ Changed
-  _N/A_

### 🐛 Fixed
- _N/A_

### 🗑️ Removed
- _N/A_

---

## [5.4.0] - 2025-04-28 
> Created by **Bayu Ardiyansyah**

### 🎉 Added
- add `dataset_rekomendasi_daur_ulang.json` as system recomendation dataset
- Setting up for Recomendation models `models/Sistemrekomendasi/dataset.json` and `models/Sistemrekomendasi/models/recycling_recommendation_model.h5`

### 🛠️ Changed
- Integration of`testing.py` with system recomendation (UnFixed(issue))
- Change `CapstoneDBS_SystemRecommendation.ipynb` as The new prototype (New one)

### 🐛 Fixed
- _N/A_

### 🗑️ Removed
- _N/A_

---

## [5.0.1] - 2025-04-28 
> Created by **Bayu Ardiyansyah**

### 🎉 Added
- Initiate `CapstoneDBS_SystemRecommendation.ipynb` as initial prototype

### 🛠️ Changed
- _N/A_

### 🐛 Fixed
- Retrain the `CapstoneDBS_ComputerVision.ipynb` model and save the model.

### 🗑️ Removed
- _N/A_

---

## [4.3.0] - 2025-04-27 
> Created by **Muhammad Rofi'ul Arham**

### 🎉 Added
- Web application in `db/` directory for database.
- Detailed web deployment instructions in `README.md` with sample SQL Database.

### 🛠️ Changed
- Updated `README.md` by adding database sql and improved project structure for web support.

### 🐛 Fixed
- _N/A_

### 🗑️ Removed
- _N/A_

---

## [4.2.0] - 2025-04-26 (3)
> Created by **Bayu Ardiyansyah**

### 🎉 Added
- Web application in `web/` directory for browser-based waste classification and pricing using TensorFlow.js.
- Detailed web deployment instructions in `README.md` with sample HTML, JavaScript, and CSS.

### 🛠️ Changed
- Updated `README.md` with corrected pricing table and enhanced project structure for web support.

### 🐛 Fixed
- _N/A_

### 🗑️ Removed
- _N/A_

---

## [4.1.0] - 2025-04-26 (2)
> Created by **Bayu Ardiyansyah**

### 🎉 Added
- Enhanced error handling for model loading in `testing.py` to improve robustness of the GUI tool.
- Improved model status updates in `testing.py` to provide clearer feedback on classification and pricing model loading.

### 🛠️ Changed
- Updated `testing.py` to ensure consistent model loading status display, setting green status only when both classification and pricing models load successfully.

### 🐛 Fixed
- Resolved TensorFlow model loading error in `testing.py` by specifying `custom_objects` for `mse` loss function in pricing model (`greensort_price_model_tf.h5`).
- Fixed Tkinter callback `NameError` in `testing.py` by properly capturing exception messages in lambda functions for asynchronous status updates.

### 🗑️ Removed
- _N/A_

---

## [4.0.0] - 2025-04-26 (1)
> Created by **Bayu Ardiyansyah**

### 🎉 Added
- Add Jupyter Notebook for model training (`CapstoneDBS_PricingModel.ipynb`)
  - Neural network price prediction model for waste valuation
  - Preprocessing pipeline for price prediction features
  - TensorFlow.js export for price model integration in web applications
  - JavaScript inference code example for browser deployment
  - Expanded project documentation with price prediction details
  - Integration of shipping cost calculation based on distance

### 🛠️ Changed
- Jupyter Notebook for model training (`CapstoneDBS_PricingModel.ipynb`)
  - Enhanced waste pricing model to include market fluctuations (±20%)
  - Improved documentation with model evaluation metrics
- Updated README with comprehensive project structure

### 🐛 Fixed
- Fixed preprocessing pipeline for numerical features
- Improved model evaluation with residual analysis

### 🗑️ Removed
- Removed static pricing logic in favor of dynamic model predictions

---

## [3.0.0] - 2025-04-23
> Created by **Muhammad Rofi'ul Arham**

### 🎉 Added
- Pre-trained MobileNetV2 model for image-based waste classification
- Jupyter Notebook for model training (`CapstoneDBS_ComputerVision.ipynb`)

### 🛠️ Changed
- _N/A_

### 🐛 Fixed
- Improved accuracy of MobileNetV2 model for garbage classification up to 89%

### 🗑️ Removed
- Removed the InceptionV3 model from the model training file (`CapstoneDBS_ComputerVision.ipynb`), visible in the preview file only (`preview_inception.ipynb`)

## [2.0.0] - 2025-04-22
> Created by **Bayu Ardiyansyah** and **Muhammad Rofi'ul Arham**

### 🎉 Added
- Added **InceptionV3** model for trash classification, as a comparison with previous models (MobileNetV2 and custom CNN)
- Support for InceptionV3 model training in `CapstoneDBS_ComputerVision.ipynb`
- Expanded EDA visualization in training notebook

### 🛠️ Changed
- Updated waste category pricing table with more accurate values
- Improved preprocessing pipeline for better model performance

### 🐛 Fixed
- _N/A_

### 🗑️ Removed
- _N/A_

---

## [1.0.0] - 2025-04-17
> Created by **Bayu Ardiyansyah**

### 🎉 Added
- Initial version of the **GreenSort** waste classification system
- Pre-trained MobileNetV2 model for image-based waste classification
- Jupyter Notebook for model training (`CapstoneDBS_ComputerVision.ipynb`)
- GUI-based testing tool (`testing.py`) for local image testing
- TensorFlow.js model export for web/browser use
- Sample price estimation and shipping cost integration (static logic)
- Waste category pricing table (demo values)
- Project documentation:
  - `README.md`
  - `requirements.txt`
  - Project directory structure
  - Licensing (MIT)

### 🛠️ Changed
- _N/A_ (first release)

### 🐛 Fixed
- _N/A_ (first release)

### 🗑️ Removed
- _N/A_ (first release)
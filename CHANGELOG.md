# 📦 CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [4.2.0] - 2025-04-26 (1)
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

---

## [4.1.0] - 2025-04-27
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

## [4.0.0] - 2025-04-26
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
- Jupyter Notebook for model training (`CapstoneDBS.ipynb`)

### 🛠️ Changed
- _N/A_

### 🐛 Fixed
- Improved accuracy of MobileNetV2 model for garbage classification up to 89%

### 🗑️ Removed
- Removed the InceptionV3 model from the model training file (`CapstoneDBS.ipynb`), visible in the preview file only (`preview_inception.ipynb`)

---

## [2.0.0] - 2025-04-22
> Created by **Bayu Ardiyansyah** and **Muhammad Rofi'ul Arham**

### 🎉 Added
- Added **InceptionV3** model for trash classification, as a comparison with previous models (MobileNetV2 and custom CNN)
- Support for InceptionV3 model training in `CapstoneDBS.ipynb`
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
- Jupyter Notebook for model training (`CapstoneDBS.ipynb`)
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
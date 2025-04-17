# GreenSort - Intelligent Waste Classification System

![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&style=flat-square)
![MIT License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)

GreenSort is a **Computer Vision-based** waste classification system designed to facilitate B2B waste recycling transactions. It allows users to classify waste materials from images, estimate their value, and calculate shipping costs based on distance and quantity. The system is trained on the [RealWaste dataset](https://archive.ics.uci.edu/dataset/908/realwaste) and supports nine waste categories: **Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation**.

## Features
- **Waste Classification**: Automatically identifies waste types from uploaded images using a deep learning model.
- **Price Estimation**: Calculates the estimated value of recyclable materials based on type and quantity.
- **Shipping Cost Integration**: Adjusts pricing based on distance and delivery options.
- **Testing Tool**: A GUI-based testing tool (`testing.py`) for classifying waste images and viewing results interactively.
- **Educational Add-On**: Provides consultation and education sessions for sustainable waste management (planned feature).

## Project Structure
```
GreenSort/
├── data/
│   └── realwaste/               # RealWaste dataset (downloaded separately)
├── models/
│   └── greensort_model.h5       # Trained model file
├── notebooks/
│   └── greensort_training.ipynb # Jupyter notebook for model training and evaluation
├── src/
│   └── utils.py                # Utility functions for data preprocessing and evaluation
├── testing.py                  # GUI-based testing tool for waste classification
├── requirements.txt            # Python dependencies
├── greensort_icon.ico          # Application icon (optional)
└── README.md                   # Project documentation
```

## Prerequisites
- Python 3.8 or higher
- pip for package management
- Access to the RealWaste dataset ([download here](https://archive.ics.uci.edu/dataset/908/realwaste))

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/greensort.git
   cd greensort
   ```

2. **Set up a virtual environment** (optional but recommended):
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
   numpy>=1.24.0
   matplotlib>=3.7.0
   pillow>=9.5.0
   tkinter
   ultralytics>=8.0.0  # Untuk YOLOv8
   ```

4. **Download the RealWaste dataset**:
   - Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/908/realwaste).
   - Extract and place the dataset in the `data/realwaste/` directory.

5. **Place the trained model**:
   - Ensure the `greensort_model.h5` file is in the `models/` directory. If not, train the model using the provided Jupyter notebook (`notebooks/greensort_training.ipynb`).

## Usage
### Training the Model
1. Open `notebooks/greensort_training.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the instructions to preprocess the RealWaste dataset and train the model (supports ResNet50 and YOLOv8).
3. Save the trained model to `models/greensort_model.h5` (for ResNet50) or export YOLOv8 weights.

### Testing the Model
The `testing.py` script provides a GUI for testing the waste classification model.

1. Run the testing tool:
   ```bash
   python testing.py
   ```

2. GUI Features:
   - **Select Image**: Choose a waste image (jpg, png, etc.) for classification.
   - **Set Quantity**: Input the waste quantity in kilograms using the spinbox.
   - **Classify Waste**: Click to classify the image and view results.
   - **Results Display**: Shows the top predicted waste type, confidence score, alternative predictions, and estimated price based on quantity.

3. Example Workflow:
   - Select an image of plastic waste.
   - Set quantity to 2.5 kg.
   - Click "Classify Waste" to see that the image is classified as "Plastic" with a confidence of 95% and an estimated value of Rp 10,000 (based on Rp 4,000/kg).

### Notes
- The GUI requires a trained model (`greensort_model.h5`). If not found, it will prompt to locate the model or create an untrained demo model (results will be random).
- For YOLOv8 integration, ensure the dataset is annotated with bounding boxes (use tools like LabelImg if needed).
- Ensure images are clear and contain identifiable waste materials for accurate classification.

## Waste Categories and Pricing
The system supports the following waste categories with their respective prices per kilogram (IDR):
| Category             | Price per kg (IDR) |
|----------------------|--------------------|
| Cardboard            | 2,000             |
| Food Organics        | 500               |
| Glass                | 1,500             |
| Metal                | 8,000             |
| Miscellaneous Trash  | 300               |
| Paper                | 2,500             |
| Plastic              | 4,000             |
| Textile Trash        | 1,000             |
| Vegetation           | 600               |

## Future Improvements
- Enhance YOLOv8 for multi-object detection to count multiple waste items in a single image.
- Add real-time shipping cost calculations using APIs (e.g., Google Maps for distance).
- Implement a web-based chatbot for consultation and waste management education.
- Improve model performance with transfer learning and hyperparameter tuning.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or support, contact [bayuardi30@outlook.com](mailto:bayuardi30@outlook.com) or open an issue on GitHub.

---

**© 2025 GreenSort Team. Built with sustainability in mind.**
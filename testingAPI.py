import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import traceback
import json
import requests
import logging
import socket
import time
import tensorflow as tf
import tensorflowjs as tfjs

# Setup logging
logging.basicConfig(
    filename='greensort_app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Define waste categories
CATEGORIES = ['Cardboard', 'Food_Organics', 'Glass', 'Metal',
              'Miscellaneous_Trash', 'Paper', 'Plastic',
              'Textile_Trash', 'Vegetation']

# Define pricing per kg for each category
PRICE_PER_KG = {
    'Cardboard': 2000,
    'Food_Organics': 500,
    'Glass': 1500,
    'Metal': 8000,
    'Miscellaneous_Trash': 300,
    'Paper': 2500,
    'Plastic': 4000,
    'Textile_Trash': 1000,
    'Vegetation': 600
}

class GreenSortApp:
    def __init__(self, root):
        """Initialize the application"""
        logging.info("Initializing GreenSortApp")
        self.root = root
        self.root.title("GreenSort - Waste Classification, Pricing, and Recommendation Tool")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

        try:
            self.root.iconbitmap("greensort_icon.ico")
        except:
            logging.warning("Failed to set window icon")

        # Initialize state
        self.classification_model_ready = False
        self.recommendation_model_ready = False
        self.error_shown = False
        self.progress_var = tk.DoubleVar()
        self.classification_model = None
        self.recommendation_model = None

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create UI elements
        self.create_ui()

        # TFJS model URLs
        self.classification_model_url = 'https://modelai14.s3.ap-southeast-2.amazonaws.com/tfjs_model/model.json'
        self.recommendation_model_url = 'https://my-model-bucket-greensort.s3.eu-north-1.amazonaws.com/Sistemrekomendasi/models/model.json'

        # Set initial status
        self.update_model_status("Loading models...", "orange")

        # Start checklist in a separate thread
        self.checklist_thread = threading.Thread(target=self.run_checklist)
        self.checklist_thread.daemon = True
        self.checklist_thread.start()

    def run_checklist(self):
        """Check availability of both TFJS models"""
        try:
            # Load classification model
            self.root.after(0, lambda: self.update_model_status("Loading classification model...", "orange"))
            self.load_classification_model()

            # Load recommendation model
            self.root.after(0, lambda: self.update_model_status("Loading recommendation model...", "orange"))
            self.load_recommendation_model()

            # Update final status
            if self.classification_model_ready and self.recommendation_model_ready:
                self.root.after(0, lambda: self.update_model_status(
                    "Both models loaded successfully", "green"))
            elif self.classification_model_ready:
                self.root.after(0, lambda: self.update_model_status(
                    "Classification model loaded, recommendation model failed", "red"))
            elif self.recommendation_model_ready:
                self.root.after(0, lambda: self.update_model_status(
                    "Recommendation model loaded, classification model failed", "red"))
            else:
                self.root.after(0, lambda: self.update_model_status(
                    "Failed to load both models", "red"))
        except Exception as e:
            error_msg = f"Checklist failed: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            self.show_error_once(error_msg)

    def verify_model_url(self, url):
        """Verify if model URL and weight files are accessible"""
        try:
            response = requests.head(url, timeout=10)
            if response.status_code != 200:
                return False, f"URL inaccessible, status: {response.status_code}"
            
            # Fetch model.json to check weightsManifest
            response = requests.get(url, timeout=10)
            model_json = response.json()
            weights_manifest = model_json.get('weightsManifest', [])
            if not weights_manifest or not weights_manifest[0].get('paths'):
                return False, "No weight files specified in weightsManifest"
            
            # Verify weight files
            base_url = url.rsplit('/', 1)[0]
            for manifest in weights_manifest:
                for path in manifest['paths']:
                    weight_url = f"{base_url}/{path}"
                    weight_response = requests.head(weight_url, timeout=10)
                    if weight_response.status_code != 200:
                        return False, f"Weight file {weight_url} inaccessible, status: {weight_response.status_code}"
            
            return True, "URL and weight files accessible"
        except Exception as e:
            return False, f"Verification failed: {str(e)}"

    def load_classification_model(self):
        """Load and convert classification TFJS model to TensorFlow"""
        # Verify URL and weights
        is_valid, msg = self.verify_model_url(self.classification_model_url)
        if not is_valid:
            logging.error(f"Classification model verification failed: {msg}")
            self.show_error_once(f"Cannot load classification model: {msg}")
            self.classification_model_ready = False
            return

        for attempt in range(5):
            try:
                self.classification_model = tfjs.converters.load_keras_model(self.classification_model_url)
                self.classification_model_ready = True
                logging.info("Classification TFJS model loaded successfully")
                return
            except Exception as e:
                error_msg = f"Classification model load attempt {attempt + 1} failed: {str(e)}\n{traceback.format_exc()}"
                logging.error(error_msg)
                if attempt == 4:
                    self.show_error_once(error_msg)
            if attempt < 4:
                time.sleep(2 ** attempt)  # Exponential backoff
        self.classification_model_ready = False
        logging.error("Failed to load classification TFJS model after retries")

    def load_recommendation_model(self):
        """Load and convert recommendation TFJS model to TensorFlow"""
        # Verify URL and weights
        is_valid, msg = self.verify_model_url(self.recommendation_model_url)
        if not is_valid:
            logging.error(f"Recommendation model verification failed: {msg}")
            self.show_error_once(f"Cannot load recommendation model: {msg}")
            self.recommendation_model_ready = False
            return

        for attempt in range(5):
            try:
                self.recommendation_model = tfjs.converters.load_keras_model(self.recommendation_model_url)
                self.recommendation_model_ready = True
                logging.info("Recommendation TFJS model loaded successfully")
                return
            except Exception as e:
                error_msg = f"Recommendation model load attempt {attempt + 1} failed: {str(e)}\n{traceback.format_exc()}"
                logging.error(error_msg)
                if attempt == 4:
                    self.show_error_once(error_msg)
            if attempt < 4:
                time.sleep(2 ** attempt)  # Exponential backoff
        self.recommendation_model_ready = False
        logging.error("Failed to load recommendation TFJS model after retries")

    def show_error_once(self, error_msg):
        """Display error message only once to prevent spam"""
        if not self.error_shown:
            self.error_shown = True
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.update_model_status("Error occurred", "red"))
            self.root.after(5000, lambda: setattr(self, 'error_shown', False))

    def get_recommendation(self, input_kategori, input_berat_kg):
        """Generate recycling recommendations using the TFJS recommendation model"""
        try:
            if not self.recommendation_model_ready:
                error_msg = "Recommendation model not loaded."
                logging.error(error_msg)
                self.show_error_once(error_msg)
                return {"message": error_msg}

            # Map category to Indonesian for consistency with model training
            category_mapping = {
                'Cardboard': 'Kardus',
                'Food_Organics': 'Bahan Organik Makanan',
                'Glass': 'Kaca',
                'Metal': 'Logam',
                'Miscellaneous_Trash': 'Sampah Lainnya',
                'Paper': 'Kertas',
                'Plastic': 'Plastik',
                'Textile_Trash': 'Sampah Tekstil',
                'Vegetation': 'Vegetasi'
            }
            mapped_category = category_mapping.get(input_kategori, input_kategori)

            # Prepare input as a string: "category weight_kg"
            input_text = f"{mapped_category} {input_berat_kg}"
            input_array = np.array([input_text])  # Shape [1,]

            # Perform inference
            recommendations = self.recommendation_model.predict(input_array)
            
            # Assuming the model outputs a string or list of recommendations
            if isinstance(recommendations, np.ndarray):
                recommendations = recommendations.tolist()
                if isinstance(recommendations[0], list):
                    recommendations = recommendations[0]
                recommendations = [str(rec) for rec in recommendations]
            
            return {
                "kategori": mapped_category,
                "berat_input_kg": input_berat_kg,
                "message": "",
                "rekomendasi": recommendations
            }

        except Exception as e:
            error_msg = f"Error generating recommendation: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            self.show_error_once(error_msg)
            return {"message": error_msg}

    def create_ui(self):
        """Create the UI elements"""
        logging.info("Creating UI elements")
        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(top_frame, text="GreenSort Waste Classification",
                  font=("Arial", 20, "bold")).pack(side=tk.LEFT)

        self.model_status = ttk.Label(top_frame, text="Loading...",
                                      font=("Arial", 10), foreground="orange")
        self.model_status.pack(side=tk.RIGHT, padx=10)

        notebook = ttk.Notebook(self.main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.classify_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.classify_tab, text="Classify Waste")

        about_tab = ttk.Frame(notebook, padding=10)
        notebook.add(about_tab, text="About")

        self.setup_classify_tab()
        self.setup_about_tab(about_tab)

        footer = ttk.Frame(self.main_frame)
        footer.pack(fill=tk.X, pady=(20, 0))
        ttk.Label(footer, text="© 2025 GreenSort. All rights reserved.").pack(side=tk.LEFT)
        ttk.Label(footer, text="v1.0.0").pack(side=tk.RIGHT)

        self.root.bind("<Configure>", self.on_window_resize)

    def on_window_resize(self, event):
        """Handle window resize events to redisplay image if needed"""
        if hasattr(self, 'last_width') and hasattr(self, 'last_height'):
            if (abs(self.last_width - self.root.winfo_width()) > 20 or
                abs(self.last_height - self.root.winfo_height()) > 20):
                if hasattr(self, 'selected_image_path') and self.selected_image_path:
                    self.display_image(self.selected_image_path)
                self.last_width = self.root.winfo_width()
                self.last_height = self.root.winfo_height()
        else:
            self.last_width = self.root.winfo_width()
            self.last_height = self.root.winfo_height()

    def setup_classify_tab(self):
        """Setup the classification tab UI"""
        logging.info("Setting up classify tab")
        left_panel = ttk.Frame(self.classify_tab)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        img_frame = ttk.LabelFrame(left_panel, text="Waste Image", padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.img_canvas = tk.Canvas(img_frame, bg="white", bd=0, highlightthickness=0, width=400, height=300)
        self.img_canvas.pack(fill=tk.BOTH, expand=True)

        self.img_canvas.update()

        self.no_img_label = ttk.Label(self.img_canvas, text="No image selected",
                                    font=("Arial", 12))
        self.no_img_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.select_btn = ttk.Button(control_frame, text="Select Image",
                                     command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))

        qty_frame = ttk.Frame(control_frame)
        qty_frame.pack(side=tk.RIGHT)

        ttk.Label(qty_frame, text="Quantity (kg):").pack(side=tk.LEFT, padx=(0, 5))

        self.quantity_var = tk.StringVar(value="1.0")
        self.quantity_spinbox = ttk.Spinbox(
            qty_frame,
            from_=0.1,
            to=100.0,
            textvariable=self.quantity_var,
            width=5,
            increment=0.1
        )
        self.quantity_spinbox.pack(side=tk.LEFT)

        self.classify_btn = ttk.Button(left_panel, text="Classify Waste",
                                      command=self.classify_image, state=tk.DISABLED)
        self.classify_btn.pack(fill=tk.X)

        self.progress = ttk.Progressbar(left_panel, variable=self.progress_var,
                                       mode='determinate')
        self.progress.pack(fill=tk.X, pady=(10, 0))

        right_panel = ttk.LabelFrame(self.classify_tab, text="Classification Results", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.predictions_frame = ttk.Frame(right_panel)
        self.predictions_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(self.predictions_frame, text="No results available",
                 font=("Arial", 12, "italic")).pack(pady=20)

        self.price_frame = ttk.LabelFrame(right_panel, text="Price Calculation", padding=10)
        self.price_frame.pack(fill=tk.BOTH, expand=True)

        self.recommendation_frame = ttk.LabelFrame(right_panel, text="Recycling Recommendations", padding=10)
        self.recommendation_frame.pack(fill=tk.BOTH, expand=True)

        self.selected_image_path = None
        self.tk_image = None
        self.original_image = None
        self.processed_image = None
        self.current_quantity = 1.0

    def setup_about_tab(self, tab):
        """Setup the about tab with project information"""
        logging.info("Setting up about tab")
        about_text = tk.Text(tab, wrap=tk.WORD, font=("Arial", 11),
                            height=20, width=70, padx=10, pady=10)
        about_text.pack(fill=tk.BOTH, expand=True)

        about_content = """
        # GreenSort - Waste Classification, Pricing, and Recommendation System

        GreenSort is an intelligent waste management system that uses Computer Vision to identify waste materials,
        predict their market value, and provide recycling recommendations based on category and weight.

        ## Features

        - Classify waste into 9 different categories
        - Estimate the value of recyclable materials
        - Provide recycling recommendations
        - Optimize waste management and recycling processes

        ## Waste Categories

        1. Cardboard - Rp 2,000/kg
        2. Food Organics - Rp 500/kg
        3. Glass - Rp 1,500/kg
        4. Metal - Rp 8,000/kg
        5. Miscellaneous Trash - Rp 300/kg
        6. Paper - Rp 2,500/kg
        7. Plastic - Rp 4,000/kg
        8. Textile Trash - Rp 1,000/kg
        9. Vegetation - Rp 600/kg

        ## How to Use

        1. Select an image of waste material using the "Select Image" button
        2. Set the quantity in kilograms
        3. Click "Classify Waste" to analyze the image, predict the price, and get recommendations
        4. View the classification, pricing, and recycling recommendations

        ## Dataset

        This application is trained on the RealWaste dataset from the UCI Machine Learning Repository:
        https://archive.ics.uci.edu/dataset/908/realwaste

        The recommendation system uses a custom TFJS model for recycling suggestions.
        """

        about_text.insert(tk.END, about_content)
        about_text.config(state=tk.DISABLED)

    def update_model_status(self, status_text, color):
        """Update the model status text and color"""
        try:
            self.model_status.config(text=status_text, foreground=color)
            if color == "green" and self.classification_model_ready and self.recommendation_model_ready:
                self.classify_btn.config(state=tk.NORMAL)
            else:
                self.classify_btn.config(state=tk.DISABLED)
        except tk.TclError as e:
            logging.error(f"Failed to update model status: {str(e)}")

    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select Waste Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.selected_image_path = file_path
            try:
                test_image = Image.open(file_path)
                test_image.verify()
                self.display_image(file_path)
                if self.classification_model_ready and self.recommendation_model_ready:
                    self.classify_btn.config(state=tk.NORMAL)
            except Exception as e:
                error_msg = f"The selected file is not a valid image: {str(e)}"
                logging.error(error_msg)
                self.show_error_once(error_msg)
                self.selected_image_path = None

    def display_image(self, img_path):
        """Display the selected image on the canvas"""
        try:
            self.root.update_idletasks()
            self.original_image = Image.open(img_path)
            canvas_width = self.img_canvas.winfo_width()
            canvas_height = self.img_canvas.winfo_height()

            if canvas_width < 100:
                canvas_width = 400
            if canvas_height < 100:
                canvas_height = 300

            img_width, img_height = self.original_image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            try:
                resized_img = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except AttributeError:
                resized_img = self.original_image.resize((new_width, new_height), Image.LANCZOS)

            self.tk_image = ImageTk.PhotoImage(resized_img)
            x_pos = canvas_width // 2
            y_pos = canvas_height // 2
            self.img_canvas.delete("all")
            self.img_canvas.create_image(x_pos, y_pos, image=self.tk_image, anchor=tk.CENTER, tags="image")
            self.no_img_label.place_forget()

        except Exception as e:
            error_msg = f"Could not display image: {str(e)}"
            logging.error(error_msg)
            self.show_error_once(error_msg)

    def classify_image(self):
        """Classify the selected waste image using TFJS model"""
        if not self.selected_image_path:
            self.show_error_once("Please select an image")
            return

        if not self.classification_model_ready:
            self.show_error_once("Classification model is not available")
            return

        if not self.recommendation_model_ready:
            self.show_error_once("Recommendation model is not available")
            return

        self.classify_btn.config(state=tk.DISABLED)
        self.progress_var.set(10)

        try:
            quantity = float(self.quantity_spinbox.get())
            if quantity <= 0:
                quantity = 1.0
        except ValueError:
            quantity = 1.0

        thread = threading.Thread(target=self.process_classification, args=(quantity,))
        thread.daemon = True
        thread.start()

    def process_classification(self, quantity):
        """Process the image classification using TFJS model"""
        try:
            self.root.after(0, lambda: self.progress_var.set(20))

            # Preprocess image
            img = Image.open(self.selected_image_path)
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            self.processed_image = img_array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            self.root.after(0, lambda: self.progress_var.set(60))

            # Perform classification inference
            predictions = self.classification_model.predict(img_array)
            predictions = predictions[0]  # Remove batch dimension

            self.root.after(0, lambda: self.progress_var.set(80))

            predicted_class_index = np.argmax(predictions)
            predicted_class = CATEGORIES[predicted_class_index]
            confidence = predictions[predicted_class_index] * 100

            top_indices = predictions.argsort()[-3:][::-1]
            top_predictions = [(CATEGORIES[i], predictions[i] * 100) for i in top_indices]

            # Get recommendations using the recommendation model
            recommendations = self.get_recommendation(predicted_class, quantity)

            self.root.after(0, lambda: self.progress_var.set(90))

            self.root.after(0, lambda: self.display_results(
                predicted_class, confidence, top_predictions, quantity, recommendations))

        except Exception as e:
            error_msg = f"Classification failed: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            self.show_error_once(error_msg)
        finally:
            self.root.after(0, lambda: self.progress_var.set(0))
            self.root.after(0, lambda: self.classify_btn.config(state=tk.NORMAL if self.classification_model_ready and self.recommendation_model_ready else tk.DISABLED))

    def display_results(self, predicted_class, confidence, top_predictions, quantity, recommendations):
        """Display the classification, pricing, and recommendation results"""
        self.classify_btn.config(state=tk.NORMAL if self.classification_model_ready and self.recommendation_model_ready else tk.DISABLED)

        for widget in self.predictions_frame.winfo_children():
            widget.destroy()

        for widget in self.price_frame.winfo_children():
            widget.destroy()

        for widget in self.recommendation_frame.winfo_children():
            widget.destroy()

        prediction_header = ttk.Frame(self.predictions_frame)
        prediction_header.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(prediction_header, text="Top Prediction:",
                 font=("Arial", 11)).pack(side=tk.LEFT)

        main_pred_frame = ttk.Frame(self.predictions_frame, padding=10)
        main_pred_frame.pack(fill=tk.X, pady=(0, 15))
        main_pred_frame.configure(style="Card.TFrame")

        color_indicators = {
            "Cardboard": "#cd853f",
            "Food_Organics": "#8fbc8f",
            "Glass": "#add8e6",
            "Metal": "#c0c0c0",
            "Miscellaneous_Trash": "#808080",
            "Paper": "#f5f5dc",
            "Plastic": "#87cefa",
            "Textile_Trash": "#db7093",
            "Vegetation": "#32cd32"
        }

        indicator_color = color_indicators.get(predicted_class, "#dcdcdc")
        indicator = tk.Frame(main_pred_frame, width=15, bg=indicator_color)
        indicator.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        pred_info = ttk.Frame(main_pred_frame)
        pred_info.pack(fill=tk.BOTH, expand=True)

        ttk.Label(pred_info, text=predicted_class,
                 font=("Arial", 14, "bold")).pack(anchor=tk.W)

        ttk.Label(pred_info, text=f"Confidence: {confidence:.2f}%",
                 font=("Arial", 11)).pack(anchor=tk.W)

        ttk.Label(self.predictions_frame, text="Alternative Predictions:",
                 font=("Arial", 11)).pack(anchor=tk.W, pady=(0, 5))

        alt_frame = ttk.Frame(self.predictions_frame)
        alt_frame.pack(fill=tk.X)

        for i, (class_name, prob) in enumerate(top_predictions[1:], 1):
            alt_color = color_indicators.get(class_name, "#dcdcdc")
            alt_item = ttk.Frame(alt_frame, padding=5)
            alt_item.pack(fill=tk.X, pady=2)

            tk.Frame(alt_item, width=10, bg=alt_color).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

            ttk.Label(alt_item, text=f"{class_name}", font=("Arial", 11)).pack(side=tk.LEFT)
            ttk.Label(alt_item, text=f"{prob:.2f}%", font=("Arial", 10)).pack(side=tk.RIGHT)

        price_per_unit = PRICE_PER_KG.get(predicted_class, 0)
        total_price = price_per_unit * quantity

        price_table = ttk.Frame(self.price_frame)
        price_table.pack(fill=tk.X, pady=10)

        row1 = ttk.Frame(price_table)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Waste Type:", font=("Arial", 11)).pack(side=tk.LEFT)
        ttk.Label(row1, text=predicted_class, font=("Arial", 11, "bold")).pack(side=tk.RIGHT)

        row2 = ttk.Frame(price_table)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Price per kg:", font=("Arial", 11)).pack(side=tk.LEFT)
        ttk.Label(row2, text=f"Rp {price_per_unit:,}", font=("Arial", 11)).pack(side=tk.RIGHT)

        row3 = ttk.Frame(price_table)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Quantity:", font=("Arial", 11)).pack(side=tk.LEFT)
        ttk.Label(row3, text=f"{quantity:.2f} kg", font=("Arial", 11)).pack(side=tk.RIGHT)

        ttk.Separator(self.price_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        row4 = ttk.Frame(price_table)
        row4.pack(fill=tk.X, pady=5)
        ttk.Label(row4, text="TOTAL VALUE:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        ttk.Label(row4, text=f"Rp {total_price:,.0f}",
                 font=("Arial", 14, "bold"), foreground="#27ae60").pack(side=tk.RIGHT)

        rec_frame = ttk.Frame(self.recommendation_frame)
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        if recommendations.get("message") and not recommendations.get("rekomendasi"):
            ttk.Label(rec_frame, text=recommendations["message"],
                     font=("Arial", 11, "italic"), wraplength=300).pack(anchor=tk.W)
        else:
            if recommendations.get("message"):
                ttk.Label(rec_frame, text=recommendations["message"],
                         font=("Arial", 11), wraplength=300).pack(anchor=tk.W, pady=(0, 5))

            ttk.Label(rec_frame, text="Recommendations:",
                     font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 5))

            for i, rec in enumerate(recommendations.get("rekomendasi", []), 1):
                ttk.Label(rec_frame, text=f"{i}. {rec}",
                         font=("Arial", 11), wraplength=300).pack(anchor=tk.W, pady=2)

        if self.processed_image is not None:
            self.root.after(0, self.show_processed_image)

    def show_processed_image(self):
        """Show the processed image used for classification"""
        try:
            processed_window = tk.Toplevel(self.root)
            processed_window.title("Processed Image")
            processed_window.geometry("300x350")

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(self.processed_image)
            ax.set_title("Processed Image (224x224)")
            ax.axis('off')

            canvas = FigureCanvasTkAgg(fig, master=processed_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            ttk.Label(processed_window,
                     text="This is the processed image used by the model for classification.",
                     wraplength=280).pack(pady=10)

            ttk.Button(processed_window, text="Close", command=processed_window.destroy).pack(pady=10)
        except Exception as e:
            logging.error(f"Failed to show processed image: {str(e)}")

def main():
    """Main function to run the application"""
    print("Starting GreenSort application...")
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("Card.TFrame", background="#ffffff", relief="ridge", borderwidth=1)
        style.configure("TButton", font=("Arial", 11))
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 11))
        style.configure("TLabelframe", background="#f0f0f0", font=("Arial", 11))
        style.configure("TLabelframe.Label", background="#f0f0f0", font=("Arial", 11, "bold"))
    except Exception as e:
        print(f"Warning: Could not set all theme settings: {str(e)}")

    print("Creating Tkinter root...")
    root = tk.Tk()
    print("Initializing GreenSortApp...")
    app = GreenSortApp(root)
    print("Starting mainloop...")
    root.mainloop()

if __name__ == "__main__":
    main()
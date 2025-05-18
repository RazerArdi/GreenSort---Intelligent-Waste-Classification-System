"""
GreenSort - Waste Classification, Pricing and Recommendation Tool
This script provides a GUI for testing the waste classification pricing, and recommendation model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import traceback
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import logging

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

# Define pricing per kg for each category (used as Harga_Pasar)
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
        self.root = root
        self.root.title("GreenSort - Waste Classification, Pricing, and Recommendation Tool")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")
        
        # Attempt to set window icon
        try:
            self.root.iconbitmap("greensort_icon.ico")
        except:
            pass
            
        # Initialize models and preprocessing
        self.classification_model = None
        self.recommendation_model = None
        self.recommendation_df = None
        self.encoder = None
        self.scaler = None
        
        # Create a progress indicator
        self.progress_var = tk.DoubleVar()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create and organize the UI elements
        self.create_ui()
        
        # Load classification model in a separate thread
        self.load_class_model_thread = threading.Thread(target=self.load_classification_model)
        self.load_class_model_thread.daemon = True
        self.load_class_model_thread.start()
        
        # Load recommendation model and dataset
        self.load_recommendation_model()
    
    def load_classification_model(self):
        """Load the waste classification model"""
        try:
            model_path = 'models/ComputerVision/greensort_model.h5'
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                self.root.after(0, lambda: self.update_model_status("Model loaded successfully", "green"))
            else:
                self.root.after(0, lambda: self.update_model_status("Model file not found", "red"))
                self.root.after(1000, self.locate_model)
        except Exception as e:
            self.root.after(0, lambda: self.update_model_status(f"Error loading model: {str(e)}", "red"))
    
    def locate_model(self):
        """Ask user to locate the model file"""
        response = messagebox.askyesno("Model Not Found", 
                                      "The model file 'greensort_model.h5' was not found.\n\n"
                                      "Do you want to locate the model file manually?")
        if response:
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
            )
            if model_path:
                try:
                    self.model = load_model(model_path)
                    self.update_model_status("Model loaded successfully", "green")
                except Exception as e:
                    self.update_model_status(f"Error loading model: {str(e)}", "red")
        else:
            self.create_test_model()
    
    def create_test_model(self):
        """Create a simple test model for demonstration"""
        try:
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras import layers, models
            
            base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
            
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(CATEGORIES), activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            self.root.after(0, lambda: self.update_model_status("Demo model created (untrained)", "orange"))
            self.root.after(0, lambda: messagebox.showwarning("Demo Mode", 
                                                             "Running in demo mode with an untrained model.\n"
                                                             "Classifications will be random and not meaningful."))
        except Exception as e:
            self.root.after(0, lambda: self.update_model_status(f"Error creating demo model: {str(e)}", "red"))
            
    def load_recommendation_model(self):
        """Load the recommendation model and dataset with robust error handling"""
        try:
            # Define path
            model_path = 'models/Sistemrekomendasi/models/recycling_recommendation_model.h5'
            dataset_path = 'models/Sistemrekomendasi/dataset.json'

            # Load model
            if os.path.exists(model_path):
                try:
                    self.recommendation_model = load_model(
                        model_path,
                        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
                    )
                    logging.info(f"Recommendation model loaded from {model_path}")
                except Exception as e:
                    logging.error(f"Failed to load model from {model_path}: {str(e)}")
                    self.root.after(0, lambda: self.update_model_status("Failed to load recommendation model", "orange"))
                    self.create_fallback_recommendation_system()
                    return
            else:
                logging.error("Recommendation model file not found")
                self.root.after(0, lambda: self.update_model_status("Recommendation model not found", "orange"))
                self.create_fallback_recommendation_system()
                return

            # Load dataset
            if os.path.exists(dataset_path):
                try:
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.recommendation_df = pd.DataFrame(data)
                    logging.info(f"Recommendation dataset loaded from {dataset_path}")
                except Exception as e:
                    logging.error(f"Failed to load dataset from {dataset_path}: {str(e)}")
                    self.root.after(0, lambda: self.update_model_status("Failed to load recommendation dataset", "orange"))
                    self.create_fallback_recommendation_system()
                    return
            else:
                logging.error("Recommendation dataset file not found")
                self.root.after(0, lambda: self.update_model_status("Recommendation dataset not found", "orange"))
                self.create_fallback_recommendation_system()
                return

            # Verify dataset structure
            required_columns = ['kategori', 'rekomendasi', 'berat_min_kg', 'berat_max_kg']
            if not all(col in self.recommendation_df.columns for col in required_columns):
                error_msg = f"Dataset missing required columns: {required_columns}"
                logging.error(error_msg)
                self.root.after(0, lambda: self.update_model_status(error_msg, "orange"))
                self.create_fallback_recommendation_system()
                return

            # Add combined_rekomendasi column if it doesn't exist
            if 'combined_rekomendasi' not in self.recommendation_df.columns:
                self.recommendation_df['combined_rekomendasi'] = self.recommendation_df['rekomendasi'].apply(
                    lambda x: " ".join(x) if isinstance(x, list) else str(x)
                )

            # Success
            self.root.after(0, lambda: self.update_model_status("Models loaded successfully", "green"))

        except Exception as e:
            error_msg = f"Error in recommendation system setup: {str(e)}"
            logging.error(error_msg)
            self.root.after(0, lambda: self.update_model_status(error_msg, "orange"))
            self.create_fallback_recommendation_system()


    def get_recommendation(self, input_kategori, input_berat_kg, tolerance=0.2):
        try:
            # Mapping model categories to datasets
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

            # Take the row that matches the category
            matching_rows = self.recommendation_df[self.recommendation_df['kategori'] == mapped_category]
            if matching_rows.empty:
                return {"message": f"Kategori '{mapped_category}' tidak ditemukan dalam dataset."}

            # Look for lines in the appropriate weight range
            weight_match = matching_rows[
                (matching_rows['berat_min_kg'] <= input_berat_kg) &
                (input_berat_kg <= matching_rows['berat_max_kg'])
            ]

            if not weight_match.empty:
                best_match_idx = weight_match.index[0]
                message = ""
            else:
                best_match_idx = matching_rows.index[0]
                berat_min_kg = matching_rows.loc[best_match_idx]['berat_min_kg']
                berat_max_kg = matching_rows.loc[best_match_idx]['berat_max_kg']
                message = (
                    f"Berat sampah Anda ({input_berat_kg} kg) sedikit tidak sesuai dengan rekomendasi "
                    f"untuk kategori ini ({berat_min_kg} kg - {berat_max_kg} kg).\n\n"
                    "Namun, berikut adalah beberapa rekomendasi yang bisa diterapkan:\n"
                )

            best_kategori = matching_rows.loc[best_match_idx]['kategori']
            rekomendasi_list = matching_rows.loc[best_match_idx]['rekomendasi']
            berat_min_kg = matching_rows.loc[best_match_idx]['berat_min_kg']
            berat_max_kg = matching_rows.loc[best_match_idx]['berat_max_kg']

            return {
                "kategori": best_kategori,
                "berat_input_kg": input_berat_kg,
                "berat_min_kg": berat_min_kg,
                "berat_max_kg": berat_max_kg,
                "message": message,
                "rekomendasi": rekomendasi_list
            }

        except Exception as e:
            error_msg = f"Error generating recommendation: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            return {"message": error_msg}    
        
    def create_ui(self):
        """Create the UI elements"""
        # Top section for title and model status
        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(top_frame, text="GreenSort Waste Classification", 
                  font=("Arial", 20, "bold")).pack(side=tk.LEFT)
        
        self.model_status = ttk.Label(top_frame, text="Loading model...", 
                                      font=("Arial", 10), foreground="orange")
        self.model_status.pack(side=tk.RIGHT, padx=10)
        
        # Create a notebook for tabs
        notebook = ttk.Notebook(self.main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Classification tab
        self.classify_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.classify_tab, text="Classify Waste")
        
        # About tab
        about_tab = ttk.Frame(notebook, padding=10)
        notebook.add(about_tab, text="About")
        
        # Setup the classification tab
        self.setup_classify_tab()
        
        # Setup the about tab
        self.setup_about_tab(about_tab)
        
        # Setup the footer
        footer = ttk.Frame(self.main_frame)
        footer.pack(fill=tk.X, pady=(20, 0))
        ttk.Label(footer, text="© 2025 GreenSort. All rights reserved.").pack(side=tk.LEFT)
        ttk.Label(footer, text="v1.0.0").pack(side=tk.RIGHT)
        
        # Bind the resize event to trigger image redisplay
        self.root.bind("<Configure>", self.on_window_resize)
        
    def on_window_resize(self, event):
        """Handle window resize events to redisplay image if needed"""
        # Only process if it's a substantial resize
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
        # Left panel for image and selection
        left_panel = ttk.Frame(self.classify_tab)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
            
        # Frame for image display
        img_frame = ttk.LabelFrame(left_panel, text="Waste Image", padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
        # Canvas for displaying the image with fixed size
        self.img_canvas = tk.Canvas(img_frame, bg="white", bd=0, highlightthickness=0, width=400, height=300)
        self.img_canvas.pack(fill=tk.BOTH, expand=True)
            
        # Make sure the canvas has some initial size
        self.img_canvas.update()
            
        # Label to show when no image is selected
        self.no_img_label = ttk.Label(self.img_canvas, text="No image selected", 
                                    font=("Arial", 12))
        self.no_img_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                
        # Control frame for buttons and quantity
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Button to select image
        self.select_btn = ttk.Button(control_frame, text="Select Image", 
                                     command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Quantity frame
        qty_frame = ttk.Frame(control_frame)
        qty_frame.pack(side=tk.RIGHT)
        
        ttk.Label(qty_frame, text="Quantity (kg):").pack(side=tk.LEFT, padx=(0, 5))
        
        # Use StringVar instead of DoubleVar for better control
        self.quantity_var = tk.StringVar(value="1.0")
        
        # Create the spinbox with more direct control
        self.quantity_spinbox = ttk.Spinbox(
            qty_frame, 
            from_=0.1, 
            to=100.0, 
            textvariable=self.quantity_var, 
            width=5, 
            increment=0.1
        )
        self.quantity_spinbox.pack(side=tk.LEFT)
        
        # Button to classify
        self.classify_btn = ttk.Button(left_panel, text="Classify Waste", 
                                      command=self.classify_image, state=tk.DISABLED)
        self.classify_btn.pack(fill=tk.X)
        
        # Progress bar
        self.progress = ttk.Progressbar(left_panel, variable=self.progress_var, 
                                       mode='determinate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        # Right panel for results
        right_panel = ttk.LabelFrame(self.classify_tab, text="Classification Results", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Top predictions frame
        self.predictions_frame = ttk.Frame(right_panel)
        self.predictions_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.predictions_frame, text="No results available",
                 font=("Arial", 12, "italic")).pack(pady=20)
        
        # Price calculation frame
        self.price_frame = ttk.LabelFrame(right_panel, text="Price Calculation", padding=10)
        self.price_frame.pack(fill=tk.BOTH, expand=True)
        
        # Recomendation
        self.recommendation_frame = ttk.LabelFrame(right_panel, text="Recycling Recommendations", padding=10)
        self.recommendation_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize variables to store the selected image
        self.selected_image_path = None
        self.tk_image = None
        self.original_image = None
        self.processed_image = None
        
        # Class variable to store current quantity safely between threads
        self.current_quantity = 1.0
        
    def validate_quantity(self, value):
        """Validate the quantity input to ensure it's a positive float"""
        try:
            if value:
                val = float(value)
                return val > 0
            return True  # Allow empty input temporarily
        except ValueError:
            return False
        
    def setup_about_tab(self, tab):
        """Setup the about tab with project information"""
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
        - Calculate total price including delivery costs
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
        2. Set the quantity in kilograms and delivery distance in kilometers
        3. Click "Classify Waste" to analyze the image, predict the price, and get recommendations
        4. View the classification, pricing, and recycling recommendations
        
        ## Dataset
        
        This application is trained on the RealWaste dataset from the UCI Machine Learning Repository:
        https://archive.ics.uci.edu/dataset/908/realwaste
        
        The recommendation system uses a custom dataset for recycling suggestions.
        """
        
        about_text.insert(tk.END, about_content)
        about_text.config(state=tk.DISABLED)
    
    def update_model_status(self, status_text, color):
        """Update the model status text and color"""
        self.model_status.config(text=status_text, foreground=color)
        if (color == "green" and self.classification_model and 
            self.recommendation_model and self.recommendation_df is not None):
            self.classify_btn.config(state=tk.NORMAL)
    
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
                # Validate image before attempting to display
                test_image = Image.open(file_path)
                test_image.verify()  # Verify it's a valid image
                self.display_image(file_path)
                if self.model:
                    self.classify_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Invalid Image", f"The selected file is not a valid image: {str(e)}")
                self.selected_image_path = None
    
    def display_image(self, img_path):
        """Display the selected image on the canvas"""
        try:
            # Force update before getting dimensions
            self.root.update_idletasks()
            
            # Load the original image and keep a reference
            try:
                self.original_image = Image.open(img_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image file: {str(e)}")
                return
            
            # Get canvas dimensions
            canvas_width = self.img_canvas.winfo_width()
            canvas_height = self.img_canvas.winfo_height()
            
            # Ensure minimum canvas dimensions
            if canvas_width < 100:
                canvas_width = 400
            if canvas_height < 100:
                canvas_height = 300
            
            # Calculate the new size to fit in canvas while preserving aspect ratio
            img_width, img_height = self.original_image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Resize the image using LANCZOS resampling
            try:
                resized_img = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except AttributeError:
                # For older PIL versions
                resized_img = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage for display
            try:
                self.tk_image = ImageTk.PhotoImage(resized_img)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to convert image: {str(e)}")
                return
            
            # Clear canvas and display image
            self.img_canvas.delete("all")
            # Center the image
            x_pos = canvas_width // 2
            y_pos = canvas_height // 2
            self.img_canvas.create_image(x_pos, y_pos, image=self.tk_image, anchor=tk.CENTER, tags="image")
            
            # Hide the "No image selected" label
            self.no_img_label.place_forget()
            
            # Print debug info
            print(f"Image displayed: {img_path}")
            print(f"Canvas size: {canvas_width}x{canvas_height}")
            print(f"Original size: {img_width}x{img_height}")
            print(f"Display size: {new_width}x{new_height}")
            
        except Exception as e:
            error_msg = f"Could not display image: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Log to console
            messagebox.showerror("Error", error_msg)
    
    def classify_image(self):
        """Classify the selected waste image"""
        if not self.selected_image_path or not self.model:
            messagebox.showwarning("Warning", "Please select an image and ensure model is loaded")
            return
        
        # Disable classify button and show progress
        self.classify_btn.config(state=tk.DISABLED)
        self.progress_var.set(10)
        
        # Get the quantity directly from the spinbox widget
        try:
            # Force an update to ensure we get the current value
            self.root.update_idletasks()
            quantity_str = self.quantity_spinbox.get()
            quantity = float(quantity_str)
            print(f"Spinbox direct value: {quantity_str}, Parsed quantity: {quantity}")  # Debug
            
            # Only update if valid
            if quantity > 0:
                self.current_quantity = quantity
            else:
                self.current_quantity = 1.0
                print("Invalid quantity (<=0), using default")
        except ValueError:
            self.current_quantity = 1.0
            print(f"Failed to parse quantity from '{quantity_str}', using default 1.0")
        
        print(f"Final quantity for classification: {self.current_quantity}")  # Debug
        
        # Store a copy for the thread to use
        quantity = self.current_quantity
        
        # Start classification in a separate thread with a copy of the quantity
        thread = threading.Thread(target=self.process_classification, args=(quantity,))
        thread.daemon = True
        thread.start()
    
    def process_classification(self, quantity):
        """Process the image classification, price prediction, and recommendations"""
        try:
            print(f"Process classification started with quantity: {quantity}")  # Debug
            
            # Preprocess the image
            self.root.after(0, lambda: self.progress_var.set(20))
            
            img = image.load_img(self.selected_image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            self.processed_image = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            self.root.after(0, lambda: self.progress_var.set(40))
            
            # Make prediction
            prediction = self.model.predict(img_array)
            
            self.root.after(0, lambda: self.progress_var.set(70))
            
            # Get predicted class and confidence
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = CATEGORIES[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100
            
            # Get top 3 predictions
            top_indices = prediction[0].argsort()[-3:][::-1]
            top_predictions = [(CATEGORIES[i], prediction[0][i] * 100) for i in top_indices]
            
            self.root.after(0, lambda: self.progress_var.set(90))
            
            # Pass quantity to display_results using lambda to ensure it's captured correctly
            final_quantity = quantity  # Make a copy to be sure
            print(f"Quantity before display_results: {final_quantity}")  # Debug
            
            # Use lambda to capture the current value of final_quantity
            self.root.after(0, lambda q=final_quantity: self.display_results(
                predicted_class, confidence, top_predictions, q))
            
            recommendations = self.get_recommendation(predicted_class, quantity)
            
            self.root.after(0, lambda: self.progress_var.set(90))
            
            self.root.after(0, lambda: self.display_results(
                predicted_class, confidence, top_predictions, quantity, recommendations))
            
        except Exception as e:
            error_msg = f"Classification failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Log to console for debugging
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.classify_btn.config(state=tk.NORMAL))
        finally:
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def display_results(self, predicted_class, confidence, top_predictions, quantity, recommendations):
        """Display the classification, pricing, and recommendation results"""
        print(f"Displaying results with quantity: {quantity}")  # Debug
        
        self.classify_btn.config(state=tk.NORMAL)
        
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

def main():
    """Main function to run the application"""
    try:
        # Configure Tkinter styles
        style = ttk.Style()
        
        if "clam" in style.theme_names():
            style.theme_use("clam")
            
        # Create custom styles for widgets
        style.configure("TFrame", background="#f0f0f0")
        style.configure("Card.TFrame", background="#ffffff", relief="ridge", borderwidth=1)
        style.configure("TButton", font=("Arial", 11))
        style.configure("TLabel", background="#f0f0f0", font=("Arial", 11))
        style.configure("TLabelframe", background="#f0f0f0", font=("Arial", 11))
        style.configure("TLabelframe.Label", background="#f0f0f0", font=("Arial", 11, "bold"))
    except Exception as e:
        print(f"Warning: Could not set all theme settings: {str(e)}")
    
    # Create and run the application
    root = tk.Tk()
    app = GreenSortApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
import tkinter as tk  # GUI toolkit for creating windows, buttons, and other UI elements
from tkinter import ttk  # For progress bar and styled widgets
import sv_ttk  # For themed Tkinter widgets
from tkinter import filedialog, messagebox  # File dialogs and messagebox for GUI alerts and file browsing
import torch  # PyTorch for machine learning and deep learning tasks
import torch.nn as nn  # PyTorch module for creating neural networks
import torchvision.transforms as transforms  # Image transformation utilities from torchvision
import torchvision.models as models  # Pre-trained models from torchvision
from torch.utils.data import DataLoader  # Utility for loading datasets and handling batches
from PIL import Image  # Python Imaging Library (PIL) for image processing
from PIL import ImageTk  # For displaying images in Tkinter
import os  # OS-related utilities like file and directory operations
import shutil  # File operations such as copying or moving files
import logging  # Logging library for outputting debug and error information
import csv  # CSV file reading and writing functionality
import yaml  # YAML file parsing for configuration settings
from concurrent.futures import ThreadPoolExecutor  # For concurrent and parallel task execution
import matplotlib.pyplot as plt  # Matplotlib for creating plots and graphs
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embeds Matplotlib graphs in Tkinter GUIs

# Load configuration settings from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set up logging for the program, with messages written to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the device (GPU or CPU) to use for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Define image transformations based on the YAML configuration
# Resize the image, convert it to grayscale, and normalize it
transform_test = transforms.Compose([
    transforms.Resize(config['image_transform']['resize']),
    transforms.Grayscale(num_output_channels=config['image_transform']['grayscale']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['image_transform']['normalize_mean'], std=config['image_transform']['normalize_std'])
])

class CustomResNet50(nn.Module):
    """
    Custom ResNet50 model for binary classification (crack/nocrack).

    The model is built using a pre-trained ResNet50 backbone, 
    with the fully connected (fc) layer modified for two-class classification.
    
    Attributes:
        model: The underlying ResNet50 architecture with modified final layers.
    """
    def __init__(self, weights='IMAGENET1K_V2'):
        super(CustomResNet50, self).__init__()
        # Load pre-trained ResNet50 model and modify the fully connected layer
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.7),  # Regularization using dropout
            nn.Linear(self.model.fc.in_features, 2)  # Output layer for binary classification
        )

    def forward(self, x):
        """Performs the forward pass of the model."""
        return self.model(x)

class SolderJointDataset(torch.utils.data.Dataset):
    """
    Custom dataset class to load solder joint images from a folder.

    This class allows for loading images, applying transformations, 
    and retrieving them for prediction.
    
    Attributes:
        image_folder: The directory containing the image files.
        image_paths: List of paths to all PNG images in the folder.
        transform: Image transformations to be applied.
    """
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads an image and applies transformations."""
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")  # Convert to RGB format
        except Exception as e:
            logging.error(f"Failed to open image {img_path}: {e}")
            raise
        if self.transform:
            image = self.transform(image)  # Apply transformations
        return image, img_path

def load_model(model_path):
    """
    Loads a saved model from the specified file path.

    Args:
        model_path: Path to the PyTorch model file (.pth).
    
    Returns:
        A loaded CustomResNet50 model ready for inference.
    """
    logging.info(f"Loading model from: {model_path}")
    try:
        # Load the pre-trained model and move it to the appropriate device (CPU/GPU)
        model = CustomResNet50().to(device)
        state_dict = torch.load(model_path, map_location=device)  # Load model weights
        model.load_state_dict(state_dict)
        model.eval()  # Set model to evaluation mode (important for inference)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def predict_images(model, dataloader, confidence_threshold):
    """
    Uses the loaded model to predict the class of each image in the dataloader.
    
    Args:
        model: The trained CustomResNet50 model.
        dataloader: PyTorch DataLoader to load the images in batches.
        confidence_threshold: The minimum confidence score required to classify an image.
    
    Returns:
        List of predictions, each containing image path, predicted class label, and confidence score.
    """
    class_names = ['crack', 'nocrack']  # Class names for binary classification
    predictions = []  # Store predictions for each image
    
    # Disable gradient computation for faster inference
    with torch.no_grad():
        for inputs, img_paths in dataloader:
            inputs = inputs.to(device)  # Move input data to GPU/CPU
            outputs = model(inputs)  # Perform inference
            _, preds = torch.max(outputs, 1)  # Get predicted class
            softmax_scores = torch.nn.functional.softmax(outputs, dim=1)  # Compute softmax probabilities

            # Process each image in the batch
            for i, (pred, score) in enumerate(zip(preds, softmax_scores)):
                img_path = img_paths[i]
                pred_label = class_names[pred.item()]  # Get class name (crack/nocrack)
                score_value = score[pred.item()].item()  # Get confidence score

                # If the confidence score is below the threshold, label as 'unknown'
                if score_value < confidence_threshold:
                    pred_label = 'unknown'

                logging.info(f"Image: {img_path}, Prediction: {pred_label}, Confidence Score: {score_value:.4f}")
                predictions.append((img_path, pred_label, score_value))  # Save prediction details

    return predictions

def save_predictions_to_csv(predictions, output_folder):
    """
    Saves prediction results to a CSV file.

    Args:
        predictions: List of predictions containing image path, class label, and score.
        output_folder: Directory where the CSV file will be saved.
    """
    csv_file_path = os.path.join(output_folder, 'predictions.csv')

    try:
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['File Name', 'File Path', 'Prediction Score', 'Class Label'])  # Header row
            for img_path, pred_label, score_value in predictions:
                file_name = os.path.basename(img_path)
                writer.writerow([file_name, img_path, score_value, pred_label])  # Write each prediction
                logging.debug(f"Written to CSV: {file_name}, {img_path}, {score_value}, {pred_label}")
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")
        raise

def copy_images_concurrently(predictions, output_folder):
    """
    Copies the predicted images into corresponding folders based on their class labels.

    Args:
        predictions: List of predictions containing image path, class label, and score.
        output_folder: The parent directory where the 'crack', 'nocrack', and 'unknown' folders will be created.
    """
    # Define subdirectories for each class
    output_folders = {
        'crack': os.path.join(output_folder, 'sorted_crack'),
        'nocrack': os.path.join(output_folder, 'sorted_nocrack'),
        'unknown': os.path.join(output_folder, 'sorted_unknown')
    }

    # Create the subdirectories if they do not exist
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    # Use multithreading to copy images concurrently for better performance
    with ThreadPoolExecutor() as executor:
        futures = []
        for img_path, label, _ in predictions:
            destination = os.path.join(output_folders[label], os.path.basename(img_path))  # Destination path
            futures.append(executor.submit(shutil.copy, img_path, destination))  # Copy image
            logging.debug(f"Queued copy of {img_path} to {destination}")
    
    # Wait for all copy operations to complete
    for future in futures:
        try:
            future.result()  # Check if any exception occurred during the copy operation
        except Exception as e:
            logging.error(f"Error during file copy: {e}")

    # Save predictions to CSV file in the output folder
    save_predictions_to_csv(predictions, output_folder)

class SortingApp:
    """
    Tkinter-based GUI application for sorting solder joint images using a pre-trained model.
    
    Attributes:
        model: The loaded machine learning model for image classification.
        image_folder: Directory where images are loaded from.
        confidence_threshold: Confidence threshold to classify images.
        output_folder_name: Name of the folder where sorted images will be stored.
        canvas: Tkinter canvas to display sorting results.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Solder Joint Sorting Application")
        self.model = None
        self.model_info = None  # Store model info
        self.image_folder = None
        self.confidence_threshold = tk.DoubleVar(value=0.9)
        self.output_folder_name = tk.StringVar(value="sorted")
        self.predictions = []  # Store predictions for preview
        self.current_preview_index = 0  # For image preview navigation
        self.preview_images = []  # Store loaded preview images
        self.create_widgets()

    def create_widgets(self):
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.load_folder_button = tk.Button(control_frame, text="Load Image Folder", command=self.load_folder)
        self.load_folder_button.pack(pady=5, fill=tk.X)

        self.load_model_button = tk.Button(control_frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=5, fill=tk.X)

        self.model_info_button = tk.Button(control_frame, text="Show Model Info", command=self.show_model_info)
        self.model_info_button.pack(pady=5, fill=tk.X)

        tk.Label(control_frame, text="Confidence Threshold").pack(pady=2)
        self.confidence_slider = tk.Scale(control_frame, from_=0.5, to=1.0, orient=tk.HORIZONTAL, resolution=0.01, variable=self.confidence_threshold)
        self.confidence_slider.pack(pady=2, fill=tk.X)

        tk.Label(control_frame, text="Output Folder Name").pack(pady=2)
        self.output_folder_entry = tk.Entry(control_frame, textvariable=self.output_folder_name)
        self.output_folder_entry.pack(pady=2, fill=tk.X)

        self.start_sort_button = tk.Button(control_frame, text="Start Sorting", command=self.start_sorting)
        self.start_sort_button.pack(pady=5, fill=tk.X)

        # Progress bar and status label
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(pady=5, fill=tk.X)
        self.status_label = tk.Label(control_frame, text="Status: Idle")
        self.status_label.pack(pady=2, fill=tk.X)

        # Image preview panel
        preview_frame = tk.Frame(self.root)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        tk.Label(preview_frame, text="Image Preview").pack()
        self.image_label = tk.Label(preview_frame)
        self.image_label.pack(pady=5)
        self.prediction_label = tk.Label(preview_frame, text="Prediction: N/A\nConfidence: N/A")
        self.prediction_label.pack(pady=2)
        nav_frame = tk.Frame(preview_frame)
        nav_frame.pack(pady=2)
        self.prev_button = tk.Button(nav_frame, text="Previous", command=self.show_prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=2)
        self.next_button = tk.Button(nav_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.LEFT, padx=2)

        # Placeholder for sorting results graph
        self.canvas = None

    def set_status(self, text):
        self.status_label.config(text=f"Status: {text}")
        self.root.update_idletasks()

    def update_progress(self, value, maximum=None):
        if maximum is not None:
            self.progress['maximum'] = maximum
        self.progress['value'] = value
        self.root.update_idletasks()

    def load_folder(self):
        self.image_folder = filedialog.askdirectory()
        if self.image_folder:
            logging.info(f"Image folder loaded: {self.image_folder}")
            self.set_status("Image folder loaded.")
            self.load_preview_images()
        else:
            messagebox.showerror("Error", "No folder selected.")

    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if model_path:
            try:
                self.model = load_model(model_path)
                self.model_info = self.get_model_info(model_path)
                logging.info(f"Model loaded from: {model_path}")
                self.set_status("Model loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
        else:
            messagebox.showerror("Error", "No model selected.")

    def get_model_info(self, model_path):
        info = {}
        info['Architecture'] = 'CustomResNet50 (ResNet50 backbone)'
        info['Model Path'] = model_path
        try:
            state = torch.load(model_path, map_location='cpu')
            if isinstance(state, dict):
                if 'accuracy' in state:
                    info['Accuracy'] = state['accuracy']
                if 'epoch' in state:
                    info['Epoch'] = state['epoch']
                if 'date' in state:
                    info['Training Date'] = state['date']
        except Exception as e:
            info['Error'] = f"Could not read extra info: {e}"
        return info

    def show_model_info(self):
        if not self.model_info:
            messagebox.showinfo("Model Info", "No model loaded.")
            return
        info_str = "\n".join([f"{k}: {v}" for k, v in self.model_info.items()])
        messagebox.showinfo("Model Info", info_str)

    def start_sorting(self):
        if not self.model or not self.image_folder:
            messagebox.showerror("Error", "Please load both the model and image folder before sorting.")
            return
        dataset = SolderJointDataset(image_folder=self.image_folder, transform=transform_test)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
        confidence_threshold = self.confidence_threshold.get()
        output_folder = os.path.join(self.image_folder, self.output_folder_name.get())
        if os.path.exists(output_folder):
            messagebox.showerror("Error", f"Output folder '{output_folder}' already exists.")
            return
        try:
            os.makedirs(output_folder)
            logging.info(f"Created output folder: {output_folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create output folder: {e}")
            return
        # Run predictions and sort images with progress
        try:
            self.set_status("Predicting images...")
            self.update_progress(0, maximum=len(dataset))
            self.predictions = []
            total = len(dataset)
            processed = 0
            class_names = ['crack', 'nocrack']
            with torch.no_grad():
                for inputs, img_paths in dataloader:
                    inputs = inputs.to(device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    softmax_scores = torch.nn.functional.softmax(outputs, dim=1)
                    for i, (pred, score) in enumerate(zip(preds, softmax_scores)):
                        img_path = img_paths[i]
                        pred_label = class_names[pred.item()]
                        score_value = score[pred.item()].item()
                        if score_value < confidence_threshold:
                            pred_label = 'unknown'
                        self.predictions.append((img_path, pred_label, score_value))
                        processed += 1
                        self.update_progress(processed)
            self.set_status("Copying images...")
            self.update_progress(0, maximum=len(self.predictions))
            self.copy_images_with_progress(self.predictions, output_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.set_status("Error during prediction.")
            return
        self.set_status("Done.")
        self.load_preview_images(predictions=self.predictions)
        self.show_results(output_folder)

    def copy_images_with_progress(self, predictions, output_folder):
        output_folders = {
            'crack': os.path.join(output_folder, 'sorted_crack'),
            'nocrack': os.path.join(output_folder, 'sorted_nocrack'),
            'unknown': os.path.join(output_folder, 'sorted_unknown')
        }
        for folder in output_folders.values():
            os.makedirs(folder, exist_ok=True)
        processed = 0
        total = len(predictions)
        for img_path, label, _ in predictions:
            destination = os.path.join(output_folders[label], os.path.basename(img_path))
            try:
                shutil.copy(img_path, destination)
            except Exception as e:
                logging.error(f"Error copying {img_path}: {e}")
            processed += 1
            self.update_progress(processed)
        save_predictions_to_csv(predictions, output_folder)

    def load_preview_images(self, predictions=None):
        # Load images for preview (before or after sorting)
        self.preview_images = []
        self.current_preview_index = 0
        if predictions is None:
            # Before sorting: just show images in folder
            if not self.image_folder:
                return
            image_paths = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith('.png')]
            self.preview_predictions = [(p, None, None) for p in image_paths]
        else:
            # After sorting: show predictions
            self.preview_predictions = predictions
        for img_path, pred_label, score in self.preview_predictions:
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((256, 256))
                img_tk = ImageTk.PhotoImage(img)
                self.preview_images.append((img_tk, pred_label, score, img_path))
            except Exception as e:
                logging.error(f"Error loading preview image {img_path}: {e}")
        self.show_preview_image()

    def show_preview_image(self):
        if not self.preview_images:
            self.image_label.config(image='', text='No image')
            self.prediction_label.config(text='Prediction: N/A\nConfidence: N/A')
            return
        img_tk, pred_label, score, img_path = self.preview_images[self.current_preview_index]
        self.image_label.config(image=img_tk)
        self._current_image_tk = img_tk  # Keep reference to avoid garbage collection
        pred_str = f"Prediction: {pred_label if pred_label is not None else 'N/A'}\nConfidence: {score:.4f}" if score is not None else "Prediction: N/A\nConfidence: N/A"
        self.prediction_label.config(text=pred_str + f"\nFile: {os.path.basename(img_path)}")

    def show_next_image(self):
        if not self.preview_images:
            return
        self.current_preview_index = (self.current_preview_index + 1) % len(self.preview_images)
        self.show_preview_image()

    def show_prev_image(self):
        if not self.preview_images:
            return
        self.current_preview_index = (self.current_preview_index - 1) % len(self.preview_images)
        self.show_preview_image()

    def show_results(self, output_folder):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        folders = {
            'crack': os.path.join(output_folder, 'sorted_crack'),
            'nocrack': os.path.join(output_folder, 'sorted_nocrack'),
            'unknown': os.path.join(output_folder, 'sorted_unknown')
        }
        counts = {label: len(os.listdir(path)) for label, path in folders.items() if os.path.exists(path)}
        logging.info(f"Image counts: {counts}")
        fig, ax = plt.subplots()
        labels = list(counts.keys())
        values = list(counts.values())
        ax.bar(labels, values, color=['blue', 'green', 'orange'])
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Images')
        ax.set_title('Number of Images in Each Class Folder')
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()  # Create the Tkinter root window
    root.geometry("1200x700")  # Set a larger default window size
    app = SortingApp(root)  # Initialize the SortingApp
    sv_ttk.use_light_theme()  # Set the theme to light
    def on_closing():
        root.destroy()
        import sys
        sys.exit(0)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()  # Start the Tkinter event loop

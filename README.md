# Solder Joint Classifier

A machine learning application for automated classification of solder joint images using deep learning. The application uses a custom ResNet50 model to classify solder joints as "crack", "nocrack", or "unknown" based on confidence thresholds.

## Features

- **GUI-based Image Classification**: User-friendly Tkinter interface for easy interaction
- **Batch Processing**: Process multiple images simultaneously with progress tracking
- **Confidence-based Classification**: Adjustable confidence threshold for uncertain predictions
- **Image Preview**: Navigate through images with prediction results
- **Automated Sorting**: Automatically sorts classified images into separate folders
- **Results Visualization**: Bar chart showing classification distribution
- **CSV Export**: Detailed prediction results exported to CSV format
- **Multi-threaded Processing**: Concurrent image copying for improved performance

## Technology Stack

- **Deep Learning Framework**: PyTorch with torchvision
- **Model Architecture**: Custom ResNet50 with modified final layers for binary classification
- **GUI Framework**: Tkinter with sv_ttk theming
- **Image Processing**: PIL (Python Imaging Library)
- **Data Visualization**: Matplotlib
- **Configuration**: YAML-based configuration management
- **Logging**: Comprehensive logging for debugging and monitoring

## Model Architecture

The application uses a **CustomResNet50** model:
- **Backbone**: Pre-trained ResNet50 (ImageNet weights)
- **Modifications**: 
  - Final fully connected layer replaced with dropout (0.7) + linear layer
  - Output: 2 classes (crack/nocrack)
  - Input: Grayscale images, resized and normalized

## Installation and Setup

### Docker Installation (Recommended)

1. **Build the Docker image**:
   ```bash
   docker build -t solder-joint-classifier .
   ```

2. **Run with X11 forwarding** (Linux/macOS):
   ```bash
   # Allow X11 connections
   xhost +local:docker
   
   # Run the container
   docker run -it --rm \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v $(pwd)/input:/app/input \
     -v $(pwd)/output:/app/output \
     solder-joint-classifier
   ```

3. **Run on Windows** (with VcXsrv or similar X server):
   ```bash
   docker run -it --rm \
     -e DISPLAY=host.docker.internal:0.0 \
     -v %cd%/input:/app/input \
     -v %cd%/output:/app/output \
     solder-joint-classifier
   ```

### Local Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare configuration**:
   Ensure `config.yaml` is present with proper settings

3. **Run the application**:
   ```bash
   python inference_gui.py
   ```

## Requirements

See `requirements.txt` for complete dependency list:

```txt
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.1.78
Pillow==10.0.0
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
matplotlib==3.7.2
tqdm==4.66.1
joblib==1.3.2
PyYAML
sv_ttk
```

## Configuration

The application uses a `config.yaml` file for configuration:

```yaml
# Image preprocessing settings
image_transform:
  resize: [224, 224]
  grayscale: 1
  normalize_mean: [0.485]
  normalize_std: [0.229]

# Model settings
batch_size: 32
```

## Usage

### 1. Load Model
- Click "Load Model" and select your trained `.pth` model file
- View model information using "Show Model Info"

### 2. Load Images
- Click "Load Image Folder" and select a directory containing PNG images
- Images will be loaded for preview

### 3. Configure Settings
- **Confidence Threshold**: Adjust slider (0.5-1.0) for classification sensitivity
- **Output Folder Name**: Specify name for results folder

### 4. Start Classification
- Click "Start Sorting" to begin processing
- Monitor progress through the progress bar
- View real-time status updates

### 5. Review Results
- Navigate through processed images using Previous/Next buttons
- View prediction confidence and class labels
- Examine the results bar chart

## Output Structure

After processing, the application creates:

```
output_folder/
├── sorted_crack/          # Images classified as having cracks
├── sorted_nocrack/        # Images classified as crack-free
├── sorted_unknown/        # Images below confidence threshold
└── predictions.csv        # Detailed results with confidence scores
```

## CSV Output Format

The `predictions.csv` file contains:
- **File Name**: Original image filename
- **File Path**: Full path to source image
- **Prediction Score**: Confidence score (0.0-1.0)
- **Class Label**: crack/nocrack/unknown

## Docker Architecture

The application uses a **multi-stage Docker build**:

### Stage 1: Builder
- Full development environment with build tools
- Compiles and installs all Python dependencies
- Includes development headers and compilation tools

### Stage 2: Runtime
- Minimal runtime environment
- Only essential system libraries for GUI and ML inference
- Copies pre-built Python packages from builder stage
- Significantly smaller final image size

### Key Docker Features:
- **X11 Support**: GUI applications run in containerized environment
- **Volume Mounting**: Easy data exchange between host and container
- **Multi-platform**: Supports Linux, macOS, and Windows (with X server)

## System Requirements

### Minimum Requirements:
- **OS**: Linux, macOS, or Windows with X server
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.9+

### GPU Support:
- CUDA-compatible GPU (optional but recommended)
- CUDA 11.7+ and cuDNN for GPU acceleration

## Troubleshooting

### Common Issues:

1. **GUI not displaying in Docker**:
   - Ensure X11 forwarding is properly configured
   - Check DISPLAY environment variable
   - Verify X server permissions

2. **Model loading errors**:
   - Verify model file path and format
   - Check CUDA/CPU compatibility
   - Ensure sufficient memory

3. **Image processing failures**:
   - Verify image format (PNG required)
   - Check file permissions
   - Ensure sufficient disk space

## Performance Optimization

- **GPU Acceleration**: Automatically detects and uses CUDA if available
- **Multi-threading**: Concurrent image copying and processing
- **Batch Processing**: Efficient memory usage with configurable batch sizes
- **Memory Management**: Proper cleanup and garbage collection

## Development

### Project Structure:
```
project/
├── inference_gui.py       # Main application file
├── config.yaml           # Configuration settings
├── models/               # Trained model files
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Key Classes:
- **`CustomResNet50`**: Neural network model definition
- **`SolderJointDataset`**: Custom dataset for image loading
- **`SortingApp`**: Main GUI application class

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Support

For issues and questions:
- Check the troubleshooting section
- Review application logs for error details
- Ensure all dependencies are properly installed
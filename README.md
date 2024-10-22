# FLUX Image Generator

An interactive web application for generating images using the FLUX model with CUDA optimization and a stylish Dracula-themed UI.

## Features
- Real-time image generation with FLUX model
- CUDA GPU optimization
- Interactive controls for generation parameters
- Image history with download capability
- Responsive Dracula-themed UI
- Progress tracking and error handling

## Requirements
- Python 3.8+
- CUDA-capable NVIDIA GPU (recommended)
- At least 8GB VRAM for optimal performance

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd flux-image-generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers streamlit accelerate xformers
```

## Running the Application

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to:
```
http://localhost:8501
```

## Configuration

The application includes several configurable parameters:
- Guidance Scale: Controls how closely the generation follows the prompt
- Image Dimensions: Adjustable width and height
- Inference Steps: Controls generation quality and speed
- Model Settings: CUDA optimization and memory efficiency

## Troubleshooting

1. If you encounter CUDA out-of-memory errors:
   - Reduce the image dimensions
   - Decrease the number of inference steps
   - Ensure no other GPU-intensive applications are running

2. If the model loads slowly:
   - The first load caches the model for subsequent use
   - Check your internet connection for initial model download

## License
This project uses the FLUX model from Black Forest Labs. Please check their licensing terms for commercial use.

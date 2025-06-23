# EMG Hand Control Project

This project uses EMG signals to control a robotic hand simulation using MuJoCo.

## Prerequisites

- Python 3.8+
- pyenv (recommended for virtual environment)

## Setup Instructions

### 1. Download and Extract Dataset

1. Go to [Ninapro DB1 Dataset](https://www.kaggle.com/datasets/mansibmursalin/ninapro-db1-full-dataset)
2. Download the dataset (you'll need a Kaggle account)
3. Extract the downloaded file:

**Windows:**
- Right-click the downloaded `.zip` file
- Select "Extract All..."
- Choose your project directory

**Mac/Linux:**
```bash
unzip ninapro-db1-full-dataset.zip
```

**If it's a `.tar.gz` file:**
```bash
tar -xzf ninapro-db1-full-dataset.tar.gz
```

**If it's a `.7z` file:**
```bash
7z x ninapro-db1-full-dataset.7z
```

### 2. Set Up Python Environment

**Using pyenv (recommended):**
```bash
# Create virtual environment
pyenv virtualenv 3.9.0 emg-hand-control
pyenv activate emg-hand-control

# Or if using regular Python venv:
python -m venv emg-hand-control
source emg-hand-control/bin/activate  # Linux/Mac
# emg-hand-control\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install mujoco
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
pip install torchviz
```

### 4. Required Files

Make sure you have these files in your project directory:
- `shadow_hand/scene_right.xml` - MuJoCo scene file
- `main.py` - Training script
- `hand.py` - Hand interaction script
- The extracted Ninapro dataset

### 5. Environment Variable

Set the MuJoCo XML path:
```bash
export MUJOCO_XML_PATH="shadow_hand/scene_right.xml"
```

**Windows:**
```cmd
set MUJOCO_XML_PATH=shadow_hand/scene_right.xml
```

## Usage

### Train the Model
```bash
python main.py
```

### Run Hand Interaction
```bash
python hand.py
```

## Project Structure
```
├── main.py                    # Training script
├── hand.py                    # Hand interaction script
├── shadow_hand/
│   └── scene_right.xml       # MuJoCo scene file
├── ninapro-db1-dataset/      # Extracted dataset
└── README.md
```

## Troubleshooting

- If MuJoCo fails to load, ensure you have the correct XML file path
- If dataset loading fails, verify the CSV files are properly extracted
- For GPU training, ensure CUDA is installed and compatible with PyTorch

## Dependencies Overview

- **MuJoCo**: Physics simulation
- **PyTorch**: Deep learning framework
- **NumPy/Pandas**: Data manipulation
- **Scikit-learn**: Data preprocessing and metrics
- **Matplotlib/Seaborn**: Visualization
- **TorchViz**: Model visualization


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import torch
from torchviz import make_dot
warnings.filterwarnings('ignore')

# =====================================================================
# SYSTEM CONFIGURATION AND INITIALIZATION
# =====================================================================

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# =====================================================================
# DATASET DEFINITION
# =====================================================================

class EMGDataset(Dataset):
    """
    Custom PyTorch Dataset for EMG-to-Glove mapping.
    
    
    Attributes:
        emg_data (torch.FloatTensor): Tensor containing EMG sensor readings
                                      Shape: (n_samples, sequence_length, n_emg_channels)
        glove_data (torch.FloatTensor): Tensor containing glove sensor positions
                                        Shape: (n_samples, n_glove_sensors)
    """
    
    def __init__(self, emg_data, glove_data):
        """
        Initialize the EMG dataset.
        
        Args:
            emg_data (np.ndarray): Raw EMG sensor data
            glove_data (np.ndarray): Raw glove sensor positions
        """
        self.emg_data = torch.FloatTensor(emg_data)
        self.glove_data = torch.FloatTensor(glove_data)
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.emg_data)
    
    def __getitem__(self, idx):
        return self.emg_data[idx], self.glove_data[idx]

# =====================================================================
# MODEL ARCHITECTURES
# =====================================================================

class BasicRNN(nn.Module):
    """
    Convolutional-LSTM Architecture with Attention Mechanism
    
    This model implements a hybrid architecture combining:
    1. Convolutional layers for local feature extraction
    2. LSTM layers for temporal modeling
    3. Attention mechanism for focusing on relevant time steps
    
    Architecture Details:
        - 3 convolutional layers for feature extraction
        - 2-layer bidirectional LSTM for sequence modeling
        - Attention mechanism for temporal weighting
        - Dropout regularization to prevent overfitting
    """
    
    def __init__(self, input_size=10, hidden_size=64, output_size=22, num_layers=2):
        """
        Initialize the RNN model architecture.
        
        Args:
            input_size (int): Number of EMG channels (default: 10)
            hidden_size (int): Hidden dimension for LSTM layers (default: 64)
            output_size (int): Number of glove sensors to predict (default: 22)
            num_layers (int): Number of LSTM layers (default: 2)
        """
        super(BasicRNN, self).__init__()
        
        # Convolutional Feature Extraction Block
        # These layers extract local temporal patterns from EMG signals
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        
        # Batch normalization for stable training
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        
        # LSTM Sequence Modeling Block
        # Captures long-term temporal dependencies in EMG signals
        self.lstm = nn.LSTM(32, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2, bidirectional=False)
        
        # Attention Mechanism
        # Learns to weight important time steps in the sequence
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output Projection Block
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass through the RNN model.
        
        Args:
            x (torch.Tensor): Input EMG sequences
                             Shape: (batch_size, sequence_length, n_channels)
                             
        Returns:
            torch.Tensor: Predicted glove positions
                         Shape: (batch_size, n_glove_sensors)
        """
        # Reshape for convolutional processing: (batch, channels, sequence)
        x = x.transpose(1, 2)
        
        # Apply convolutional feature extraction with batch normalization
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Reshape back for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)
        
        # Process through LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism
        # Compute attention weights for each time step
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention-weighted aggregation
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Generate final predictions with dropout regularization
        output = self.fc(self.dropout(attended))
        
        return output

class BasicTransformer(nn.Module):
    """
    Transformer-based Architecture
    
    Architecture Details:
        - Linear projection to transformer dimension
        - Learnable positional encodings
        - Multi-head self-attention layers
        - Feed-forward networks with residual connections
        - Global average pooling for sequence aggregation
    """
    
    def __init__(self, input_size=10, d_model=64, nhead=8, num_layers=4, output_size=22):
        """
        Initialize the Transformer model architecture.
        
        Args:
            input_size (int): Number of EMG channels (default: 10)
            d_model (int): Transformer model dimension (default: 64)
            nhead (int): Number of attention heads (default: 8)
            num_layers (int): Number of transformer encoder layers (default: 4)
            output_size (int): Number of glove sensors to predict (default: 22)
        """
        super(BasicTransformer, self).__init__()
        

        self.input_proj = nn.Linear(input_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation='gelu'  # GELU activation for better gradient 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output Projection Block
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass through the Transformer model.
        
        Args:
            x (torch.Tensor): Input EMG sequences
                             Shape: (batch_size, sequence_length, n_channels)
                             
        Returns:
            torch.Tensor: Predicted glove positions
                         Shape: (batch_size, n_glove_sensors)
        """
        seq_len = x.size(1)
        
        # Project input features to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding to maintain sequence order information
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Process through transformer encoder layers
        x = self.transformer(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Global average pooling across sequence dimension
        # This aggregates the entire sequence into a fixed-size representation
        x = x.mean(dim=1)
        
        # Generate final predictions with dropout regularization
        output = self.fc(self.dropout(x))
        
        return output

# =====================================================================
# DATA LOADING AND PREPROCESSING
# =====================================================================

def load_small_dataset(file_path, max_samples=15000, seq_length=100):
    """
    Load and preprocess a subset of the EMG dataset.

    Args:
        file_path (str): Path to the CSV file containing EMG data
        max_samples (int): Maximum number of sequences to generate
        seq_length (int): Length of each sequence window
        
    Returns:
        tuple: (X, y, scaler_emg, scaler_glove) where:
               - X: EMG sequences (n_samples, seq_length, n_channels)
               - y: Target glove positions (n_samples, n_sensors)
               - scaler_emg: Fitted StandardScaler for EMG data
               - scaler_glove: Fitted StandardScaler for glove data
    """
    print("="*70)
    print("INITIALIZING DATA LOADING PIPELINE")
    print("="*70)
    
    # Load a manageable chunk of the dataset
    chunk_size = 100000  # Limit memory usage
    df_chunk = pd.read_csv(file_path, nrows=chunk_size)
    
    print(f"Successfully loaded data chunk with shape: {df_chunk.shape}")
    print(f"Available columns: {list(df_chunk.columns)}")
    
    # Define column mappings for EMG and glove sensors
    emg_cols = [f'emg_{i}' for i in range(10)]  # 10 EMG channels
    glove_cols = [f'glove_{i}' for i in range(22)]  # 22 glove sensors
    
    print(f"\nEMG sensor columns (10 channels): {emg_cols}")
    print(f"Glove sensor columns (22 positions): {glove_cols}")
    
    # Extract sensor data
    emg_data = df_chunk[emg_cols].values
    glove_data = df_chunk[glove_cols].values
    
    # Display data statistics
    print(f"\nRaw data shapes:")
    print(f"  EMG data: {emg_data.shape}")
    print(f"  Glove data: {glove_data.shape}")
    print(f"\nRaw data ranges:")
    print(f"  EMG: [{emg_data.min():.4f}, {emg_data.max():.4f}]")
    print(f"  Glove: [{glove_data.min():.1f}, {glove_data.max():.1f}]")
    
    # Filter out rest periods (stimulus = 0) to focus on active movements
    if 'stimulus' in df_chunk.columns:
        active_mask = df_chunk['stimulus'] != 0
        emg_data = emg_data[active_mask]
        glove_data = glove_data[active_mask]
        print(f"\nAfter filtering inactive periods: {emg_data.shape}")
    
    # Subsample if necessary to reduce computational load
    if len(emg_data) > max_samples * 2:
        step = len(emg_data) // (max_samples * 2)
        emg_data = emg_data[::step]
        glove_data = glove_data[::step]
        print(f"After subsampling: {emg_data.shape}")
    
    # Normalize data using StandardScaler for stable training
    print("\nApplying standardization normalization...")
    scaler_emg = StandardScaler()
    scaler_glove = StandardScaler()
    
    emg_data = scaler_emg.fit_transform(emg_data)
    glove_data = scaler_glove.fit_transform(glove_data)
    
    print(f"Post-normalization ranges:")
    print(f"  EMG: [{emg_data.min():.3f}, {emg_data.max():.3f}]")
    print(f"  Glove: [{glove_data.min():.3f}, {glove_data.max():.3f}]")
    
    # Create overlapping sequences for temporal modeling
    print(f"\nGenerating sequences of length {seq_length}...")
    X, y = [], []
    step_size = seq_length // 3  # 66.7% overlap between sequences
    
    for i in range(0, len(emg_data) - seq_length, step_size):
        # Extract EMG sequence window
        X.append(emg_data[i:i+seq_length])
        # Use the last timestep's glove position as target
        y.append(glove_data[i+seq_length-1])
        
        if len(X) >= max_samples:
            break
    
    print(f"Successfully created {len(X)} sequences")
    
    return np.array(X), np.array(y), scaler_emg, scaler_glove

# =====================================================================
# MODEL EVALUATION AND ANALYSIS
# =====================================================================

def generate_confusion_matrix_analysis(model, val_loader, device, model_name="Model"):
    """
    Generate comprehensive confusion matrix analysis for multi-output regression.
    
    """
    print(f"\n{'='*70}")
    print(f"GENERATING CONFUSION MATRIX ANALYSIS FOR {model_name.upper()}")
    print(f"{'='*70}")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    # Collect all predictions
    with torch.no_grad():
        for emg, glove in tqdm(val_loader, desc="Evaluating"):
            emg, glove = emg.to(device), glove.to(device)
            output = model(emg)
            all_predictions.append(output.cpu().numpy())
            all_targets.append(glove.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Calculate R² scores for each glove sensor
    r2_scores = []
    for i in range(22):
        r2 = r2_score(targets[:, i], predictions[:, i])
        r2_scores.append(r2)
    
    print(f"\nR² Scores by Glove Sensor:")
    for i, r2 in enumerate(r2_scores):
        print(f"  Sensor {i:2d}: R² = {r2:.4f}")
    
    # Create confusion matrices for selected sensors
    # We'll analyze sensors with highest, median, and lowest R² scores
    sorted_indices = np.argsort(r2_scores)
    sensors_to_analyze = [
        sorted_indices[-1],  # Best performing
        sorted_indices[len(sorted_indices)//2],  # Median performing
        sorted_indices[0]   # Worst performing
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model_name} - Confusion Matrix Analysis for Selected Glove Sensors', fontsize=16)
    
    for idx, sensor_id in enumerate(sensors_to_analyze):
        # Discretize predictions and targets into bins
        n_bins = 10
        pred_discrete = pd.cut(predictions[:, sensor_id], bins=n_bins, labels=False)
        target_discrete = pd.cut(targets[:, sensor_id], bins=n_bins, labels=False)
        
        # Generate confusion matrix
        cm = confusion_matrix(target_discrete, pred_discrete, labels=range(n_bins))
        
        # Plot confusion matrix
        ax = axes[0, idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title(f'Sensor {sensor_id} (R² = {r2_scores[sensor_id]:.3f})')
        ax.set_xlabel('Predicted Bin')
        ax.set_ylabel('Actual Bin')
        
        # Plot scatter plot for continuous values
        ax = axes[1, idx]
        ax.scatter(targets[:, sensor_id], predictions[:, sensor_id], alpha=0.5, s=1)
        ax.plot([-3, 3], [-3, 3], 'r--', lw=2)  # Perfect prediction line
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_xlabel('Actual Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(f'Sensor {sensor_id} - Scatter Plot')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_analysis.png', dpi=300, bbox_inches='tight')
    
    # Generate correlation heatmap between sensors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prediction correlation
    pred_corr = np.corrcoef(predictions.T)
    sns.heatmap(pred_corr, cmap='coolwarm', center=0, ax=ax1, 
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax1.set_title(f'{model_name} - Predicted Sensor Correlations')
    ax1.set_xlabel('Glove Sensor ID')
    ax1.set_ylabel('Glove Sensor ID')
    
    # Target correlation
    target_corr = np.corrcoef(targets.T)
    sns.heatmap(target_corr, cmap='coolwarm', center=0, ax=ax2,
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    ax2.set_title('Actual Sensor Correlations')
    ax2.set_xlabel('Glove Sensor ID')
    ax2.set_ylabel('Glove Sensor ID')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_correlation_analysis.png', dpi=300, bbox_inches='tight')
    
    return {
        'predictions': predictions,
        'targets': targets,
        'r2_scores': r2_scores,
        'mean_r2': np.mean(r2_scores)
    }

# =====================================================================
# TRAINING PIPELINE
# =====================================================================

def train_basic_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    Enhanced training loop with comprehensive monitoring and early stopping.
    
    This function implements:
    1. Adaptive learning rate scheduling
    2. Early stopping based on validation loss
    3. Gradient clipping for stable training
    4. Detailed progress monitoring
    
    Returns:
        tuple: (train_losses, val_losses, best_model_state)
    """
    # Configure device for GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler for adaptive learning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training history tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 10  # Early stopping patience
    
    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Initial Learning Rate: {lr}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"{'='*70}\n")
    
    for epoch in range(epochs):
        # -------------------- Training Phase --------------------
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch_idx, (emg, glove) in enumerate(train_progress):
            emg, glove = emg.to(device), glove.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(emg)
            loss = criterion(output, glove)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # -------------------- Validation Phase --------------------
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for emg, glove in val_progress:
                emg, glove = emg.to(device), glove.to(device)
                output = model(emg)
                loss = criterion(output, glove)
                val_loss += loss.item()
                
                val_progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  New best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Patience: {patience_counter}/{max_patience}")
        print("-"*50)
        
        # Early stopping check
        if patience_counter >= max_patience:
            print(f"\n Early stopping triggered after {epoch+1} epochs")
            break
        
        # Stability check
        if train_loss > 10 or val_loss > 10 or np.isnan(train_loss) or np.isnan(val_loss):
            print(f"\n Training instability detected, stopping")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Restored best model with validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses

# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================

def main():
    """
    Main execution pipeline for EMG signal classification.
    
    This function orchestrates the entire training and evaluation process:
    1. Data loading and preprocessing
    2. Model initialization
    3. Training both RNN and Transformer models
    4. Comprehensive evaluation with confusion matrices
    5. Results visualization and model persistence
    """
    print("\n" + "="*70)
    print(" EMG TO HAND POSITION MAPPING SYSTEM")
    print(" Deep Learning Pipeline v2.0")
    print("="*70)
    print("\nSystem Overview:")
    print("  • Input: 10 EMG muscle sensors")
    print("  • Output: 22 hand position sensors")
    print("  • Models: RNN with Attention & Transformer")
    print("="*70 + "\n")
    
    # -------------------- Data Loading --------------------
    X, y, scaler_emg, scaler_glove = load_small_dataset(
        'Ninapro_DB1.csv', 
        max_samples=12000, 
        seq_length=80
    )
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"EMG sequences shape: {X.shape}")
    print(f"   {X.shape[0]} samples")
    print(f"   {X.shape[1]} timesteps per sequence")
    print(f"   {X.shape[2]} EMG channels")
    print(f"\nGlove positions shape: {y.shape}")
    print(f"  {y.shape[0]} samples")
    print(f"   {y.shape[1]} glove sensors")
    
    # -------------------- Data Splitting --------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"\nTrain/Validation Split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    
    # -------------------- Dataset Creation --------------------
    train_dataset = EMGDataset(X_train, y_train)
    val_dataset = EMGDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    
    # -------------------- RNN Model Training --------------------
    print(f"\n{'='*70}")
    print("PHASE 1: RNN MODEL TRAINING")
    print(f"{'='*70}")
    
    rnn_model = BasicRNN(
        input_size=10,
        hidden_size=128,
        output_size=22,
        num_layers=3
    )
    
    # Print model architecture
    total_params = sum(p.numel() for p in rnn_model.parameters())
    trainable_params = sum(p.numel() for p in rnn_model.parameters() if p.requires_grad)
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    rnn_losses = train_basic_model(
        rnn_model, train_loader, val_loader, epochs=20, lr=0.001
    )
    
    # -------------------- Transformer Model Training --------------------
    print(f"\n{'='*70}")
    print("PHASE 2: TRANSFORMER MODEL TRAINING")
    print(f"{'='*70}")
    
    transformer_model = BasicTransformer(
        input_size=10,
        d_model=128,
        nhead=8,
        num_layers=6,
        output_size=22
    )
    
    # Print model architecture
    total_params = sum(p.numel() for p in transformer_model.parameters())
    trainable_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    print(f"Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    transformer_losses = train_basic_model(
        transformer_model, train_loader, val_loader, epochs=20, lr=0.001
    )
    
    # -------------------- Confusion Matrix Analysis --------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rnn_results = generate_confusion_matrix_analysis(
        rnn_model, val_loader, device, "RNN"
    )
    
    transformer_results = generate_confusion_matrix_analysis(
        transformer_model, val_loader, device, "Transformer"
    )
    
    # -------------------- Comparative Visualization --------------------
    print(f"\n{'='*70}")
    print("GENERATING COMPARATIVE VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training curves comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(rnn_losses[0], label='RNN Train', linewidth=2, color='blue')
    ax1.plot(rnn_losses[1], label='RNN Val', linewidth=2, color='lightblue')
    ax1.plot(transformer_losses[0], label='Transformer Train', linewidth=2, color='green')
    ax1.plot(transformer_losses[1], label='Transformer Val', linewidth=2, color='lightgreen')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. R2 scores comparison
    ax2 = plt.subplot(2, 3, 2)
    x_pos = np.arange(22)
    width = 0.35
    ax2.bar(x_pos - width/2, rnn_results['r2_scores'], width, label='RNN', color='blue', alpha=0.7)
    ax2.bar(x_pos + width/2, transformer_results['r2_scores'], width, label='Transformer', color='green', alpha=0.7)
    ax2.set_xlabel('Glove Sensor ID', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('R² Scores by Sensor', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_pos, rotation=45)
    
    # 3. Mean R2 comparison
    ax3 = plt.subplot(2, 3, 3)
    models = ['RNN', 'Transformer']
    mean_r2s = [rnn_results['mean_r2'], transformer_results['mean_r2']]
    bars = ax3.bar(models, mean_r2s, color=['blue', 'green'], alpha=0.7)
    ax3.set_ylabel('Mean R² Score', fontsize=12)
    ax3.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_r2s):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. Error distribution - RNN
    ax4 = plt.subplot(2, 3, 4)
    rnn_errors = rnn_results['predictions'] - rnn_results['targets']
    ax4.hist(rnn_errors.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('Prediction Error', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('RNN - Error Distribution', fontsize=14, fontweight='bold')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Error distribution - Transformer
    ax5 = plt.subplot(2, 3, 5)
    transformer_errors = transformer_results['predictions'] - transformer_results['targets']
    ax5.hist(transformer_errors.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.set_xlabel('Prediction Error', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Transformer - Error Distribution', fontsize=14, fontweight='bold')
    ax5.axvline(0, color='red', linestyle='--', linewidth=2)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Sensor-wise error comparison
    ax6 = plt.subplot(2, 3, 6)
    rnn_rmse = np.sqrt(np.mean((rnn_results['predictions'] - rnn_results['targets'])**2, axis=0))
    transformer_rmse = np.sqrt(np.mean((transformer_results['predictions'] - transformer_results['targets'])**2, axis=0))
    
    ax6.plot(rnn_rmse, 'o-', label='RNN', color='blue', linewidth=2, markersize=6)
    ax6.plot(transformer_rmse, 's-', label='Transformer', color='green', linewidth=2, markersize=6)
    ax6.set_xlabel('Glove Sensor ID', fontsize=12)
    ax6.set_ylabel('RMSE', fontsize=12)
    ax6.set_title('Root Mean Square Error by Sensor', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('EMG Signal Classification - Comprehensive Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('emg_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # -------------------- Save Models --------------------
    print(f"\n{'='*70}")
    print("SAVING TRAINED MODELS")
    print(f"{'='*70}")
    
    # Save model states
    torch.save({
        'model_state_dict': rnn_model.state_dict(),
        'r2_scores': rnn_results['r2_scores'],
        'mean_r2': rnn_results['mean_r2'],
        'final_train_loss': rnn_losses[0][-1],
        'final_val_loss': rnn_losses[1][-1]
    }, 'rnn_emg_model_enhanced.pth')
    
    torch.save({
        'model_state_dict': transformer_model.state_dict(),
        'r2_scores': transformer_results['r2_scores'],
        'mean_r2': transformer_results['mean_r2'],
        'final_train_loss': transformer_losses[0][-1],
        'final_val_loss': transformer_losses[1][-1]
    }, 'transformer_emg_model_enhanced.pth')
    
    # Save scalers for future inference
    import joblib
    joblib.dump(scaler_emg, 'scaler_emg.pkl')
    joblib.dump(scaler_glove, 'scaler_glove.pkl')
    
    print(" Models saved successfully")
    print(" Scalers saved for inference pipeline")
    
    # -------------------- Final Summary Report --------------------
    print(f"\n{'='*70}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    print("\nRNN Model Performance:")
    print(f"   Final Training Loss: {rnn_losses[0][-1]:.4f}")
    print(f"   Final Validation Loss: {rnn_losses[1][-1]:.4f}")
    print(f"   Mean R² Score: {rnn_results['mean_r2']:.4f}")
    print(f"   Best Sensor R²: {max(rnn_results['r2_scores']):.4f} (Sensor {np.argmax(rnn_results['r2_scores'])})")
    print(f"   Worst Sensor R²: {min(rnn_results['r2_scores']):.4f} (Sensor {np.argmin(rnn_results['r2_scores'])})")
    
    print("\nTransformer Model Performance:")
    print(f"   Final Training Loss: {transformer_losses[0][-1]:.4f}")
    print(f"   Final Validation Loss: {transformer_losses[1][-1]:.4f}")
    print(f"   Mean R² Score: {transformer_results['mean_r2']:.4f}")
    print(f"   Best Sensor R²: {max(transformer_results['r2_scores']):.4f} (Sensor {np.argmax(transformer_results['r2_scores'])})")
    print(f"   Worst Sensor R²: {min(transformer_results['r2_scores']):.4f} (Sensor {np.argmin(transformer_results['r2_scores'])})")
    
    # Determine best model
    best_model = "RNN" if rnn_results['mean_r2'] > transformer_results['mean_r2'] else "Transformer"
    improvement = abs(rnn_results['mean_r2'] - transformer_results['mean_r2']) * 100
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION: {best_model} model performs better by {improvement:.2f}%")
    print(f"{'='*70}")
    
    # Generate sensor mapping insights
    print("\nSensor Mapping Insights:")
    
    # Find highly correlated sensor pairs
    target_corr = np.corrcoef(rnn_results['targets'].T)
    high_corr_pairs = []
    for i in range(22):
        for j in range(i+1, 22):
            if abs(target_corr[i, j]) > 0.7:
                high_corr_pairs.append((i, j, target_corr[i, j]))
    
    if high_corr_pairs:
        print("\n  Highly Correlated Sensor Pairs (|correlation| > 0.7):")
        for sensor1, sensor2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
            print(f"    • Sensors {sensor1} & {sensor2}: correlation = {corr:.3f}")
    
    # Identify challenging sensors
    challenging_threshold = 0.5
    rnn_challenging = [i for i, r2 in enumerate(rnn_results['r2_scores']) if r2 < challenging_threshold]
    transformer_challenging = [i for i, r2 in enumerate(transformer_results['r2_scores']) if r2 < challenging_threshold]
    
    if rnn_challenging or transformer_challenging:
        print(f"\n  Challenging Sensors (R² < {challenging_threshold}):")
        print(f"     RNN: {rnn_challenging if rnn_challenging else 'None'}")
        print(f"    Transformer: {transformer_challenging if transformer_challenging else 'None'}")
    
    return {
        'rnn_model': rnn_model,
        'transformer_model': transformer_model,
        'rnn_results': rnn_results,
        'transformer_results': transformer_results,
        'scalers': (scaler_emg, scaler_glove)
    }

# =====================================================================
# INFERENCE UTILITIES
# =====================================================================

class EMGInferenceEngine:
    """
    Production-ready inference engine for EMG signal classification.
    
    This class provides a clean interface for loading trained models
    and performing real-time predictions on new EMG data.
    """
    
    def __init__(self, model_path, scaler_emg_path, scaler_glove_path, model_type='rnn'):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to saved model checkpoint
            scaler_emg_path (str): Path to EMG scaler
            scaler_glove_path (str): Path to glove scaler
            model_type (str): Type of model ('rnn' or 'transformer')
        """
        import joblib
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # Load scalers
        self.scaler_emg = joblib.load(scaler_emg_path)
        self.scaler_glove = joblib.load(scaler_glove_path)
        
        # Load model - FIX: Add weights_only=False
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if model_type == 'rnn':
            self.model = BasicRNN(input_size=10, hidden_size=128, output_size=22, num_layers=3)
        else:
            self.model = BasicTransformer(input_size=10, d_model=128, nhead=8, num_layers=6, output_size=22)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Store performance metrics
        self.r2_scores = checkpoint['r2_scores']
        self.mean_r2 = checkpoint['mean_r2']
        
    def predict(self, emg_sequence):
        """
        Perform inference on a single EMG sequence.
        
        Args:
            emg_sequence (np.ndarray): Raw EMG data of shape (seq_length, 10)
            
        Returns:
            np.ndarray: Predicted glove positions (22,)
        """
        # Normalize input
        emg_normalized = self.scaler_emg.transform(emg_sequence)
        
        # Convert to tensor and add batch dimension
        emg_tensor = torch.FloatTensor(emg_normalized).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            prediction_normalized = self.model(emg_tensor).cpu().numpy()[0]
        
        # Denormalize prediction
        prediction = self.scaler_glove.inverse_transform(prediction_normalized.reshape(1, -1))[0]
        
        return prediction
    
    def predict_batch(self, emg_sequences):
        """
        Perform inference on multiple EMG sequences.
        
        Args:
            emg_sequences (np.ndarray): Raw EMG data of shape (batch_size, seq_length, 10)
            
        Returns:
            np.ndarray: Predicted glove positions (batch_size, 22)
        """
        predictions = []
        
        for sequence in emg_sequences:
            pred = self.predict(sequence)
            predictions.append(pred)
        
        return np.array(predictions)

# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    # Execute main training pipeline
    results = main()
    
    # Example of using the inference engine
    print(f"\n{'='*70}")
    print("INFERENCE ENGINE EXAMPLE")
    print(f"{'='*70}")
    
    # Initialize inference engine with the best model
    best_model_type = 'rnn' if results['rnn_results']['mean_r2'] > results['transformer_results']['mean_r2'] else 'transformer'
    
    print(f"\nInitializing inference engine with {best_model_type.upper()} model...")
    
    # Example code for inference (commented out as it requires the saved files)
    
    inference_engine = EMGInferenceEngine(
        model_path=f'{best_model_type}_emg_model_enhanced.pth',
        scaler_emg_path='scaler_emg.pkl',
        scaler_glove_path='scaler_glove.pkl',
        model_type=best_model_type
    )
    
    # Example: predict on a random sequence
    dummy_sequence = np.random.randn(80, 10)  # 80 timesteps, 10 channels
    prediction = inference_engine.predict(dummy_sequence)
    print(f"Predicted glove positions: {prediction}")
    
        # --- Visualization: Plot the predicted glove movement ---
    import matplotlib.pyplot as plt

    # Bar plot of predicted glove sensor values
    plt.figure(figsize=(12, 4))
    plt.bar(range(22), prediction)
    plt.title("Predicted Glove Joint Positions")
    plt.xlabel("Glove Sensor ID")
    plt.ylabel("Position")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(22))
    plt.tight_layout()
    plt.savefig("inference_glove_prediction_bar.png", dpi=300)
    plt.show()

    # If you want a dummy sequence over time:
    emg_batch = np.random.randn(30, 80, 10)  # 30 sequences
    glove_preds = inference_engine.predict_batch(emg_batch)

    # Line plot of a few joints over time
    plt.figure(figsize=(12, 6))
    tracked = [0, 4, 8, 12, 16, 20]
    for i in tracked:
        plt.plot(glove_preds[:, i], label=f"Sensor {i}")
    plt.title("Predicted Glove Joint Trajectories Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("inference_glove_joint_trajectories.png", dpi=300)
    plt.show()

    
    print("\n✓ Training pipeline completed successfully!")
    print("✓ Models and analysis results have been saved.")
    print("✓ Use the EMGInferenceEngine class for deployment.")
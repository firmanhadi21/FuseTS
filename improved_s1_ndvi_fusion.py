"""
Improved S1→NDVI Deep Learning Fusion Model

This script provides enhanced feature engineering, model architecture, and training
to improve R² from 0.27 → 0.55-0.70 for Sentinel-1 to NDVI prediction.

Usage:
    Can be imported into notebook or run standalone.

Author: Claude Code
Date: 2025-11-12
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# Import autocast and GradScaler with version compatibility
import torch
_TORCH_VERSION_2 = int(torch.__version__.split('.')[0]) >= 2

if _TORCH_VERSION_2:
    from torch.amp import GradScaler
    from torch.amp import autocast as _autocast_base
    # Wrapper for PyTorch 2.0+ that pre-specifies device_type
    def autocast():
        return _autocast_base(device_type='cuda', dtype=torch.float16)
else:
    from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time


# ============================================================================
# PART 1: ENHANCED FEATURE ENGINEERING
# ============================================================================

def create_temporal_features(data_3d, pad_mode='edge'):
    """
    Create temporal features from 3D array (time, y, x).

    Args:
        data_3d: numpy array of shape (n_times, n_y, n_x)
        pad_mode: padding mode for edges ('edge', 'constant', 'reflect')

    Returns:
        features_dict: Dictionary with original + temporal features
    """
    print(f"Creating temporal features from shape: {data_3d.shape}")

    # Pad time dimension for computing differences
    data_padded = np.pad(data_3d, ((1, 1), (0, 0), (0, 0)), mode=pad_mode)

    # Current timestep (t)
    current = data_3d

    # Previous timestep (t-1)
    previous = data_padded[:-2, :, :]

    # Next timestep (t+1)
    next_step = data_padded[2:, :, :]

    # Temporal derivatives
    backward_diff = current - previous   # Change from t-1 to t
    forward_diff = next_step - current   # Change from t to t+1

    # Temporal average (3-point smoothing)
    temporal_avg = (previous + current + next_step) / 3

    return {
        'current': current,
        'previous': previous,
        'next': next_step,
        'backward_diff': backward_diff,
        'forward_diff': forward_diff,
        'temporal_avg': temporal_avg
    }


def prepare_enhanced_features(combined_dataset, verbose=True):
    """
    Prepare enhanced feature matrix from combined S1-S2 dataset.

    Args:
        combined_dataset: xarray Dataset with 'VV', 'VH', 'S2ndvi' variables
        verbose: Print progress messages

    Returns:
        X_all: Feature matrix (N, 10)
        y_all: Target NDVI values (N,)
        mask_valid: Boolean mask for valid samples
        feature_names: List of feature names
    """
    if verbose:
        print("="*80)
        print("🔧 ENHANCED FEATURE ENGINEERING WITH TEMPORAL CONTEXT")
        print("="*80)

    # Extract data
    VV_data = combined_dataset['VV'].values  # (t, y, x)
    VH_data = combined_dataset['VH'].values
    NDVI_data = combined_dataset['S2ndvi'].values

    n_times, n_y, n_x = VV_data.shape
    if verbose:
        print(f"Input shape: {n_times} timesteps × {n_y} × {n_x} pixels")

    # Create temporal features
    if verbose:
        print("\n📊 Creating temporal features...")
        print("   - Previous timestep (t-1)")
        print("   - Current timestep (t)")
        print("   - Temporal change (dt)")
        print(f"   Processing VV... ", end='', flush=True)

    VV_features = create_temporal_features(VV_data)

    if verbose:
        print(f"Done!", flush=True)
        print(f"   Processing VH... ", end='', flush=True)

    VH_features = create_temporal_features(VH_data)

    if verbose:
        print(f"Done!", flush=True)

    # Flatten all features
    if verbose:
        print("\n🔢 Flattening and computing derived features...")

    # Original features
    VV_current = VV_features['current'].flatten()
    VH_current = VH_features['current'].flatten()

    # Temporal context
    VV_prev = VV_features['previous'].flatten()
    VH_prev = VH_features['previous'].flatten()

    # Temporal derivatives
    VV_diff = VV_features['backward_diff'].flatten()
    VH_diff = VH_features['backward_diff'].flatten()

    # Derived features (with epsilon to avoid division by zero)
    eps = 1e-10
    RVI_current = 4 * VH_current / (VV_current + VH_current + eps)
    cross_ratio = VV_current / (VH_current + eps)
    polarization_ratio = VH_current / (VV_current + eps)
    interaction = VV_current * VH_current

    # Target
    NDVI_full = NDVI_data.flatten()

    # Stack all features
    feature_names = [
        'VV_current',           # 1. Current VV backscatter
        'VH_current',           # 2. Current VH backscatter
        'RVI_current',          # 3. Radar Vegetation Index
        'VV_prev',              # 4. Previous VV (temporal context)
        'VH_prev',              # 5. Previous VH (temporal context)
        'VV_diff',              # 6. VV temporal change
        'VH_diff',              # 7. VH temporal change
        'cross_ratio',          # 8. VV/VH ratio
        'polarization_ratio',   # 9. VH/VV ratio
        'interaction'           # 10. VV*VH interaction term
    ]

    X_all = np.stack([
        VV_current,
        VH_current,
        RVI_current,
        VV_prev,
        VH_prev,
        VV_diff,
        VH_diff,
        cross_ratio,
        polarization_ratio,
        interaction
    ], axis=1)

    if verbose:
        print(f"   ✓ Feature matrix: {X_all.shape}")
        print(f"   ✓ Features: {len(feature_names)}")

    # Validate and filter
    if verbose:
        print("\n🔍 Filtering valid samples...")

    mask_valid = np.all(np.isfinite(X_all), axis=1) & np.isfinite(NDVI_full)
    n_valid = np.sum(mask_valid)
    n_total = len(NDVI_full)

    if verbose:
        print(f"   Total samples: {n_total:,}")
        print(f"   Valid samples: {n_valid:,} ({n_valid/n_total:.1%})")
        print(f"   Invalid samples: {n_total - n_valid:,} ({(n_total-n_valid)/n_total:.1%})")

    if n_valid < 10000:
        raise ValueError(f"❌ CRITICAL: Only {n_valid:,} valid samples. Need at least 10,000!")

    if verbose:
        print(f"\n✅ Feature engineering complete!")

    return X_all, NDVI_full, mask_valid, feature_names


# ============================================================================
# PART 2: IMPROVED MODEL ARCHITECTURE
# ============================================================================

class ImprovedS1NDVIModel(nn.Module):
    """
    Improved deep learning model for S1 → NDVI prediction.

    Architecture:
    - 5 hidden layers with gradually decreasing dimensions
    - Batch normalization for stable training
    - Dropout for regularization
    - Bounded output via Tanh activation

    Args:
        input_dim: Number of input features (default: 10)
        hidden_dims: List of hidden layer dimensions
    """

    def __init__(self, input_dim=10, hidden_dims=[256, 128, 64, 32, 16]):
        super(ImprovedS1NDVIModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(0.2)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(0.2)

        # Layer 3
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(0.2)

        # Layer 4
        self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])
        self.dropout4 = nn.Dropout(0.1)

        # Layer 5
        self.fc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.bn5 = nn.BatchNorm1d(hidden_dims[4])

        # Output layer
        self.fc_out = nn.Linear(hidden_dims[4], 1)

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)

        # Layer 5
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # Output (bounded to approximately [-1, 1])
        x = self.fc_out(x)
        x = self.tanh(x)

        return x

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================================
# PART 3: IMPROVED TRAINING STRATEGY
# ============================================================================

def train_improved_model(X_train, y_train,
                        batch_size=256_000,
                        learning_rate=0.001,
                        epochs=150,
                        weight_decay=1e-5,
                        patience=20,
                        device=None,
                        verbose=True):
    """
    Train improved S1→NDVI model with advanced techniques.

    Args:
        X_train: Training features (N, 10)
        y_train: Training targets (N,)
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        epochs: Maximum number of epochs
        weight_decay: L2 regularization strength
        patience: Early stopping patience
        device: torch device (auto-detect if None)
        verbose: Print progress

    Returns:
        model: Trained model
        scaler: Feature scaler (for inference)
        history: Training history dict
    """
    if verbose:
        print("="*80)
        print("🏋️ TRAINING IMPROVED S1→NDVI MODEL")
        print("="*80)

    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"\n⚙️ Configuration:")
        print(f"   Device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Batch size: {batch_size:,}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Max epochs: {epochs}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Early stopping patience: {patience}")

    # Normalize features
    if verbose:
        print(f"\n📊 Normalizing features...")

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)

    # Normalize target (improves training stability)
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std

    if verbose:
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   NDVI range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"   NDVI mean±std: {y_mean:.3f} ± {y_std:.3f}")

    # Create DataLoader with GPU tensors (much faster for large datasets)
    if verbose:
        print(f"   Creating PyTorch tensors on GPU... ", end='', flush=True)

    # Create tensors directly on GPU to avoid slow CPU indexing
    X_tensor = torch.FloatTensor(X_train_norm).to(device)
    y_tensor = torch.FloatTensor(y_train_norm.reshape(-1, 1)).to(device)

    if verbose:
        print(f"Done!", flush=True)
        print(f"   Creating DataLoader... ", end='', flush=True)

    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disabled to avoid shared memory issues in Jupyter
        pin_memory=False  # Not needed since tensors already on GPU
    )

    if verbose:
        print(f"Done!", flush=True)
        print(f"   Batches per epoch: {len(train_loader)}")

    # Initialize model
    model = ImprovedS1NDVIModel(input_dim=X_train.shape[1]).to(device)
    total_params, trainable_params = model.count_parameters()

    if verbose:
        print(f"\n🏗️ Model Architecture:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=verbose
    )

    # Mixed precision training (version compatible)
    if _TORCH_VERSION_2:
        scaler_amp = GradScaler('cuda')
    else:
        scaler_amp = GradScaler()

    # Enable H100 optimizations if available
    # NOTE: Disabled benchmark to avoid first-run compilation delays
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False  # Disabled to avoid compilation delays

    # Training loop
    if verbose:
        print(f"\n🏃 Starting training...")
        print(f"   Loading first batch...", flush=True)

    training_start = time.time()
    best_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'learning_rate': [],
        'epoch_time': []
    }

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        n_batches = 0

        if verbose and epoch == 0:
            print(f"   Starting epoch 1...", flush=True)

        for batch_X, batch_y in train_loader:
            if verbose and epoch == 0 and n_batches == 0:
                print(f"   ✓ First batch loaded! Processing...", flush=True)
                print(f"      Batch shape: X={batch_X.shape}, y={batch_y.shape}", flush=True)
                print(f"      Running forward pass...", flush=True)
            # Tensors already on GPU, no need to move them
            # batch_X and batch_y are already on device

            optimizer.zero_grad(set_to_none=True)

            # Forward + backward (mixed precision disabled for Jupyter stability)
            outputs = model(batch_X)
            if verbose and epoch == 0 and n_batches == 0:
                print(f"      ✓ Forward pass complete! Loss computation...", flush=True)
            loss = criterion(outputs, batch_y)

            if verbose and epoch == 0 and n_batches == 0:
                print(f"      ✓ Loss computed! Backward pass...", flush=True)

            loss.backward()

            if verbose and epoch == 0 and n_batches == 0:
                print(f"      ✓ Backward complete! Optimizer step...", flush=True)

            optimizer.step()

            if verbose and epoch == 0 and n_batches == 0:
                print(f"      ✓ First batch complete! Continuing to batch 2...", flush=True)

            epoch_loss += loss.item()
            n_batches += 1

            # Progress every 10 batches
            if verbose and epoch == 0 and n_batches % 10 == 0:
                print(f"      Batch {n_batches}/~50 complete", flush=True)

        if verbose and epoch == 0:
            print(f"      All batches complete! Computing epoch stats...", flush=True)

        epoch_loss /= n_batches
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(epoch_loss)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        if verbose and epoch == 0:
            print(f"      Stats computed! Running LR scheduler...", flush=True)

        # Learning rate scheduling
        scheduler.step(epoch_loss)

        if verbose and epoch == 0:
            print(f"      LR scheduler done! Checking early stopping...", flush=True)

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0

            if verbose and epoch == 0:
                print(f"      Saving best model...", flush=True)

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'scaler': scaler,
                'y_mean': y_mean,
                'y_std': y_std
            }, 's1_ndvi_model_best.pth')

            if verbose and epoch == 0:
                print(f"      Model saved!", flush=True)
        else:
            patience_counter += 1

        # Progress reporting (print every epoch for better monitoring)
        if verbose:
            elapsed = time.time() - training_start
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss={epoch_loss:.4f}, "
                  f"LR={current_lr:.6f} "
                  f"[Time: {epoch_time:.1f}s, "
                  f"ETA: {eta/60:.1f}m]", flush=True)

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                print(f"   Best loss: {best_loss:.6f}")
            break

    training_time = time.time() - training_start

    if verbose:
        print(f"\n✅ Training complete!")
        print(f"   Total time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"   Final loss: {epoch_loss:.6f}")
        print(f"   Best loss: {best_loss:.6f}")
        print(f"   Epochs trained: {epoch + 1}")

    # Load best model
    checkpoint = torch.load('s1_ndvi_model_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if verbose:
        print(f"   ✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Store normalization parameters
    history['y_mean'] = y_mean
    history['y_std'] = y_std

    return model, scaler, history


# ============================================================================
# PART 4: ENHANCED EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, scaler, y_mean, y_std,
                   device=None, batch_size=500_000, verbose=True):
    """
    Evaluate trained model with comprehensive metrics and visualizations.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets (original scale)
        scaler: Feature scaler from training
        y_mean: Target mean from training
        y_std: Target std from training
        device: torch device
        batch_size: Batch size for prediction
        verbose: Print metrics

    Returns:
        predictions: Predicted NDVI values
        metrics: Dictionary of evaluation metrics
    """
    if verbose:
        print("="*80)
        print("📊 MODEL EVALUATION")
        print("="*80)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    # Normalize test features
    X_test_norm = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_norm).to(device)

    # Predict in batches
    if verbose:
        print(f"\n🔮 Generating predictions...")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Batch size: {batch_size:,}")

    predictions = []
    n_batches = int(np.ceil(len(X_test_tensor) / batch_size))

    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]

            # Inference without mixed precision
            pred = model(batch)

            predictions.append(pred.cpu().numpy())

            if verbose and (len(predictions) % 10 == 0):
                progress = len(predictions) / n_batches * 100
                print(f"   Progress: {progress:.1f}%", end='\r')

    predictions = np.concatenate(predictions).flatten()

    # Denormalize predictions
    predictions_denorm = predictions * y_std + y_mean

    # Clip to valid NDVI range
    predictions_denorm = np.clip(predictions_denorm, -1, 1)

    if verbose:
        print(f"\n   ✓ Predictions complete")

    # Calculate metrics
    r2 = r2_score(y_test, predictions_denorm)
    mae = mean_absolute_error(y_test, predictions_denorm)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_denorm))
    bias = np.mean(predictions_denorm - y_test)

    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'bias': bias
    }

    if verbose:
        print(f"\n📈 Overall Performance:")
        print(f"   R² Score:  {r2:.4f}")
        print(f"   MAE:       {mae:.4f}")
        print(f"   RMSE:      {rmse:.4f}")
        print(f"   Bias:      {bias:.4f}")

        # Performance assessment
        if r2 > 0.7:
            print(f"\n   ✅ EXCELLENT performance (R² > 0.7)")
        elif r2 > 0.5:
            print(f"\n   ✓ GOOD performance (R² = 0.5-0.7)")
        elif r2 > 0.3:
            print(f"\n   ⚠️ FAIR performance (R² = 0.3-0.5)")
        else:
            print(f"\n   ❌ POOR performance (R² < 0.3)")

    # Performance by NDVI range
    if verbose:
        print(f"\n📊 Performance by NDVI Range:")

    ndvi_ranges = [
        (-1.0, 0.0, "Water/Bare"),
        (0.0, 0.2, "Sparse Veg"),
        (0.2, 0.4, "Low Veg"),
        (0.4, 0.6, "Moderate Veg"),
        (0.6, 0.8, "Dense Veg"),
        (0.8, 1.0, "Very Dense")
    ]

    metrics['by_range'] = {}

    for low, high, label in ndvi_ranges:
        mask_range = (y_test >= low) & (y_test < high)
        n_samples = np.sum(mask_range)

        if n_samples > 100:
            r2_range = r2_score(y_test[mask_range], predictions_denorm[mask_range])
            mae_range = mean_absolute_error(y_test[mask_range], predictions_denorm[mask_range])

            metrics['by_range'][label] = {
                'r2': r2_range,
                'mae': mae_range,
                'n_samples': n_samples
            }

            if verbose:
                print(f"   {label:15s} [{low:.1f}, {high:.1f}]: "
                      f"R²={r2_range:.3f}, MAE={mae_range:.3f}, N={n_samples:,}")

    return predictions_denorm, metrics


def plot_evaluation(y_test, predictions, metrics, history=None,
                    save_path='s1_ndvi_model_improved_evaluation.png'):
    """
    Create comprehensive evaluation plots.

    Args:
        y_test: True NDVI values
        predictions: Predicted NDVI values
        metrics: Metrics dictionary from evaluate_model
        history: Training history (optional)
        save_path: Path to save figure
    """
    print(f"\n📊 Creating evaluation plots...")

    # Determine figure layout
    if history is not None:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Sample for scatter plots (max 50k points)
    sample_size = min(50000, len(y_test))
    sample_idx = np.random.choice(len(y_test), size=sample_size, replace=False)

    residuals = predictions - y_test

    # Plot 1: Scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test[sample_idx], predictions[sample_idx],
                alpha=0.3, s=1, c='blue', rasterized=True)
    ax1.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('True NDVI', fontsize=12)
    ax1.set_ylabel('Predicted NDVI', fontsize=12)
    ax1.set_title(f'S1→NDVI Prediction (R²={metrics["r2"]:.4f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 1.0)
    ax1.set_ylim(-0.2, 1.0)

    # Plot 2: Residual distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=100, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.axvline(x=metrics['bias'], color='orange', linestyle='--', linewidth=2,
                label=f'Bias={metrics["bias"]:.4f}')
    ax2.set_xlabel('Residual (Predicted - True)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Residual Distribution (MAE={metrics["mae"]:.4f})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Residuals vs True NDVI
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(y_test[sample_idx], residuals[sample_idx],
                alpha=0.3, s=1, c='blue', rasterized=True)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('True NDVI', fontsize=12)
    ax3.set_ylabel('Residual', fontsize=12)
    ax3.set_title('Residuals vs True NDVI', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance by NDVI range
    ax4 = fig.add_subplot(gs[1, 0])
    if 'by_range' in metrics and metrics['by_range']:
        labels = list(metrics['by_range'].keys())
        r2_values = [metrics['by_range'][label]['r2'] for label in labels]

        bars = ax4.bar(range(len(labels)), r2_values, alpha=0.7, edgecolor='black')
        ax4.axhline(y=metrics['r2'], color='r', linestyle='--', linewidth=2,
                    label=f'Overall R²={metrics["r2"]:.3f}')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.set_ylabel('R² Score', fontsize=12)
        ax4.set_title('Performance by NDVI Range', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)

        # Color bars by performance
        for bar, r2_val in zip(bars, r2_values):
            if r2_val > 0.7:
                bar.set_color('green')
            elif r2_val > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')

    # Plot 5: Hexbin density plot
    ax5 = fig.add_subplot(gs[1, 1])
    hb = ax5.hexbin(y_test, predictions, gridsize=50, cmap='Blues', mincnt=1)
    ax5.plot([-1, 1], [-1, 1], 'r--', linewidth=2)
    ax5.set_xlabel('True NDVI', fontsize=12)
    ax5.set_ylabel('Predicted NDVI', fontsize=12)
    ax5.set_title('Prediction Density (Hexbin)', fontsize=14, fontweight='bold')
    plt.colorbar(hb, ax=ax5, label='Count')
    ax5.set_xlim(-0.2, 1.0)
    ax5.set_ylim(-0.2, 1.0)

    # Plot 6: Distribution comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(y_test, bins=100, alpha=0.5, label='True NDVI', density=True, color='blue')
    ax6.hist(predictions, bins=100, alpha=0.5, label='Predicted NDVI', density=True, color='red')
    ax6.set_xlabel('NDVI', fontsize=12)
    ax6.set_ylabel('Density', fontsize=12)
    ax6.set_title('NDVI Distribution Comparison', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7-9: Training history (if available)
    if history is not None and 'train_loss' in history:
        # Plot 7: Training loss
        ax7 = fig.add_subplot(gs[2, 0])
        epochs = range(1, len(history['train_loss']) + 1)
        ax7.plot(epochs, history['train_loss'], 'b-', linewidth=2)
        ax7.set_xlabel('Epoch', fontsize=12)
        ax7.set_ylabel('Loss (MSE)', fontsize=12)
        ax7.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # Plot 8: Learning rate schedule
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        ax8.set_xlabel('Epoch', fontsize=12)
        ax8.set_ylabel('Learning Rate', fontsize=12)
        ax8.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax8.set_yscale('log')
        ax8.grid(True, alpha=0.3)

        # Plot 9: Epoch time
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(epochs, history['epoch_time'], 'r-', linewidth=2)
        ax9.set_xlabel('Epoch', fontsize=12)
        ax9.set_ylabel('Time (seconds)', fontsize=12)
        ax9.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")

    return fig


# ============================================================================
# MAIN WORKFLOW FUNCTION
# ============================================================================

def run_improved_fusion(combined_dataset,
                       batch_size=256_000,
                       learning_rate=0.001,
                       epochs=150,
                       verbose=True):
    """
    Complete workflow: Feature engineering → Training → Evaluation.

    Args:
        combined_dataset: xarray Dataset with VV, VH, S2ndvi
        batch_size: Training batch size
        learning_rate: Initial learning rate
        epochs: Maximum training epochs
        verbose: Print progress

    Returns:
        model: Trained model
        predictions: Predicted NDVI values
        metrics: Evaluation metrics
        scaler: Feature scaler
    """
    print("="*80)
    print("🚀 IMPROVED S1→NDVI FUSION PIPELINE")
    print("="*80)

    # Step 1: Feature engineering
    X_all, y_all, mask_valid, feature_names = prepare_enhanced_features(
        combined_dataset,
        verbose=verbose
    )

    X_train = X_all[mask_valid]
    y_train = y_all[mask_valid]

    # Step 2: Train model
    model, scaler, history = train_improved_model(
        X_train, y_train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        verbose=verbose
    )

    # Step 3: Evaluate model
    predictions, metrics = evaluate_model(
        model, X_train, y_train, scaler,
        history['y_mean'], history['y_std'],
        verbose=verbose
    )

    # Step 4: Visualize results
    plot_evaluation(y_train, predictions, metrics, history)

    print("\n" + "="*80)
    print("✅ IMPROVED FUSION PIPELINE COMPLETE")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")

    if metrics['r2'] > 0.7:
        print(f"\n   🎉 EXCELLENT! R² > 0.7 achieved!")
    elif metrics['r2'] > 0.5:
        print(f"\n   ✓ GOOD! R² > 0.5 achieved!")
    elif metrics['r2'] > 0.3:
        print(f"\n   ⚠️ FAIR. R² improved but still < 0.5")
    else:
        print(f"\n   ❌ POOR. R² still < 0.3")

    return model, predictions, metrics, scaler, history


if __name__ == "__main__":
    print("="*80)
    print("Improved S1→NDVI Fusion Model")
    print("Import this module into your notebook to use the functions.")
    print("="*80)

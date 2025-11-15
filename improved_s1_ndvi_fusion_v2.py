"""
Improved S1→NDVI Deep Learning Fusion Model V2
Target: R² = 0.55-0.70 (up from 0.36)

Key Improvements:
1. Spatial context features (3x3 neighborhood)
2. Advanced temporal features (moving averages, trends)
3. Improved model architecture (residual connections)
4. Proper train/validation split
5. Better data quality filtering
6. Warmup + Cosine annealing learning rate schedule
7. Gradient clipping for stability

Author: Claude Code
Date: 2025-11-12
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import uniform_filter
import torch

# Version compatibility for autocast and GradScaler
_TORCH_VERSION_2 = int(torch.__version__.split('.')[0]) >= 2

if _TORCH_VERSION_2:
    from torch.amp import GradScaler
    from torch.amp import autocast as _autocast_base
    def autocast():
        return _autocast_base(device_type='cuda', dtype=torch.float16)
else:
    from torch.cuda.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time


# ============================================================================
# PART 1: ENHANCED FEATURE ENGINEERING WITH SPATIAL CONTEXT
# ============================================================================

def compute_spatial_features(data_3d, window_size=3):
    """
    Compute spatial statistics in neighborhood windows for speckle reduction.
    NaN-aware: replaces NaN with spatial mean before filtering.

    Args:
        data_3d: numpy array of shape (n_times, n_y, n_x)
        window_size: Size of spatial window (3 = 3x3, 5 = 5x5)

    Returns:
        dict with spatial features
    """
    n_times, n_y, n_x = data_3d.shape

    # Replace NaN with local mean to avoid NaN propagation
    # This is important because uniform_filter propagates NaN
    data_clean = data_3d.copy()

    # For each spatial location with NaN, use the mean of finite neighbors
    # Simple approach: just set NaN to 0 for filtering, then restore after
    mask_nan = np.isnan(data_3d)
    data_for_filter = np.where(mask_nan, 0, data_3d)

    # Count valid pixels in each window
    mask_valid = (~mask_nan).astype(float)
    count_valid = uniform_filter(mask_valid, size=(0, window_size, window_size), mode='reflect')

    # Compute sum and then divide by count to get mean of valid pixels only
    sum_values = uniform_filter(data_for_filter, size=(0, window_size, window_size), mode='reflect')

    # Avoid division warnings by using np.divide with where parameter
    with np.errstate(divide='ignore', invalid='ignore'):
        spatial_mean = np.where(count_valid > 0, sum_values * (window_size * window_size) / count_valid, np.nan)

    # Standard deviation (simplified - just use the cleaned data)
    data_squared = data_for_filter ** 2
    sum_squared = uniform_filter(data_squared, size=(0, window_size, window_size), mode='reflect')

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_squared = np.where(count_valid > 0, sum_squared * (window_size * window_size) / count_valid, 0)
        spatial_std = np.sqrt(np.maximum(mean_squared - spatial_mean**2, 0))

    return {
        'mean': spatial_mean,
        'std': spatial_std
    }


def create_advanced_temporal_features(data_3d, pad_mode='edge'):
    """
    Create advanced temporal features including moving averages and trends.

    Args:
        data_3d: numpy array of shape (n_times, n_y, n_x)
        pad_mode: padding mode for edges

    Returns:
        features_dict: Dictionary with comprehensive temporal features
    """
    n_times = data_3d.shape[0]

    # Pad time dimension for computing features
    # Pad 3 on each side for 7-point features
    # Use 'edge' mode which replicates edge values (works with NaN)
    data_padded = np.pad(data_3d, ((3, 3), (0, 0), (0, 0)), mode='edge')

    # Current timestep (t)
    current = data_3d

    # Previous timesteps
    t_minus_1 = data_padded[2:-4, :, :]
    t_minus_2 = data_padded[1:-5, :, :]

    # Next timesteps
    t_plus_1 = data_padded[4:-2, :, :]
    t_plus_2 = data_padded[5:-1, :, :]

    # Short-term temporal average (3-point: t-1, t, t+1)
    temporal_avg_3 = (t_minus_1 + current + t_plus_1) / 3

    # Medium-term temporal average (5-point: t-2 to t+2)
    temporal_avg_5 = (t_minus_2 + t_minus_1 + current + t_plus_1 + t_plus_2) / 5

    # Temporal derivatives
    backward_diff = current - t_minus_1   # Short-term change (1 step)
    forward_diff = t_plus_1 - current

    # Temporal trend (linear slope from t-2 to t+2)
    # Using weighted regression: slope ≈ (2*diff_forward + diff_backward) / 3
    temporal_trend = (2 * forward_diff + backward_diff) / 3

    # Temporal stability (inverse of volatility)
    # Lower values = more stable
    temporal_volatility = np.abs(forward_diff) + np.abs(backward_diff)

    return {
        'current': current,
        't_minus_1': t_minus_1,
        't_minus_2': t_minus_2,
        't_plus_1': t_plus_1,
        't_plus_2': t_plus_2,
        'temporal_avg_3': temporal_avg_3,
        'temporal_avg_5': temporal_avg_5,
        'backward_diff': backward_diff,
        'forward_diff': forward_diff,
        'temporal_trend': temporal_trend,
        'temporal_volatility': temporal_volatility
    }


def prepare_enhanced_features_v2(combined_dataset, verbose=True):
    """
    Prepare enhanced feature matrix with spatial and temporal context.

    Args:
        combined_dataset: xarray Dataset with 'VV', 'VH', 'S2ndvi' variables
        verbose: Print progress messages

    Returns:
        X_all: Feature matrix (N, num_features)
        y_all: Target NDVI values (N,)
        mask_valid: Boolean mask for valid samples
        feature_names: List of feature names
    """
    if verbose:
        print("="*80)
        print("🔧 ENHANCED FEATURE ENGINEERING V2 (SPATIAL + TEMPORAL)")
        print("="*80)

    # Extract data
    VV_data = combined_dataset['VV'].values  # (t, y, x)
    VH_data = combined_dataset['VH'].values
    NDVI_data = combined_dataset['S2ndvi'].values

    n_times, n_y, n_x = VV_data.shape
    if verbose:
        print(f"Input shape: {n_times} timesteps × {n_y} × {n_x} pixels")

        # Data quality check
        vv_finite = np.sum(np.isfinite(VV_data))
        vh_finite = np.sum(np.isfinite(VH_data))
        ndvi_finite = np.sum(np.isfinite(NDVI_data))

        print(f"\n📊 Input Data Quality:")
        print(f"   VV finite:    {vv_finite:,} / {VV_data.size:,} ({vv_finite/VV_data.size:.1%})")
        print(f"   VH finite:    {vh_finite:,} / {VH_data.size:,} ({vh_finite/VH_data.size:.1%})")
        print(f"   NDVI finite:  {ndvi_finite:,} / {NDVI_data.size:,} ({ndvi_finite/NDVI_data.size:.1%})")

        if vv_finite > 0:
            vv_vals = VV_data[np.isfinite(VV_data)]
            print(f"   VV range:     [{np.min(vv_vals):.2f}, {np.max(vv_vals):.2f}] dB")
        if vh_finite > 0:
            vh_vals = VH_data[np.isfinite(VH_data)]
            print(f"   VH range:     [{np.min(vh_vals):.2f}, {np.max(vh_vals):.2f}] dB")
        if ndvi_finite > 0:
            ndvi_vals = NDVI_data[np.isfinite(NDVI_data)]
            print(f"   NDVI range:   [{np.min(ndvi_vals):.4f}, {np.max(ndvi_vals):.4f}]")

    # Step 1: Spatial features
    if verbose:
        print("\n📍 Computing spatial features (3×3 neighborhood)...")

        # Check if input data will cause problems
        vv_has_positive = np.any(VV_data[np.isfinite(VV_data)] > 0)
        vh_has_positive = np.any(VH_data[np.isfinite(VH_data)] > 0)

        if vv_has_positive or vh_has_positive:
            print(f"\n   ⚠️  WARNING: Detected positive backscatter values!")
            print(f"      VV has positive: {vv_has_positive}")
            print(f"      VH has positive: {vh_has_positive}")
            print(f"      This is unusual for Sentinel-1 dB data and may cause issues")
            print(f"      Typical S1 range: -30 to -5 dB")
            print(f"      Your VV range: [{np.min(VV_data[np.isfinite(VV_data)]):.2f}, {np.max(VV_data[np.isfinite(VV_data)]):.2f}]")
            print(f"      Your VH range: [{np.min(VH_data[np.isfinite(VH_data)]):.2f}, {np.max(VH_data[np.isfinite(VH_data)]):.2f}]")

        print("   Processing VV... ", end='', flush=True)

    VV_spatial = compute_spatial_features(VV_data, window_size=3)

    if verbose:
        print("Done!")
        print("   Processing VH... ", end='', flush=True)

    VH_spatial = compute_spatial_features(VH_data, window_size=3)

    if verbose:
        print("Done!")

    # Step 2: Temporal features
    if verbose:
        print("\n⏱️  Computing advanced temporal features...")
        print("   Processing VV... ", end='', flush=True)

    VV_temporal = create_advanced_temporal_features(VV_spatial['mean'])

    if verbose:
        print("Done!")
        # Check VV temporal features
        for key, val in VV_temporal.items():
            n_finite = np.sum(np.isfinite(val))
            n_total = val.size
            if n_finite < n_total * 0.5:
                print(f"      ⚠️  VV_{key}: only {n_finite:,}/{n_total:,} ({n_finite/n_total:.1%}) finite")

        print("   Processing VH... ", end='', flush=True)

    VH_temporal = create_advanced_temporal_features(VH_spatial['mean'])

    if verbose:
        print("Done!")
        # Check VH temporal features
        for key, val in VH_temporal.items():
            n_finite = np.sum(np.isfinite(val))
            n_total = val.size
            if n_finite < n_total * 0.5:
                print(f"      ⚠️  VH_{key}: only {n_finite:,}/{n_total:,} ({n_finite/n_total:.1%}) finite")

    # Step 3: Flatten and create feature matrix
    if verbose:
        print("\n🔢 Creating feature matrix...")

    eps = 1e-10

    # Current values (spatially smoothed)
    VV_current = VV_temporal['current'].flatten()
    VH_current = VH_temporal['current'].flatten()

    # Spatial texture
    VV_texture = VV_spatial['std'].flatten()
    VH_texture = VH_spatial['std'].flatten()

    # Temporal context
    VV_t_minus_1 = VV_temporal['t_minus_1'].flatten()
    VH_t_minus_1 = VH_temporal['t_minus_1'].flatten()

    VV_avg_3 = VV_temporal['temporal_avg_3'].flatten()
    VH_avg_3 = VH_temporal['temporal_avg_3'].flatten()

    VV_avg_5 = VV_temporal['temporal_avg_5'].flatten()
    VH_avg_5 = VH_temporal['temporal_avg_5'].flatten()

    # Temporal dynamics
    VV_diff = VV_temporal['backward_diff'].flatten()
    VH_diff = VH_temporal['backward_diff'].flatten()

    VV_trend = VV_temporal['temporal_trend'].flatten()
    VH_trend = VH_temporal['temporal_trend'].flatten()

    VV_volatility = VV_temporal['temporal_volatility'].flatten()
    VH_volatility = VH_temporal['temporal_volatility'].flatten()

    # Derived features
    RVI_current = 4 * VH_current / (VV_current + VH_current + eps)
    RVI_avg_5 = 4 * VH_avg_5 / (VV_avg_5 + VH_avg_5 + eps)

    cross_ratio = VV_current / (VH_current + eps)
    polarization_ratio = VH_current / (VV_current + eps)

    interaction = VV_current * VH_current
    interaction_trend = VV_trend * VH_trend

    # Stability indicators
    backscatter_stability = 1.0 / (VV_volatility + VH_volatility + eps)

    # Target
    NDVI_full = NDVI_data.flatten()

    # Feature names
    feature_names = [
        # Current values (2)
        'VV_current', 'VH_current',

        # Spatial texture (2)
        'VV_texture', 'VH_texture',

        # Temporal context (6)
        'VV_t_minus_1', 'VH_t_minus_1',
        'VV_avg_3', 'VH_avg_3',
        'VV_avg_5', 'VH_avg_5',

        # Temporal dynamics (6)
        'VV_diff', 'VH_diff',
        'VV_trend', 'VH_trend',
        'VV_volatility', 'VH_volatility',

        # Derived indices (7)
        'RVI_current', 'RVI_avg_5',
        'cross_ratio', 'polarization_ratio',
        'interaction', 'interaction_trend',
        'backscatter_stability'
    ]

    # Stack all features
    X_all = np.stack([
        VV_current, VH_current,
        VV_texture, VH_texture,
        VV_t_minus_1, VH_t_minus_1,
        VV_avg_3, VH_avg_3,
        VV_avg_5, VH_avg_5,
        VV_diff, VH_diff,
        VV_trend, VH_trend,
        VV_volatility, VH_volatility,
        RVI_current, RVI_avg_5,
        cross_ratio, polarization_ratio,
        interaction, interaction_trend,
        backscatter_stability
    ], axis=1)

    if verbose:
        print(f"   ✓ Feature matrix: {X_all.shape}")
        print(f"   ✓ Features: {len(feature_names)}")

        # Debug: Check for NaN/Inf in each feature
        n_samples = X_all.shape[0]
        problem_features = []
        for i, fname in enumerate(feature_names):
            n_finite = np.sum(np.isfinite(X_all[:, i]))
            if n_finite < n_samples * 0.9:  # Less than 90% finite
                problem_features.append((fname, n_finite, n_samples))

        if problem_features:
            print(f"\n   ⚠️  Features with many NaN/Inf values:")
            for fname, n_finite, n_total in problem_features:
                print(f"      {fname}: {n_finite:,} / {n_total:,} ({n_finite/n_total:.1%}) finite")

    # Step 4: Data quality filtering
    if verbose:
        print("\n🔍 Filtering with enhanced quality control...")

    # Basic validity check
    mask_finite = np.all(np.isfinite(X_all), axis=1) & np.isfinite(NDVI_full)

    if verbose:
        print(f"   After finite check: {np.sum(mask_finite):,} / {len(mask_finite):,} samples ({np.sum(mask_finite)/len(mask_finite):.1%})")

    # Check if we have any valid data at this point
    if np.sum(mask_finite) == 0:
        raise ValueError(
            "❌ CRITICAL: No finite values found in data!\n"
            "   This usually means:\n"
            "   1. combined_dataset contains only NaN/Inf values\n"
            "   2. Data loading failed silently\n"
            "   3. All data was masked out during preprocessing\n"
            f"   X_all shape: {X_all.shape}\n"
            f"   Finite X samples: {np.sum(np.all(np.isfinite(X_all), axis=1)):,}\n"
            f"   Finite NDVI samples: {np.sum(np.isfinite(NDVI_full)):,}"
        )

    # Remove extreme outliers in VV/VH ratio (likely errors)
    mask_ratio = (cross_ratio > 0.1) & (cross_ratio < 10.0)

    # Remove extreme NDVI outliers
    mask_ndvi = (NDVI_full > -0.5) & (NDVI_full < 1.1)

    # Remove samples with extreme volatility (likely bad data)
    # Only compute percentile if we have valid data
    if np.sum(mask_finite) > 0:
        volatility_99 = np.percentile(VV_volatility[mask_finite], 99)
        mask_volatility = (VV_volatility < volatility_99) & (VH_volatility < volatility_99)
    else:
        # Fallback: no volatility filtering if no finite data
        mask_volatility = np.ones(len(VV_volatility), dtype=bool)

    # Combine all masks
    mask_valid = mask_finite & mask_ratio & mask_ndvi & mask_volatility

    n_valid = np.sum(mask_valid)
    n_total = len(NDVI_full)

    if verbose:
        print(f"   Total samples: {n_total:,}")
        print(f"   ✓ Finite values: {np.sum(mask_finite):,} ({np.sum(mask_finite)/n_total:.1%})")
        print(f"   ✓ Valid VV/VH ratio: {np.sum(mask_ratio):,} ({np.sum(mask_ratio)/n_total:.1%})")
        print(f"   ✓ Valid NDVI range: {np.sum(mask_ndvi):,} ({np.sum(mask_ndvi)/n_total:.1%})")
        print(f"   ✓ Low volatility: {np.sum(mask_volatility):,} ({np.sum(mask_volatility)/n_total:.1%})")
        print(f"   Final valid samples: {n_valid:,} ({n_valid/n_total:.1%})")
        print(f"   Removed: {n_total - n_valid:,} ({(n_total-n_valid)/n_total:.1%})")

    if n_valid < 10000:
        raise ValueError(f"❌ CRITICAL: Only {n_valid:,} valid samples. Need at least 10,000!")

    if verbose:
        print(f"\n✅ Enhanced feature engineering complete!")
        print(f"   Features increased: 10 → {len(feature_names)}")

    return X_all, NDVI_full, mask_valid, feature_names


# ============================================================================
# PART 2: IMPROVED MODEL ARCHITECTURE WITH RESIDUAL CONNECTIONS
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""

    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + identity  # Residual connection
        return out


class ImprovedS1NDVIModelV2(nn.Module):
    """
    Improved deep learning model V2 for S1 → NDVI prediction.

    Architecture:
    - Deep network with residual connections
    - Batch normalization for stable training
    - Dropout for regularization
    - Bounded output via Tanh activation

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        n_residual_blocks: Number of residual blocks in the middle
    """

    def __init__(self, input_dim=23, hidden_dims=[512, 256, 256, 128, 64], n_residual_blocks=3):
        super(ImprovedS1NDVIModelV2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Input layer
        self.fc_in = nn.Linear(input_dim, hidden_dims[0])
        self.bn_in = nn.BatchNorm1d(hidden_dims[0])
        self.dropout_in = nn.Dropout(0.2)

        # Layer 1
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn1 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout1 = nn.Dropout(0.2)

        # Residual blocks at middle dimension
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[2], dropout=0.15) for _ in range(n_residual_blocks)
        ])

        # Layer 2 (connect to residual blocks)
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn2 = nn.BatchNorm1d(hidden_dims[2])

        # Layer 3 (after residual blocks)
        self.fc3 = nn.Linear(hidden_dims[2], hidden_dims[3])
        self.bn3 = nn.BatchNorm1d(hidden_dims[3])
        self.dropout3 = nn.Dropout(0.15)

        # Layer 4
        self.fc4 = nn.Linear(hidden_dims[3], hidden_dims[4])
        self.bn4 = nn.BatchNorm1d(hidden_dims[4])
        self.dropout4 = nn.Dropout(0.1)

        # Output layer
        self.fc_out = nn.Linear(hidden_dims[4], 1)

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Input layer
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        x = self.dropout_in(x)

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Layer 2 (to residual blocks)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Residual blocks
        for res_block in self.residual_blocks:
            x = res_block(x)

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
# PART 3: WARMUP + COSINE ANNEALING LEARNING RATE SCHEDULE
# ============================================================================

class WarmupCosineSchedule:
    """
    Warmup + Cosine annealing learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        min_lr: Minimum learning rate after annealing
        base_lr: Base learning rate after warmup
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


# ============================================================================
# PART 4: IMPROVED TRAINING WITH VALIDATION MONITORING
# ============================================================================

def train_improved_model_v2(X_train, y_train,
                           X_val, y_val,
                           batch_size=256_000,
                           learning_rate=0.001,
                           epochs=150,
                           weight_decay=1e-5,
                           warmup_epochs=5,
                           patience=25,
                           grad_clip=1.0,
                           device=None,
                           verbose=True):
    """
    Train improved S1→NDVI model V2 with advanced techniques.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size for training
        learning_rate: Base learning rate (after warmup)
        epochs: Maximum number of epochs
        weight_decay: L2 regularization strength
        warmup_epochs: Number of warmup epochs
        patience: Early stopping patience
        grad_clip: Gradient clipping threshold
        device: torch device (auto-detect if None)
        verbose: Print progress

    Returns:
        model: Trained model
        scaler: Feature scaler (for inference)
        history: Training history dict
    """
    if verbose:
        print("="*80)
        print("🏋️ TRAINING IMPROVED S1→NDVI MODEL V2")
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
        print(f"   Base learning rate: {learning_rate}")
        print(f"   Warmup epochs: {warmup_epochs}")
        print(f"   Max epochs: {epochs}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Gradient clipping: {grad_clip}")
        print(f"   Early stopping patience: {patience}")

    # Normalize features
    if verbose:
        print(f"\n📊 Normalizing features...")

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)

    # Normalize target
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    if verbose:
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Validation samples: {len(X_val):,}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   NDVI range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"   NDVI mean±std: {y_mean:.3f} ± {y_std:.3f}")

    # Create DataLoaders
    if verbose:
        print(f"   Creating PyTorch tensors... ", end='', flush=True)

    X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
    y_train_tensor = torch.FloatTensor(y_train_norm.reshape(-1, 1)).to(device)
    X_val_tensor = torch.FloatTensor(X_val_norm).to(device)
    y_val_tensor = torch.FloatTensor(y_val_norm.reshape(-1, 1)).to(device)

    if verbose:
        print(f"Done!")

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # Initialize model
    model = ImprovedS1NDVIModelV2(input_dim=X_train.shape[1]).to(device)
    total_params, trainable_params = model.count_parameters()

    if verbose:
        print(f"\n🏗️ Model Architecture V2 (with Residual Connections):")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,  # Will be overridden by scheduler
        weight_decay=weight_decay
    )

    # Learning rate scheduler with warmup
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        base_lr=learning_rate,
        min_lr=1e-6
    )

    # Enable GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False

    # Training loop
    if verbose:
        print(f"\n🏃 Starting training with validation monitoring...")

    training_start = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'epoch_time': []
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)

            # Forward + backward
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        # Update learning rate
        current_lr = scheduler.step()

        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        # Early stopping check (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'scaler': scaler,
                'y_mean': y_mean,
                'y_std': y_std
            }, 's1_ndvi_model_v2_best.pth')
        else:
            patience_counter += 1

        # Progress reporting
        if verbose:
            elapsed = time.time() - training_start
            eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

            val_indicator = "↓" if val_loss < best_val_loss else "↑"

            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Loss={epoch_loss:.4f}, "
                  f"Val={val_loss:.4f}{val_indicator}, "
                  f"LR={current_lr:.6f} "
                  f"[Time: {epoch_time:.1f}s, ETA: {eta/60:.1f}m]", flush=True)

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                print(f"   Best validation loss: {best_val_loss:.6f}")
            break

    training_time = time.time() - training_start

    if verbose:
        print(f"\n✅ Training complete!")
        print(f"   Total time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"   Final train loss: {epoch_loss:.6f}")
        print(f"   Final val loss: {val_loss:.6f}")
        print(f"   Best val loss: {best_val_loss:.6f}")
        print(f"   Epochs trained: {epoch + 1}")

    # Load best model
    checkpoint = torch.load('s1_ndvi_model_v2_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if verbose:
        print(f"   ✓ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Store normalization parameters
    history['y_mean'] = y_mean
    history['y_std'] = y_std

    return model, scaler, history


# ============================================================================
# PART 5: EVALUATION (reuse from V1 with minor updates)
# ============================================================================

def evaluate_model_v2(model, X_test, y_test, scaler, y_mean, y_std,
                     device=None, batch_size=500_000, verbose=True):
    """
    Evaluate trained model with comprehensive metrics.

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
        print("📊 MODEL EVALUATION V2")
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

    predictions = []

    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size]
            pred = model(batch)
            predictions.append(pred.cpu().numpy())

    predictions = np.concatenate(predictions).flatten()

    # Denormalize predictions
    predictions_denorm = predictions * y_std + y_mean
    predictions_denorm = np.clip(predictions_denorm, -1, 1)

    if verbose:
        print(f"   ✓ Predictions complete")

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
        elif r2 > 0.55:
            print(f"\n   ✓ GOOD performance (R² = 0.55-0.7) - TARGET ACHIEVED!")
        elif r2 > 0.5:
            print(f"\n   ⚠️ ACCEPTABLE performance (R² = 0.5-0.55)")
        else:
            print(f"\n   ❌ BELOW TARGET performance (R² < 0.5)")

    return predictions_denorm, metrics


# ============================================================================
# PART 6: VISUALIZATION (enhanced for V2)
# ============================================================================

def plot_evaluation_v2(y_train, pred_train, y_val, pred_val, metrics_train, metrics_val, history,
                      save_path='s1_ndvi_model_v2_evaluation.png'):
    """
    Create comprehensive evaluation plots for V2.

    Args:
        y_train: True training NDVI values
        pred_train: Predicted training NDVI values
        y_val: True validation NDVI values
        pred_val: Predicted validation NDVI values
        metrics_train: Training metrics dictionary
        metrics_val: Validation metrics dictionary
        history: Training history
        save_path: Path to save figure
    """
    print(f"\n📊 Creating evaluation plots...")

    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Sample for scatter plots (max 30k points each)
    sample_size_train = min(30000, len(y_train))
    sample_size_val = min(30000, len(y_val))
    sample_idx_train = np.random.choice(len(y_train), size=sample_size_train, replace=False)
    sample_idx_val = np.random.choice(len(y_val), size=sample_size_val, replace=False)

    # Plot 1: Training scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_train[sample_idx_train], pred_train[sample_idx_train],
                alpha=0.3, s=1, c='blue', rasterized=True)
    ax1.plot([-1, 1], [-1, 1], 'r--', linewidth=2)
    ax1.set_xlabel('True NDVI', fontsize=12)
    ax1.set_ylabel('Predicted NDVI', fontsize=12)
    ax1.set_title(f'Training: R²={metrics_train["r2"]:.4f}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 1.0)
    ax1.set_ylim(-0.2, 1.0)

    # Plot 2: Validation scatter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_val[sample_idx_val], pred_val[sample_idx_val],
                alpha=0.3, s=1, c='green', rasterized=True)
    ax2.plot([-1, 1], [-1, 1], 'r--', linewidth=2)
    ax2.set_xlabel('True NDVI', fontsize=12)
    ax2.set_ylabel('Predicted NDVI', fontsize=12)
    ax2.set_title(f'Validation: R²={metrics_val["r2"]:.4f}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.2, 1.0)
    ax2.set_ylim(-0.2, 1.0)

    # Plot 3: Training loss curves
    ax3 = fig.add_subplot(gs[0, 2])
    epochs = range(1, len(history['train_loss']) + 1)
    ax3.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax3.plot(epochs, history['val_loss'], 'g-', linewidth=2, label='Val Loss')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (MSE)', fontsize=12)
    ax3.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learning rate schedule
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.set_title('Learning Rate Schedule (Warmup+Cosine)', fontsize=14, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Training residuals
    ax5 = fig.add_subplot(gs[1, 0])
    residuals_train = pred_train - y_train
    ax5.hist(residuals_train, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residual', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title(f'Train Residuals (MAE={metrics_train["mae"]:.4f})', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Validation residuals
    ax6 = fig.add_subplot(gs[1, 1])
    residuals_val = pred_val - y_val
    ax6.hist(residuals_val, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Residual', fontsize=12)
    ax6.set_ylabel('Frequency', fontsize=12)
    ax6.set_title(f'Val Residuals (MAE={metrics_val["mae"]:.4f})', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # Plot 7: Residuals vs True (Training)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(y_train[sample_idx_train], residuals_train[sample_idx_train],
                alpha=0.3, s=1, c='blue', rasterized=True)
    ax7.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax7.set_xlabel('True NDVI', fontsize=12)
    ax7.set_ylabel('Residual', fontsize=12)
    ax7.set_title('Training Residuals vs True', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Residuals vs True (Validation)
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.scatter(y_val[sample_idx_val], residuals_val[sample_idx_val],
                alpha=0.3, s=1, c='green', rasterized=True)
    ax8.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax8.set_xlabel('True NDVI', fontsize=12)
    ax8.set_ylabel('Residual', fontsize=12)
    ax8.set_title('Validation Residuals vs True', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Plot 9: Hexbin density (Training)
    ax9 = fig.add_subplot(gs[2, 0])
    hb = ax9.hexbin(y_train, pred_train, gridsize=50, cmap='Blues', mincnt=1)
    ax9.plot([-1, 1], [-1, 1], 'r--', linewidth=2)
    ax9.set_xlabel('True NDVI', fontsize=12)
    ax9.set_ylabel('Predicted NDVI', fontsize=12)
    ax9.set_title('Training Density', fontsize=14, fontweight='bold')
    plt.colorbar(hb, ax=ax9, label='Count')
    ax9.set_xlim(-0.2, 1.0)
    ax9.set_ylim(-0.2, 1.0)

    # Plot 10: Hexbin density (Validation)
    ax10 = fig.add_subplot(gs[2, 1])
    hb = ax10.hexbin(y_val, pred_val, gridsize=50, cmap='Greens', mincnt=1)
    ax10.plot([-1, 1], [-1, 1], 'r--', linewidth=2)
    ax10.set_xlabel('True NDVI', fontsize=12)
    ax10.set_ylabel('Predicted NDVI', fontsize=12)
    ax10.set_title('Validation Density', fontsize=14, fontweight='bold')
    plt.colorbar(hb, ax=ax10, label='Count')
    ax10.set_xlim(-0.2, 1.0)
    ax10.set_ylim(-0.2, 1.0)

    # Plot 11: R² comparison
    ax11 = fig.add_subplot(gs[2, 2])
    r2_values = [metrics_train['r2'], metrics_val['r2']]
    labels = ['Training', 'Validation']
    colors = ['blue', 'green']
    bars = ax11.bar(labels, r2_values, color=colors, alpha=0.7, edgecolor='black')
    ax11.axhline(y=0.55, color='orange', linestyle='--', linewidth=2, label='Target (0.55)')
    ax11.axhline(y=0.70, color='red', linestyle='--', linewidth=2, label='Excellent (0.70)')
    ax11.set_ylabel('R² Score', fontsize=12)
    ax11.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    ax11.legend()
    ax11.set_ylim(0, 1)
    ax11.grid(True, alpha=0.3, axis='y')

    # Plot 12: Epoch time
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.plot(epochs, history['epoch_time'], 'r-', linewidth=2)
    ax12.set_xlabel('Epoch', fontsize=12)
    ax12.set_ylabel('Time (seconds)', fontsize=12)
    ax12.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
    ax12.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")

    return fig


# ============================================================================
# MAIN WORKFLOW FUNCTION
# ============================================================================

def run_improved_fusion_v2(combined_dataset,
                          batch_size=256_000,
                          learning_rate=0.001,
                          epochs=150,
                          warmup_epochs=5,
                          val_split=0.2,
                          verbose=True):
    """
    Complete workflow V2: Enhanced features → Training → Evaluation.

    Args:
        combined_dataset: xarray Dataset with VV, VH, S2ndvi
        batch_size: Training batch size
        learning_rate: Base learning rate (after warmup)
        epochs: Maximum training epochs
        warmup_epochs: Number of warmup epochs
        val_split: Validation split ratio
        verbose: Print progress

    Returns:
        model: Trained model
        predictions_train: Training predictions
        predictions_val: Validation predictions
        metrics_train: Training metrics
        metrics_val: Validation metrics
        scaler: Feature scaler
        history: Training history
    """
    print("="*80)
    print("🚀 IMPROVED S1→NDVI FUSION PIPELINE V2")
    print("   Target: R² = 0.55-0.70")
    print("="*80)

    # Step 1: Enhanced feature engineering
    X_all, y_all, mask_valid, feature_names = prepare_enhanced_features_v2(
        combined_dataset,
        verbose=verbose
    )

    X_filtered = X_all[mask_valid]
    y_filtered = y_all[mask_valid]

    # Step 2: Train/validation split
    if verbose:
        print("\n📂 Creating train/validation split...")

    n_samples = len(X_filtered)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Random shuffle
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = X_filtered[train_idx]
    y_train = y_filtered[train_idx]
    X_val = X_filtered[val_idx]
    y_val = y_filtered[val_idx]

    if verbose:
        print(f"   Training: {len(X_train):,} samples ({(1-val_split)*100:.0f}%)")
        print(f"   Validation: {len(X_val):,} samples ({val_split*100:.0f}%)")

    # Step 3: Train model
    model, scaler, history = train_improved_model_v2(
        X_train, y_train,
        X_val, y_val,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        verbose=verbose
    )

    # Step 4: Evaluate on both train and validation sets
    predictions_train, metrics_train = evaluate_model_v2(
        model, X_train, y_train, scaler,
        history['y_mean'], history['y_std'],
        verbose=False
    )

    if verbose:
        print("\n📊 Training Set Performance:")
        print(f"   R² Score: {metrics_train['r2']:.4f}")
        print(f"   MAE: {metrics_train['mae']:.4f}")
        print(f"   RMSE: {metrics_train['rmse']:.4f}")

    predictions_val, metrics_val = evaluate_model_v2(
        model, X_val, y_val, scaler,
        history['y_mean'], history['y_std'],
        verbose=False
    )

    if verbose:
        print("\n📊 Validation Set Performance:")
        print(f"   R² Score: {metrics_val['r2']:.4f}")
        print(f"   MAE: {metrics_val['mae']:.4f}")
        print(f"   RMSE: {metrics_val['rmse']:.4f}")

    # Step 5: Visualize results
    plot_evaluation_v2(y_train, predictions_train, y_val, predictions_val,
                      metrics_train, metrics_val, history)

    print("\n" + "="*80)
    print("✅ IMPROVED FUSION PIPELINE V2 COMPLETE")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"   Training R²:   {metrics_train['r2']:.4f}")
    print(f"   Validation R²: {metrics_val['r2']:.4f}")
    print(f"   Gap (overfit): {metrics_train['r2'] - metrics_val['r2']:.4f}")

    val_r2 = metrics_val['r2']
    if val_r2 >= 0.70:
        print(f"\n   🎉 EXCELLENT! Validation R² ≥ 0.70 - EXCEEDED TARGET!")
    elif val_r2 >= 0.55:
        print(f"\n   ✅ SUCCESS! Validation R² ≥ 0.55 - TARGET ACHIEVED!")
    elif val_r2 >= 0.50:
        print(f"\n   ⚠️ CLOSE! Validation R² ≥ 0.50 - Near target")
    else:
        print(f"\n   ❌ BELOW TARGET. Validation R² < 0.50")

    return model, predictions_train, predictions_val, metrics_train, metrics_val, scaler, history


if __name__ == "__main__":
    print("="*80)
    print("Improved S1→NDVI Fusion Model V2")
    print("Target: R² = 0.55-0.70")
    print("Import this module into your notebook to use the functions.")
    print("="*80)

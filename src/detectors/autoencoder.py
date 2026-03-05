"""
Autoencoder (FC) 기반 변경점 탐지기 (Wafer 기반)
PyTorch Fully-Connected Autoencoder로 Ref 패턴을 학습하고,
feature별 reconstruction error로 유의차를 탐지한다.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseDetector, DetectionResult


class _FCAutoencoder(nn.Module):
    """Fully Connected Autoencoder"""

    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (symmetric)
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        for i, h_dim in enumerate(reversed_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if i < len(reversed_dims) - 1:
                decoder_layers.extend([nn.ReLU(), nn.BatchNorm1d(h_dim)])
            else:
                decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseDetector):
    """Autoencoder reconstruction error 기반 탐지기 (Wafer 기반)"""

    name = "Autoencoder"

    def __init__(
        self,
        hidden_dims: list = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        error_threshold: float = 2.0,
        train_ratio: float = 0.8,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.error_threshold = error_threshold
        self.train_ratio = train_ratio
        self.random_state = random_state

    def detect_feature(self, ref_values, comp_values) -> DetectionResult:
        return DetectionResult(confidence=0.0, is_detected=False)

    def detect_all(self, dataset) -> list:
        """다변량: 전체 Feature 행렬을 AE로 분석"""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        ref_matrix = dataset.ref_data    # (n_ref, n_features)
        comp_matrix = dataset.comp_data  # (n_comp, n_features)
        n_features = ref_matrix.shape[1]

        # Ref 기준 표준화
        ref_mean = np.mean(ref_matrix, axis=0, keepdims=True)
        ref_std = np.std(ref_matrix, axis=0, keepdims=True)
        ref_std[ref_std < 1e-10] = 1e-10

        ref_scaled = (ref_matrix - ref_mean) / ref_std
        comp_scaled = (comp_matrix - ref_mean) / ref_std

        # Train / Holdout 분리
        n_ref = ref_scaled.shape[0]
        n_train = int(n_ref * self.train_ratio)
        indices = np.arange(n_ref)
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(indices)
        train_data = ref_scaled[indices[:n_train]]
        holdout_data = ref_scaled[indices[n_train:]]

        # PyTorch 학습
        device = torch.device("cpu")
        train_tensor = torch.FloatTensor(train_data).to(device)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        model = _FCAutoencoder(n_features, self.hidden_dims).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(self.epochs):
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        # Reconstruction error 산출 (feature별)
        model.eval()
        with torch.no_grad():
            holdout_recon = model(torch.FloatTensor(holdout_data)).numpy()
            comp_recon = model(torch.FloatTensor(comp_scaled)).numpy()

        holdout_errors = np.mean((holdout_data - holdout_recon) ** 2, axis=0)
        comp_errors = np.mean((comp_scaled - comp_recon) ** 2, axis=0)

        # IQR 기반 threshold
        error_ratio = comp_errors / (holdout_errors + 1e-10)
        q75 = np.percentile(error_ratio, 75)
        q25 = np.percentile(error_ratio, 25)
        iqr = q75 - q25
        threshold = q75 + self.error_threshold * iqr

        results = []
        for i in range(n_features):
            ratio = error_ratio[i]
            is_detected = ratio > threshold
            confidence = min(ratio / (threshold * 2), 1.0) if threshold > 0 else 0.0

            results.append(DetectionResult(
                feature_index=i,
                is_detected=is_detected,
                confidence=confidence,
                method_name=self.name,
                extra={
                    "comp_error": float(comp_errors[i]),
                    "holdout_error": float(holdout_errors[i]),
                    "error_ratio": float(ratio),
                    "threshold": float(threshold),
                },
            ))

        return results

"""
Autoencoder (FC) 기반 변경점 탐지기
PyTorch Fully-Connected Autoencoder로 Ref 패턴을 학습하고,
feature별 reconstruction error로 변경점을 탐지한다.
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
                decoder_layers.append(nn.ReLU())  # 출력은 non-negative
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderDetector(BaseDetector):
    """Autoencoder reconstruction error 기반 변경점 탐지기"""

    name = "Autoencoder"

    def __init__(
        self,
        hidden_dims: list = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        error_threshold: float = 2.0,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.error_threshold = error_threshold
        self.random_state = random_state

    def detect(self, ref_data, comp_data, full_series=None) -> DetectionResult:
        # 단변량에서는 사용하지 않음
        return DetectionResult(confidence=0.0, is_detected=False)

    def detect_all(self, dataset) -> list:
        """다변량: 전체 BIN 행렬을 AE로 분석"""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        ref_matrix = dataset.data[:dataset.ref_end_index, :]  # (ref_len, n_bins)
        comp_matrix = dataset.data[dataset.ref_end_index:, :]  # (comp_len, n_bins)
        n_bins = dataset.data.shape[1]

        # 표준화 (ref 기준)
        ref_mean = np.mean(ref_matrix, axis=0, keepdims=True)
        ref_std = np.std(ref_matrix, axis=0, keepdims=True)
        ref_std[ref_std < 1e-10] = 1e-10

        ref_scaled = (ref_matrix - ref_mean) / ref_std
        comp_scaled = (comp_matrix - ref_mean) / ref_std

        # Train/Holdout 분리 (80/20)
        n_ref = ref_scaled.shape[0]
        n_train = int(n_ref * 0.8)
        train_data = ref_scaled[:n_train]
        holdout_data = ref_scaled[n_train:]

        # PyTorch 데이터 준비
        device = torch.device("cpu")
        train_tensor = torch.FloatTensor(train_data).to(device)
        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # 모델 학습
        model = _FCAutoencoder(n_bins, self.hidden_dims).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss(reduction="none")

        model.train()
        for epoch in range(self.epochs):
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                recon = model(batch_x)
                loss = criterion(recon, batch_x).mean()
                loss.backward()
                optimizer.step()

        # Reconstruction error 산출
        model.eval()
        with torch.no_grad():
            holdout_tensor = torch.FloatTensor(holdout_data).to(device)
            comp_tensor = torch.FloatTensor(comp_scaled).to(device)

            holdout_recon = model(holdout_tensor).numpy()
            comp_recon = model(comp_tensor).numpy()

        # Feature별 MSE
        holdout_errors = np.mean((holdout_data - holdout_recon) ** 2, axis=0)
        comp_errors = np.mean((comp_scaled - comp_recon) ** 2, axis=0)

        # 탐지: comp error가 holdout error 대비 유의미하게 높으면 탐지
        error_ratio = comp_errors / (holdout_errors + 1e-10)
        # IQR 기반 threshold (outlier detection)
        q75 = np.percentile(error_ratio, 75)
        q25 = np.percentile(error_ratio, 25)
        iqr = q75 - q25
        threshold = q75 + self.error_threshold * iqr

        results = []
        for i in range(n_bins):
            ratio = error_ratio[i]
            is_detected = ratio > threshold
            confidence = min(ratio / (threshold * 2), 1.0) if threshold > 0 else 0.0

            results.append(DetectionResult(
                bin_index=i,
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

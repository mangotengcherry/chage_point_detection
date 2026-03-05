"""
AE + 통계검정 이중 경로 파이프라인 (ECO 방법론)

Step 1: AE 학습 (Train 800) → Holdout 200 + Comp 100 에 대한 Feature별 Recon Error
Step 2: AE Error 통계검정 (Holdout Ref vs Comp) → 1차 후보
Step 3: Raw Feature 통계검정 (Ref vs Comp) → 2차 후보
Step 4: 교집합 → 최종 유의 Feature
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from statsmodels.stats.multitest import multipletests
from dataclasses import dataclass, field


@dataclass
class DualPathResult:
    """이중 경로 파이프라인 결과"""
    # Step 2: AE Error 검정 결과
    ae_significant: np.ndarray            # (n_features,) bool
    ae_pvalues_mw: np.ndarray             # Mann-Whitney p-values
    ae_pvalues_ks: np.ndarray             # KS test p-values
    ae_recon_error_holdout: np.ndarray    # (n_features,) holdout 평균 error
    ae_recon_error_comp: np.ndarray       # (n_features,) comp 평균 error

    # Step 3: Raw Feature 검정 결과
    raw_significant: np.ndarray           # (n_features,) bool
    raw_pvalues_mw: np.ndarray
    raw_pvalues_ks: np.ndarray
    raw_ks_statistics: np.ndarray         # KS statistic (효과 크기)

    # Step 4: 교집합
    intersection: np.ndarray              # (n_features,) bool — 최종 유의 Feature
    ae_only: np.ndarray                   # AE만 유의
    raw_only: np.ndarray                  # Raw만 유의

    # 메타데이터
    feature_names: list = field(default_factory=list)
    n_ae_significant: int = 0
    n_raw_significant: int = 0
    n_intersection: int = 0


class _FCAutoencoder(nn.Module):
    """Fully Connected Autoencoder"""

    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

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
                decoder_layers.append(nn.ReLU())  # 출력 non-negative
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DualPathPipeline:
    """AE + 통계검정 이중 경로 파이프라인"""

    def __init__(
        self,
        hidden_dims: list = None,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        alpha: float = 0.05,
        fdr_method: str = "fdr_bh",
        train_ratio: float = 0.8,
        random_state: int = 42,
    ):
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.fdr_method = fdr_method
        self.train_ratio = train_ratio
        self.random_state = random_state

    def run(self, ref_data: np.ndarray, comp_data: np.ndarray,
            feature_names: list = None) -> DualPathResult:
        """전체 파이프라인 실행"""
        n_features = ref_data.shape[1]
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        # ── Step 0: 전처리 (Ref 기준 표준화) ──
        ref_mean = np.mean(ref_data, axis=0, keepdims=True)
        ref_std = np.std(ref_data, axis=0, keepdims=True)
        ref_std[ref_std < 1e-10] = 1e-10

        ref_scaled = (ref_data - ref_mean) / ref_std
        comp_scaled = (comp_data - ref_mean) / ref_std

        # ── Step 1: AE 학습 + Error 산출 ──
        n_ref = ref_scaled.shape[0]
        n_train = int(n_ref * self.train_ratio)

        # Train / Holdout 분리
        indices = np.arange(n_ref)
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(indices)
        train_idx = indices[:n_train]
        holdout_idx = indices[n_train:]

        train_data = ref_scaled[train_idx]
        holdout_data = ref_scaled[holdout_idx]

        print(f"    [Step 1] AE 학습: Train {len(train_idx)}, Holdout {len(holdout_idx)}")

        # AE 학습
        ae_model = self._train_ae(train_data, n_features)

        # Reconstruction error 산출 (feature별)
        holdout_errors = self._compute_feature_errors(ae_model, holdout_data)  # (n_holdout, n_features)
        comp_errors = self._compute_feature_errors(ae_model, comp_scaled)      # (n_comp, n_features)

        print(f"    [Step 1] Error 산출 완료: Holdout {holdout_errors.shape}, Comp {comp_errors.shape}")

        # ── Step 2: AE Error 통계검정 ──
        print(f"    [Step 2] AE Error 통계검정...")
        ae_pvalues_mw = np.ones(n_features)
        ae_pvalues_ks = np.ones(n_features)

        for j in range(n_features):
            h_err = holdout_errors[:, j]
            c_err = comp_errors[:, j]

            if np.std(h_err) > 0 or np.std(c_err) > 0:
                try:
                    _, ae_pvalues_mw[j] = stats.mannwhitneyu(
                        h_err, c_err, alternative="two-sided"
                    )
                except ValueError:
                    pass
                try:
                    _, ae_pvalues_ks[j] = stats.ks_2samp(h_err, c_err)
                except Exception:
                    pass

        # FDR correction (둘 다 유의해야 함)
        ae_sig_mw = self._fdr_correct(ae_pvalues_mw)
        ae_sig_ks = self._fdr_correct(ae_pvalues_ks)
        ae_significant = ae_sig_mw & ae_sig_ks  # 둘 다 유의

        ae_mean_holdout = np.mean(holdout_errors, axis=0)
        ae_mean_comp = np.mean(comp_errors, axis=0)

        print(f"    [Step 2] AE Error 유의 Feature: {ae_significant.sum()}개")

        # ── Step 3: Raw Feature 통계검정 ──
        print(f"    [Step 3] Raw Feature 통계검정...")
        raw_pvalues_mw = np.ones(n_features)
        raw_pvalues_ks = np.ones(n_features)
        raw_ks_statistics = np.zeros(n_features)

        for j in range(n_features):
            r = ref_data[:, j]
            c = comp_data[:, j]

            if np.std(r) > 0 or np.std(c) > 0:
                try:
                    _, raw_pvalues_mw[j] = stats.mannwhitneyu(
                        r, c, alternative="two-sided"
                    )
                except ValueError:
                    pass
                try:
                    ks_stat, raw_pvalues_ks[j] = stats.ks_2samp(r, c)
                    raw_ks_statistics[j] = ks_stat
                except Exception:
                    pass

        raw_sig_mw = self._fdr_correct(raw_pvalues_mw)
        raw_sig_ks = self._fdr_correct(raw_pvalues_ks)
        raw_significant = raw_sig_mw & raw_sig_ks

        print(f"    [Step 3] Raw Feature 유의 Feature: {raw_significant.sum()}개")

        # ── Step 4: 교집합 ──
        intersection = ae_significant & raw_significant
        ae_only = ae_significant & ~raw_significant
        raw_only = raw_significant & ~ae_significant

        print(f"    [Step 4] 교집합(최종): {intersection.sum()}개, "
              f"AE만: {ae_only.sum()}개, Raw만: {raw_only.sum()}개")

        return DualPathResult(
            ae_significant=ae_significant,
            ae_pvalues_mw=ae_pvalues_mw,
            ae_pvalues_ks=ae_pvalues_ks,
            ae_recon_error_holdout=ae_mean_holdout,
            ae_recon_error_comp=ae_mean_comp,
            raw_significant=raw_significant,
            raw_pvalues_mw=raw_pvalues_mw,
            raw_pvalues_ks=raw_pvalues_ks,
            raw_ks_statistics=raw_ks_statistics,
            intersection=intersection,
            ae_only=ae_only,
            raw_only=raw_only,
            feature_names=feature_names,
            n_ae_significant=int(ae_significant.sum()),
            n_raw_significant=int(raw_significant.sum()),
            n_intersection=int(intersection.sum()),
        )

    def _train_ae(self, train_data, n_features):
        """AE 학습"""
        torch.manual_seed(self.random_state)

        device = torch.device("cpu")
        train_tensor = torch.FloatTensor(train_data).to(device)
        dataset = TensorDataset(train_tensor, train_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = _FCAutoencoder(n_features, self.hidden_dims).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        return model

    def _compute_feature_errors(self, model, data):
        """Feature별 reconstruction error 산출 (per-sample, per-feature)"""
        with torch.no_grad():
            tensor = torch.FloatTensor(data)
            recon = model(tensor).numpy()
        # (n_samples, n_features) 각 셀이 개별 error
        return (data - recon) ** 2

    def _fdr_correct(self, pvalues):
        """FDR correction (Benjamini-Hochberg)"""
        rejected, _, _, _ = multipletests(pvalues, alpha=self.alpha,
                                           method=self.fdr_method)
        return rejected

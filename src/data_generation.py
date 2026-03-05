"""
합성 EDS BIN 데이터 생성기 (Wafer 단위)
Ref Group (1000 wafers) vs Comp Group (100 wafers) 간
유의차가 있는 Feature를 주입하여 벤치마크 데이터를 생성한다.

BIN130~BIN629 범위의 500개 Feature, 5가지 anomaly 유형 × 3단계 난이도.
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SyntheticBINDataset:
    """합성 BIN 벤치마크 데이터셋 (Wafer 단위)"""
    ref_data: np.ndarray                       # (n_ref, n_features)
    comp_data: np.ndarray                      # (n_comp, n_features)
    feature_names: list                        # ["BIN130", ..., "BIN629"]
    labels: np.ndarray                         # (n_features,) 0=정상, 1=유의차
    anomaly_types: dict = field(default_factory=dict)      # {feat_idx: type_name}
    difficulty_levels: dict = field(default_factory=dict)   # {feat_idx: "easy"/"medium"/"hard"}
    config: dict = field(default_factory=dict)


# 난이도별 파라미터
DIFFICULTY_PARAMS = {
    "easy": {
        "spike_multiplier": (15.0, 20.0),
        "shift_multiplier": (6.0, 8.0),
        "trend_multiplier": (4.0, 6.0),
        "jump_multiplier": (20.0, 30.0),
        "spike_ratio": 0.20,
    },
    "medium": {
        "spike_multiplier": (8.0, 12.0),
        "shift_multiplier": (3.0, 5.0),
        "trend_multiplier": (2.0, 3.0),
        "jump_multiplier": (10.0, 15.0),
        "spike_ratio": 0.15,
    },
    "hard": {
        "spike_multiplier": (3.0, 5.0),
        "shift_multiplier": (1.5, 2.5),
        "trend_multiplier": (1.0, 1.5),
        "jump_multiplier": (5.0, 8.0),
        "spike_ratio": 0.10,
    },
}

ANOMALY_TYPES = [
    "sporadic_spikes",
    "level_shift",
    "gradual_trend",
    "complex_trend",
    "sudden_jump",
]


class BINDataGenerator:
    """반도체 EDS BIN 합성 데이터 생성기 (Wafer 단위)"""

    def __init__(
        self,
        n_ref: int = 1000,
        n_comp: int = 100,
        n_features: int = 500,
        n_anomaly_per_type_per_difficulty: int = 10,
        random_state: int = 42,
    ):
        self.n_ref = n_ref
        self.n_comp = n_comp
        self.n_features = n_features
        self.n_anomaly_per = n_anomaly_per_type_per_difficulty
        self.rng = np.random.RandomState(random_state)

        self.n_anomaly_total = 5 * 3 * self.n_anomaly_per  # 150
        if self.n_anomaly_total > self.n_features:
            raise ValueError(
                f"anomaly 수({self.n_anomaly_total})가 전체 feature 수({self.n_features})를 초과"
            )

    def generate(self) -> SyntheticBINDataset:
        """합성 데이터셋 생성"""
        ref_data = np.zeros((self.n_ref, self.n_features))
        comp_data = np.zeros((self.n_comp, self.n_features))
        feature_names = [f"BIN{130 + i}" for i in range(self.n_features)]
        labels = np.zeros(self.n_features, dtype=int)
        anomaly_types = {}
        difficulty_levels = {}

        # 1) baseline 유형 할당 및 Ref/Comp 기본 데이터 생성
        baseline_types = self._assign_baseline_types()
        baselines = []
        stds = []

        for i in range(self.n_features):
            ref_col, comp_col, base_val, std_val = self._generate_feature(
                baseline_types[i]
            )
            ref_data[:, i] = ref_col
            comp_data[:, i] = comp_col
            baselines.append(base_val)
            stds.append(std_val)

        # 2) anomaly feature 선택 및 comp 데이터에 변화 주입
        anomaly_indices = self.rng.choice(
            self.n_features, size=self.n_anomaly_total, replace=False
        )

        idx = 0
        for atype in ANOMALY_TYPES:
            for diff in ["easy", "medium", "hard"]:
                for _ in range(self.n_anomaly_per):
                    feat_idx = anomaly_indices[idx]
                    comp_data[:, feat_idx] = self._inject_anomaly(
                        comp_data[:, feat_idx].copy(),
                        atype,
                        diff,
                        baselines[feat_idx],
                        stds[feat_idx],
                    )
                    labels[feat_idx] = 1
                    anomaly_types[int(feat_idx)] = atype
                    difficulty_levels[int(feat_idx)] = diff
                    idx += 1

        return SyntheticBINDataset(
            ref_data=ref_data,
            comp_data=comp_data,
            feature_names=feature_names,
            labels=labels,
            anomaly_types=anomaly_types,
            difficulty_levels=difficulty_levels,
            config={
                "n_ref": self.n_ref,
                "n_comp": self.n_comp,
                "n_features": self.n_features,
                "n_anomaly_total": self.n_anomaly_total,
                "baseline_types": baseline_types,
            },
        )

    def _assign_baseline_types(self) -> list:
        """BIN baseline 유형 할당 (zero_heavy 60%, low_rate 30%, moderate_rate 10%)"""
        types = []
        for _ in range(self.n_features):
            r = self.rng.random()
            if r < 0.6:
                types.append("zero_heavy")
            elif r < 0.9:
                types.append("low_rate")
            else:
                types.append("moderate_rate")
        return types

    def _generate_feature(self, btype: str):
        """Ref와 Comp의 동일 분포 baseline 생성.
        returns (ref_col, comp_col, base_value, std_value)"""
        if btype == "zero_heavy":
            scale = self.rng.uniform(0.0005, 0.002)
            ref_col = self._zero_heavy_samples(self.n_ref, scale)
            comp_col = self._zero_heavy_samples(self.n_comp, scale)
            base = scale * 0.10
            std = max(np.std(ref_col), scale * 0.5)
        elif btype == "low_rate":
            base = self.rng.uniform(0.001, 0.005)
            std = base * self.rng.uniform(0.2, 0.4)
            ref_col = np.maximum(0, self.rng.normal(base, std, self.n_ref))
            comp_col = np.maximum(0, self.rng.normal(base, std, self.n_comp))
        else:  # moderate_rate
            base = self.rng.uniform(0.01, 0.03)
            std = base * self.rng.uniform(0.15, 0.25)
            ref_col = np.maximum(0, self.rng.normal(base, std, self.n_ref))
            comp_col = np.maximum(0, self.rng.normal(base, std, self.n_comp))

        return ref_col, comp_col, base, std

    def _zero_heavy_samples(self, n: int, scale: float) -> np.ndarray:
        """90%가 0인 zero-inflated 샘플 생성"""
        samples = np.zeros(n)
        nonzero_mask = self.rng.random(n) < 0.10
        samples[nonzero_mask] = self.rng.exponential(scale, size=nonzero_mask.sum())
        return samples

    def _inject_anomaly(self, comp_col, atype, difficulty, base_val, std_val):
        """Comp 데이터에 anomaly 주입"""
        params = DIFFICULTY_PARAMS[difficulty]
        effective_std = max(std_val, base_val * 0.1, 1e-5)

        if atype == "sporadic_spikes":
            comp_col = self._inject_sporadic_spikes(comp_col, effective_std, params)
        elif atype == "level_shift":
            comp_col = self._inject_level_shift(comp_col, effective_std, params)
        elif atype == "gradual_trend":
            comp_col = self._inject_gradual_trend(comp_col, effective_std, params)
        elif atype == "complex_trend":
            comp_col = self._inject_complex_trend(comp_col, effective_std, params)
        elif atype == "sudden_jump":
            comp_col = self._inject_sudden_jump(comp_col, effective_std, params)

        return np.maximum(0, comp_col)

    def _inject_sporadic_spikes(self, col, std, params):
        """일부 wafer에서 큰 스파이크"""
        n = len(col)
        spike_mask = self.rng.random(n) < params["spike_ratio"]
        mult = self.rng.uniform(*params["spike_multiplier"], size=spike_mask.sum())
        col[spike_mask] += std * mult
        return col

    def _inject_level_shift(self, col, std, params):
        """전체 comp wafer의 평균이 상승"""
        shift_mag = std * self.rng.uniform(*params["shift_multiplier"])
        col += shift_mag
        return col

    def _inject_gradual_trend(self, col, std, params):
        """wafer 순서대로 점진적 증가"""
        n = len(col)
        max_increase = std * self.rng.uniform(*params["trend_multiplier"])
        trend = np.linspace(0, max_increase, n)
        col += trend
        return col

    def _inject_complex_trend(self, col, std, params):
        """상승 → 정상화 → 재상승 (3단계)"""
        n = len(col)
        third = n // 3
        max_rise = std * self.rng.uniform(*params["trend_multiplier"])

        phase1 = np.linspace(0, max_rise, third)
        phase2 = np.linspace(max_rise, max_rise * 0.3, third)
        remaining = n - 2 * third
        phase3 = np.linspace(max_rise * 0.3, max_rise * 0.8, remaining)

        col += np.concatenate([phase1, phase2, phase3])
        return col

    def _inject_sudden_jump(self, col, std, params):
        """소수 wafer에서 갑작스런 스파이크"""
        n_jump = self.rng.randint(1, 4)
        jump_indices = self.rng.choice(len(col), size=n_jump, replace=False)
        jump_mag = std * self.rng.uniform(*params["jump_multiplier"])
        col[jump_indices] += jump_mag
        return col

"""
합성 BIN 데이터 생성기
BIN130~BIN599 범위의 반도체 EDS BIN 데이터를 시뮬레이션하며,
5가지 anomaly 유형 × 3단계 난이도로 변경점을 주입한다.
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SyntheticBINDataset:
    """합성 BIN 벤치마크 데이터셋"""
    data: np.ndarray                          # (n_timepoints, n_bins)
    bin_names: list                            # ["BIN130", ..., "BIN599"]
    labels: np.ndarray                         # (n_bins,) 0=정상, 1=anomaly
    change_points: dict = field(default_factory=dict)   # {bin_idx: time_idx}
    anomaly_types: dict = field(default_factory=dict)   # {bin_idx: type_name}
    difficulty_levels: dict = field(default_factory=dict)  # {bin_idx: "easy"/"medium"/"hard"}
    ref_end_index: int = 150
    config: dict = field(default_factory=dict)


# 난이도별 SNR 배수 설정
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
    """반도체 EDS BIN 합성 데이터 생성기"""

    def __init__(
        self,
        n_timepoints: int = 300,
        n_bins: int = 470,
        n_anomaly_per_type_per_difficulty: int = 10,
        ref_ratio: float = 0.5,
        random_state: int = 42,
    ):
        self.n_timepoints = n_timepoints
        self.n_bins = n_bins
        self.n_anomaly_per = n_anomaly_per_type_per_difficulty
        self.ref_ratio = ref_ratio
        self.rng = np.random.RandomState(random_state)
        self.ref_end = int(n_timepoints * ref_ratio)

        # 총 anomaly BIN 수: 5 types × 3 difficulties × n_per
        self.n_anomaly_total = 5 * 3 * self.n_anomaly_per
        if self.n_anomaly_total > self.n_bins:
            raise ValueError(
                f"anomaly 수({self.n_anomaly_total})가 전체 BIN 수({self.n_bins})를 초과"
            )

    def generate(self) -> SyntheticBINDataset:
        """합성 BIN 데이터셋 생성"""
        data = np.zeros((self.n_timepoints, self.n_bins))
        bin_names = [f"BIN{130 + i}" for i in range(self.n_bins)]
        labels = np.zeros(self.n_bins, dtype=int)
        change_points = {}
        anomaly_types = {}
        difficulty_levels = {}

        # 1) baseline 유형 할당
        baseline_types = self._assign_baseline_types()
        baselines = []
        stds = []

        # 2) 각 BIN의 baseline 시계열 생성
        for i in range(self.n_bins):
            ts, base_val, std_val = self._generate_baseline(baseline_types[i])
            data[:, i] = ts
            baselines.append(base_val)
            stds.append(std_val)

        # 3) anomaly BIN 선택 및 주입
        anomaly_indices = self.rng.choice(
            self.n_bins, size=self.n_anomaly_total, replace=False
        )

        idx = 0
        for atype in ANOMALY_TYPES:
            for diff in ["easy", "medium", "hard"]:
                for _ in range(self.n_anomaly_per):
                    bin_idx = anomaly_indices[idx]
                    cp = self._random_change_point()
                    data[:, bin_idx] = self._inject_anomaly(
                        data[:, bin_idx].copy(),
                        atype,
                        diff,
                        cp,
                        baselines[bin_idx],
                        stds[bin_idx],
                    )
                    labels[bin_idx] = 1
                    change_points[int(bin_idx)] = int(cp)
                    anomaly_types[int(bin_idx)] = atype
                    difficulty_levels[int(bin_idx)] = diff
                    idx += 1

        return SyntheticBINDataset(
            data=data,
            bin_names=bin_names,
            labels=labels,
            change_points=change_points,
            anomaly_types=anomaly_types,
            difficulty_levels=difficulty_levels,
            ref_end_index=self.ref_end,
            config={
                "n_timepoints": self.n_timepoints,
                "n_bins": self.n_bins,
                "n_anomaly_total": self.n_anomaly_total,
                "ref_end_index": self.ref_end,
                "baseline_types": baseline_types,
            },
        )

    def _assign_baseline_types(self) -> list:
        """BIN baseline 유형 할당 (zero_heavy 60%, low_rate 30%, moderate_rate 10%)"""
        types = []
        for i in range(self.n_bins):
            r = self.rng.random()
            if r < 0.6:
                types.append("zero_heavy")
            elif r < 0.9:
                types.append("low_rate")
            else:
                types.append("moderate_rate")
        return types

    def _generate_baseline(self, btype: str):
        """baseline 시계열 생성. returns (timeseries, base_value, std_value)"""
        n = self.n_timepoints

        if btype == "zero_heavy":
            ts = np.zeros(n)
            # 약 10%의 시점에서 작은 값
            nonzero_mask = self.rng.random(n) < 0.10
            scale = self.rng.uniform(0.0005, 0.002)
            ts[nonzero_mask] = self.rng.exponential(scale, size=nonzero_mask.sum())
            base = scale * 0.10  # effective mean
            std = max(np.std(ts), scale * 0.5)

        elif btype == "low_rate":
            base = self.rng.uniform(0.001, 0.005)
            std = base * self.rng.uniform(0.2, 0.4)
            ts = np.maximum(0, self.rng.normal(base, std, n))

        else:  # moderate_rate
            base = self.rng.uniform(0.01, 0.03)
            std = base * self.rng.uniform(0.15, 0.25)
            ts = np.maximum(0, self.rng.normal(base, std, n))

        return ts, base, std

    def _random_change_point(self) -> int:
        """comp period 초반~중반에서 change point 선택"""
        comp_start = self.ref_end
        comp_len = self.n_timepoints - self.ref_end
        earliest = comp_start + int(comp_len * 0.05)
        latest = comp_start + int(comp_len * 0.4)
        return self.rng.randint(earliest, latest + 1)

    def _inject_anomaly(
        self, ts, atype, difficulty, cp, base_val, std_val
    ) -> np.ndarray:
        """시계열에 anomaly를 주입"""
        params = DIFFICULTY_PARAMS[difficulty]
        # std가 매우 작은 경우 최소값 보장
        effective_std = max(std_val, base_val * 0.1, 1e-5)

        if atype == "sporadic_spikes":
            ts = self._inject_sporadic_spikes(ts, cp, effective_std, params)
        elif atype == "level_shift":
            ts = self._inject_level_shift(ts, cp, effective_std, params)
        elif atype == "gradual_trend":
            ts = self._inject_gradual_trend(ts, cp, effective_std, params)
        elif atype == "complex_trend":
            ts = self._inject_complex_trend(ts, cp, effective_std, params)
        elif atype == "sudden_jump":
            ts = self._inject_sudden_jump(ts, cp, effective_std, params)

        return np.maximum(0, ts)

    def _inject_sporadic_spikes(self, ts, cp, std, params):
        """산포 유사하나 빈번한 큰 스파이크"""
        n_after = len(ts) - cp
        spike_mask = self.rng.random(n_after) < params["spike_ratio"]
        mult = self.rng.uniform(*params["spike_multiplier"], size=spike_mask.sum())
        ts[cp:][spike_mask] += std * mult
        return ts

    def _inject_level_shift(self, ts, cp, std, params):
        """일정 기간 중심치 상승 후 복귀"""
        shift_mag = std * self.rng.uniform(*params["shift_multiplier"])
        duration = self.rng.randint(30, min(80, len(ts) - cp))
        end = min(cp + duration, len(ts))
        ts[cp:end] += shift_mag
        return ts

    def _inject_gradual_trend(self, ts, cp, std, params):
        """선형 증가 트렌드"""
        n_after = len(ts) - cp
        max_increase = std * self.rng.uniform(*params["trend_multiplier"])
        trend = np.linspace(0, max_increase, n_after)
        ts[cp:] += trend
        return ts

    def _inject_complex_trend(self, ts, cp, std, params):
        """상승 → 정상화 → 재상승 (3단계 패턴)"""
        n_after = len(ts) - cp
        third = n_after // 3
        max_rise = std * self.rng.uniform(*params["trend_multiplier"])

        # Phase 1: 급상승
        phase1 = np.linspace(0, max_rise, third)
        # Phase 2: 정상화 (50%까지 감소)
        phase2 = np.linspace(max_rise, max_rise * 0.3, third)
        # Phase 3: 완만한 재상승
        remaining = n_after - 2 * third
        phase3 = np.linspace(max_rise * 0.3, max_rise * 0.8, remaining)

        injection = np.concatenate([phase1, phase2, phase3])
        ts[cp:] += injection
        return ts

    def _inject_sudden_jump(self, ts, cp, std, params):
        """갑작스런 1~3 포인트 스파이크 후 즉시 정상"""
        n_points = self.rng.randint(1, 4)
        jump_mag = std * self.rng.uniform(*params["jump_multiplier"])
        end = min(cp + n_points, len(ts))
        ts[cp:end] += jump_mag
        # 인접 1~2 포인트에 10% 크기로 부수 효과
        if end < len(ts):
            ts[end] += jump_mag * 0.1
        return ts

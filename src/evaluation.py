"""
벤치마크 평가 프레임워크 (Wafer 기반)
각 탐지 방법의 Precision, Recall, F1 Score를 전체/유형별/난이도별로 산출한다.
"""
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import roc_curve, auc


@dataclass
class BenchmarkResult:
    """단일 방법에 대한 벤치마크 결과"""
    method_name: str = ""
    overall_metrics: dict = field(default_factory=dict)
    per_anomaly_type: dict = field(default_factory=dict)
    per_difficulty: dict = field(default_factory=dict)
    detection_results: list = field(default_factory=list)
    execution_time: float = 0.0
    roc_data: dict = field(default_factory=dict)


class BenchmarkEvaluator:
    """벤치마크 평가기 (Wafer 기반 데이터셋)"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.results: list = []

    def evaluate_detector(self, detector) -> BenchmarkResult:
        """단일 탐지기를 평가한다."""
        start = time.time()
        detection_results = detector.detect_all(self.dataset)
        elapsed = time.time() - start

        y_true = self.dataset.labels
        y_pred = np.array([r.is_detected for r in detection_results], dtype=int)
        confidences = np.array([r.confidence for r in detection_results])

        overall = self._compute_metrics(y_true, y_pred)
        overall["n_detected"] = int(y_pred.sum())
        overall["n_anomaly"] = int(y_true.sum())
        overall["n_normal"] = int((y_true == 0).sum())

        # ROC curve
        roc_data = {}
        try:
            fpr, tpr, _ = roc_curve(y_true, confidences)
            roc_auc = auc(fpr, tpr)
            roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        except Exception:
            roc_data = {"fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": 0.5}

        # anomaly 유형별 성능
        per_type = {}
        for atype in set(self.dataset.anomaly_types.values()):
            type_indices = [
                idx for idx, t in self.dataset.anomaly_types.items() if t == atype
            ]
            normal_indices = [i for i in range(len(y_true)) if y_true[i] == 0]
            eval_indices = type_indices + normal_indices

            y_true_sub = np.array([y_true[i] for i in eval_indices])
            y_pred_sub = np.array([y_pred[i] for i in eval_indices])
            per_type[atype] = self._compute_metrics(y_true_sub, y_pred_sub)

            type_pred = np.array([y_pred[i] for i in type_indices])
            per_type[atype]["type_recall"] = float(np.mean(type_pred)) if len(type_pred) > 0 else 0.0

        # 난이도별 성능
        per_diff = {}
        for diff in ["easy", "medium", "hard"]:
            diff_indices = [
                idx for idx, d in self.dataset.difficulty_levels.items() if d == diff
            ]
            normal_indices = [i for i in range(len(y_true)) if y_true[i] == 0]
            eval_indices = diff_indices + normal_indices

            y_true_sub = np.array([y_true[i] for i in eval_indices])
            y_pred_sub = np.array([y_pred[i] for i in eval_indices])
            per_diff[diff] = self._compute_metrics(y_true_sub, y_pred_sub)

            diff_pred = np.array([y_pred[i] for i in diff_indices])
            per_diff[diff]["diff_recall"] = float(np.mean(diff_pred)) if len(diff_pred) > 0 else 0.0

        result = BenchmarkResult(
            method_name=detector.name,
            overall_metrics=overall,
            per_anomaly_type=per_type,
            per_difficulty=per_diff,
            detection_results=detection_results,
            execution_time=elapsed,
            roc_data=roc_data,
        )
        self.results.append(result)
        return result

    def evaluate_dual_path(self, dual_result, method_name="AE Dual-Path") -> BenchmarkResult:
        """DualPathResult를 BenchmarkResult로 변환하여 평가한다."""
        y_true = self.dataset.labels
        y_pred = dual_result.intersection.astype(int)
        n_features = len(y_true)

        # DetectionResult 리스트 생성
        detection_results = []
        for j in range(n_features):
            detection_results.append(type('DetectionResult', (), {
                'feature_index': j,
                'is_detected': bool(y_pred[j]),
                'confidence': float(1.0 - min(dual_result.raw_pvalues_mw[j],
                                               dual_result.raw_pvalues_ks[j])),
                'method_name': method_name,
            })())

        confidences = np.array([r.confidence for r in detection_results])

        overall = self._compute_metrics(y_true, y_pred)
        overall["n_detected"] = int(y_pred.sum())
        overall["n_anomaly"] = int(y_true.sum())
        overall["n_normal"] = int((y_true == 0).sum())

        # ROC
        roc_data = {}
        try:
            fpr, tpr, _ = roc_curve(y_true, confidences)
            roc_auc = auc(fpr, tpr)
            roc_data = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        except Exception:
            roc_data = {"fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": 0.5}

        # 유형별
        per_type = {}
        for atype in set(self.dataset.anomaly_types.values()):
            type_indices = [
                idx for idx, t in self.dataset.anomaly_types.items() if t == atype
            ]
            normal_indices = [i for i in range(len(y_true)) if y_true[i] == 0]
            eval_indices = type_indices + normal_indices
            y_true_sub = np.array([y_true[i] for i in eval_indices])
            y_pred_sub = np.array([y_pred[i] for i in eval_indices])
            per_type[atype] = self._compute_metrics(y_true_sub, y_pred_sub)
            type_pred = np.array([y_pred[i] for i in type_indices])
            per_type[atype]["type_recall"] = float(np.mean(type_pred)) if len(type_pred) > 0 else 0.0

        # 난이도별
        per_diff = {}
        for diff in ["easy", "medium", "hard"]:
            diff_indices = [
                idx for idx, d in self.dataset.difficulty_levels.items() if d == diff
            ]
            normal_indices = [i for i in range(len(y_true)) if y_true[i] == 0]
            eval_indices = diff_indices + normal_indices
            y_true_sub = np.array([y_true[i] for i in eval_indices])
            y_pred_sub = np.array([y_pred[i] for i in eval_indices])
            per_diff[diff] = self._compute_metrics(y_true_sub, y_pred_sub)
            diff_pred = np.array([y_pred[i] for i in diff_indices])
            per_diff[diff]["diff_recall"] = float(np.mean(diff_pred)) if len(diff_pred) > 0 else 0.0

        result = BenchmarkResult(
            method_name=method_name,
            overall_metrics=overall,
            per_anomaly_type=per_type,
            per_difficulty=per_diff,
            detection_results=detection_results,
            execution_time=0.0,
            roc_data=roc_data,
        )
        self.results.append(result)
        return result

    def _compute_metrics(self, y_true, y_pred) -> dict:
        """Precision, Recall, F1 산출"""
        if len(y_true) == 0 or y_true.sum() == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    def summary_table(self) -> pd.DataFrame:
        """전체 결과 요약 테이블"""
        rows = []
        for r in self.results:
            row = {
                "Method": r.method_name,
                "Precision": r.overall_metrics.get("precision", 0),
                "Recall": r.overall_metrics.get("recall", 0),
                "F1": r.overall_metrics.get("f1", 0),
                "AUC": r.roc_data.get("auc", 0),
                "TP": r.overall_metrics.get("tp", 0),
                "FP": r.overall_metrics.get("fp", 0),
                "FN": r.overall_metrics.get("fn", 0),
                "Time(s)": round(r.execution_time, 3),
            }
            for diff in ["easy", "medium", "hard"]:
                if diff in r.per_difficulty:
                    row[f"Recall_{diff}"] = r.per_difficulty[diff].get("diff_recall", 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("F1", ascending=False).reset_index(drop=True)
        return df

    def type_breakdown_table(self) -> pd.DataFrame:
        """anomaly 유형별 Recall 테이블"""
        rows = []
        for r in self.results:
            row = {"Method": r.method_name}
            for atype, metrics in r.per_anomaly_type.items():
                row[atype] = metrics.get("type_recall", 0)
            rows.append(row)
        return pd.DataFrame(rows)

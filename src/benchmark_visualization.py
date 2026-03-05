"""
벤치마크 전용 시각화 모듈
6종 차트: Detection Overlay, Performance Heatmap, Method Comparison Bar,
         ROC Curves, Anomaly Type Breakdown, Execution Time
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


class BenchmarkVisualizer:
    """벤치마크 결과 시각화"""

    def __init__(self, output_dir: str = "docs/benchmark_results", dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

    def plot_all(self, evaluator, dataset):
        """모든 시각화를 한번에 생성"""
        self.plot_detection_overlay(evaluator, dataset)
        self.plot_performance_heatmap(evaluator)
        self.plot_method_comparison(evaluator)
        self.plot_roc_curves(evaluator)
        self.plot_anomaly_type_breakdown(evaluator)
        self.plot_execution_time(evaluator)
        self.plot_difficulty_breakdown(evaluator)
        print(f"[시각화] 모든 차트가 {self.output_dir}에 저장되었습니다.")

    def plot_detection_overlay(self, evaluator, dataset):
        """anomaly 유형별 시계열 + 탐지 결과 오버레이"""
        anomaly_types_list = sorted(set(dataset.anomaly_types.values()))

        for atype in anomaly_types_list:
            # 해당 유형의 첫 번째 (medium 난이도) anomaly BIN 선택
            type_bins = [
                idx for idx, t in dataset.anomaly_types.items()
                if t == atype and dataset.difficulty_levels.get(idx) == "medium"
            ]
            if not type_bins:
                type_bins = [
                    idx for idx, t in dataset.anomaly_types.items() if t == atype
                ]
            if not type_bins:
                continue

            bin_idx = type_bins[0]
            ts = dataset.data[:, bin_idx]
            cp = dataset.change_points.get(bin_idx, -1)

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(ts, color="steelblue", alpha=0.8, linewidth=0.8, label="시계열")
            ax.axvline(
                dataset.ref_end_index, color="gray", linestyle="--",
                alpha=0.5, label="Ref/Comp 경계"
            )
            if cp >= 0:
                ax.axvline(cp, color="green", linestyle="--", linewidth=2,
                           label=f"실제 변경점 (t={cp})")

            # 각 방법의 탐지 결과 표시
            for j, result in enumerate(evaluator.results):
                det = result.detection_results[bin_idx]
                if det.detected_cp_index > 0:
                    ax.axvline(
                        det.detected_cp_index,
                        color=self.colors[j % len(self.colors)],
                        linestyle=":", alpha=0.7,
                        label=f"{result.method_name} (t={det.detected_cp_index})"
                    )

            ax.set_title(
                f"Detection Overlay — {atype} "
                f"({dataset.bin_names[bin_idx]}, "
                f"{dataset.difficulty_levels.get(bin_idx, 'N/A')})",
                fontsize=13
            )
            ax.set_xlabel("Time (Lot)")
            ax.set_ylabel("BIN Value")
            ax.legend(fontsize=7, loc="upper left", ncol=2)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(
                self.output_dir / f"detection_overlay_{atype}.png",
                dpi=self.dpi, bbox_inches="tight"
            )
            plt.close(fig)

    def plot_performance_heatmap(self, evaluator):
        """방법 × anomaly 유형 F1 히트맵"""
        methods = [r.method_name for r in evaluator.results]
        anomaly_types = sorted(
            set().union(*(r.per_anomaly_type.keys() for r in evaluator.results))
        )

        data = []
        for r in evaluator.results:
            row = []
            for atype in anomaly_types:
                if atype in r.per_anomaly_type:
                    row.append(r.per_anomaly_type[atype].get("type_recall", 0))
                else:
                    row.append(0)
            row.append(r.overall_metrics.get("f1", 0))  # Overall F1
            data.append(row)

        columns = anomaly_types + ["Overall F1"]
        df = pd.DataFrame(data, index=methods, columns=columns)

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            df, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0, vmax=1, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Score"}
        )
        ax.set_title("Performance Heatmap (유형별 Recall + Overall F1)", fontsize=14)
        ax.set_ylabel("Detection Method")
        ax.set_xlabel("Anomaly Type / Overall")

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "performance_heatmap.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_method_comparison(self, evaluator):
        """전체 Precision/Recall/F1 비교 바차트"""
        methods = [r.method_name for r in evaluator.results]
        precisions = [r.overall_metrics.get("precision", 0) for r in evaluator.results]
        recalls = [r.overall_metrics.get("recall", 0) for r in evaluator.results]
        f1s = [r.overall_metrics.get("f1", 0) for r in evaluator.results]

        # F1 기준 정렬
        sorted_idx = np.argsort(f1s)[::-1]
        methods = [methods[i] for i in sorted_idx]
        precisions = [precisions[i] for i in sorted_idx]
        recalls = [recalls[i] for i in sorted_idx]
        f1s = [f1s[i] for i in sorted_idx]

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 7))
        bars1 = ax.bar(x - width, precisions, width, label="Precision", color="#1f77b4")
        bars2 = ax.bar(x, recalls, width, label="Recall", color="#ff7f0e")
        bars3 = ax.bar(x + width, f1s, width, label="F1 Score", color="#2ca02c")

        # Precision 0.8 기준선
        ax.axhline(y=0.8, color="red", linestyle="--", linewidth=2,
                    alpha=0.7, label="Precision 0.8 목표")

        ax.set_xlabel("Detection Method")
        ax.set_ylabel("Score")
        ax.set_title("Method Performance Comparison (Precision / Recall / F1)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        # 값 표시
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7,
                    )

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "method_comparison_bar.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_roc_curves(self, evaluator):
        """ROC 곡선 비교"""
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, r in enumerate(evaluator.results):
            if "fpr" in r.roc_data and "tpr" in r.roc_data:
                fpr = r.roc_data["fpr"]
                tpr = r.roc_data["tpr"]
                auc_val = r.roc_data.get("auc", 0)
                ax.plot(
                    fpr, tpr,
                    color=self.colors[i % len(self.colors)],
                    linewidth=1.5,
                    label=f"{r.method_name} (AUC={auc_val:.3f})"
                )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Methods", fontsize=14)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "roc_curves.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_anomaly_type_breakdown(self, evaluator):
        """anomaly 유형별 방법 성능 비교 (5 subplots)"""
        anomaly_types = sorted(
            set().union(*(r.per_anomaly_type.keys() for r in evaluator.results))
        )
        n_types = len(anomaly_types)
        if n_types == 0:
            return

        fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 6), sharey=True)
        if n_types == 1:
            axes = [axes]

        methods = [r.method_name for r in evaluator.results]

        for idx, atype in enumerate(anomaly_types):
            ax = axes[idx]
            recalls = []
            for r in evaluator.results:
                if atype in r.per_anomaly_type:
                    recalls.append(r.per_anomaly_type[atype].get("type_recall", 0))
                else:
                    recalls.append(0)

            # Recall 기준 정렬
            sorted_idx = np.argsort(recalls)[::-1]
            sorted_methods = [methods[i] for i in sorted_idx]
            sorted_recalls = [recalls[i] for i in sorted_idx]

            colors = ["#2ca02c" if v >= 0.8 else "#d62728" if v < 0.5 else "#ff7f0e"
                       for v in sorted_recalls]
            bars = ax.barh(range(len(sorted_methods)), sorted_recalls, color=colors)
            ax.set_yticks(range(len(sorted_methods)))
            ax.set_yticklabels(sorted_methods, fontsize=8)
            ax.set_xlim(0, 1.05)
            ax.set_title(atype, fontsize=10)
            ax.axvline(0.8, color="red", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3, axis="x")

            for bar, val in zip(bars, sorted_recalls):
                ax.text(
                    val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7
                )

        fig.suptitle("Anomaly Type별 Recall Breakdown", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "anomaly_type_breakdown.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_difficulty_breakdown(self, evaluator):
        """난이도별 Recall 비교"""
        difficulties = ["easy", "medium", "hard"]
        methods = [r.method_name for r in evaluator.results]

        data = {}
        for diff in difficulties:
            data[diff] = []
            for r in evaluator.results:
                if diff in r.per_difficulty:
                    data[diff].append(r.per_difficulty[diff].get("diff_recall", 0))
                else:
                    data[diff].append(0)

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 7))
        for i, diff in enumerate(difficulties):
            offset = (i - 1) * width
            colors_map = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}
            ax.bar(x + offset, data[diff], width,
                   label=f"{diff.capitalize()}", color=colors_map[diff], alpha=0.8)

        ax.axhline(y=0.8, color="red", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_xlabel("Detection Method")
        ax.set_ylabel("Recall")
        ax.set_title("난이도별 Recall 비교 (Easy / Medium / Hard)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "difficulty_breakdown.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_execution_time(self, evaluator):
        """실행 시간 비교 (로그 스케일)"""
        methods = [r.method_name for r in evaluator.results]
        times = [r.execution_time for r in evaluator.results]
        f1s = [r.overall_metrics.get("f1", 0) for r in evaluator.results]

        # 시간 기준 정렬
        sorted_idx = np.argsort(times)
        methods = [methods[i] for i in sorted_idx]
        times = [times[i] for i in sorted_idx]
        f1s = [f1s[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(methods)), times, color="#1f77b4", alpha=0.8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=10)
        ax.set_xlabel("Execution Time (seconds, log scale)")
        ax.set_title("실행 시간 비교 (F1 Score 표시)", fontsize=14)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")

        for bar, t, f1 in zip(bars, times, f1s):
            ax.text(
                t * 1.1, bar.get_y() + bar.get_height() / 2,
                f"{t:.3f}s | F1={f1:.2f}",
                va="center", fontsize=9
            )

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "execution_time.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

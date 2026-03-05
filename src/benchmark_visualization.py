"""
벤치마크 전용 시각화 모듈 (Wafer 기반)
Scatter 차트 + Feature 검출 시각화 + Performance 비교 + 계산비용 분석
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


class BenchmarkVisualizer:
    """벤치마크 결과 시각화 (Wafer 기반)"""

    def __init__(self, output_dir: str = "docs/benchmark_results", dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

    def plot_all(self, evaluator, dataset, dual_result=None, all_method_detections=None):
        """모든 시각화를 한번에 생성"""
        self.plot_scatter_overlay(dataset)
        self.plot_performance_heatmap(evaluator)
        self.plot_method_comparison(evaluator)
        self.plot_roc_curves(evaluator)
        self.plot_anomaly_type_breakdown(evaluator)
        self.plot_difficulty_breakdown(evaluator)
        self.plot_execution_time(evaluator)
        self.plot_cost_analysis(evaluator)
        if dual_result is not None:
            self.plot_dual_path_summary(dual_result, dataset)
            self.plot_feature_significance(dual_result, dataset)
        if all_method_detections is not None:
            self.plot_all_methods_detected_features(
                all_method_detections, dataset, evaluator
            )
        print(f"[시각화] 모든 차트가 {self.output_dir}에 저장되었습니다.")

    def plot_scatter_overlay(self, dataset):
        """anomaly 유형별 Ref vs Comp scatter 차트"""
        anomaly_types_list = sorted(set(dataset.anomaly_types.values()))

        for atype in anomaly_types_list:
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

            feat_idx = type_bins[0]
            ref_vals = dataset.ref_data[:, feat_idx]
            comp_vals = dataset.comp_data[:, feat_idx]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 좌: Scatter (wafer index vs value)
            ax = axes[0]
            ax.scatter(range(len(ref_vals)), ref_vals,
                       alpha=0.4, s=8, color="#1f77b4", label="Ref")
            ax.scatter(range(len(comp_vals)), comp_vals,
                       alpha=0.6, s=12, color="#d62728", label="Comp")
            ax.set_xlabel("Wafer Index")
            ax.set_ylabel("Feature Value")
            ax.set_title(f"Scatter - {atype} ({dataset.feature_names[feat_idx]})")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # 우: 분포 비교 (histogram)
            ax2 = axes[1]
            ax2.hist(ref_vals, bins=30, alpha=0.5, color="#1f77b4",
                     label=f"Ref (mean={np.mean(ref_vals):.4f})", density=True)
            ax2.hist(comp_vals, bins=30, alpha=0.5, color="#d62728",
                     label=f"Comp (mean={np.mean(comp_vals):.4f})", density=True)
            ax2.set_xlabel("Feature Value")
            ax2.set_ylabel("Density")
            ax2.set_title(f"Distribution - {atype}")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(
                self.output_dir / f"scatter_{atype}.png",
                dpi=self.dpi, bbox_inches="tight"
            )
            plt.close(fig)

    def plot_dual_path_summary(self, dual_result, dataset):
        """AE Dual-path 파이프라인 결과 요약 (벤다이어그램 스타일)"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1) AE Error 검정 결과 scatter
        ax = axes[0]
        ae_err_diff = dual_result.ae_recon_error_comp - dual_result.ae_recon_error_holdout
        colors_ae = ["#d62728" if s else "#1f77b4" for s in dual_result.ae_significant]
        ax.scatter(range(len(ae_err_diff)), ae_err_diff, c=colors_ae, s=8, alpha=0.6)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("AE Error Diff (Comp - Holdout)")
        ax.set_title(f"Step 2: AE Error 검정\n(유의: {dual_result.n_ae_significant}개)")
        ax.grid(True, alpha=0.3)

        # 2) Raw Feature 검정 결과 scatter
        ax2 = axes[1]
        colors_raw = ["#d62728" if s else "#1f77b4" for s in dual_result.raw_significant]
        ax2.scatter(range(len(dual_result.raw_ks_statistics)),
                    dual_result.raw_ks_statistics, c=colors_raw, s=8, alpha=0.6)
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("KS Statistic")
        ax2.set_title(f"Step 3: Raw Feature 검정\n(유의: {dual_result.n_raw_significant}개)")
        ax2.grid(True, alpha=0.3)

        # 3) 교집합 Venn-style bar
        ax3 = axes[2]
        categories = ["AE Only", "Intersection\n(Final)", "Raw Only"]
        counts = [
            int(dual_result.ae_only.sum()),
            int(dual_result.intersection.sum()),
            int(dual_result.raw_only.sum()),
        ]
        bar_colors = ["#ff7f0e", "#2ca02c", "#9467bd"]
        bars = ax3.bar(categories, counts, color=bar_colors, edgecolor="black", linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(cnt), ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax3.set_ylabel("Feature Count")
        ax3.set_title("Step 4: 교집합 (Cross-Validation)")
        ax3.grid(True, alpha=0.3, axis="y")

        fig.suptitle("AE Dual-Path Pipeline 결과 요약", fontsize=15, y=1.02)
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "dual_path_summary.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_feature_significance(self, dual_result, dataset):
        """Feature별 유의차 검출 결과 시각화 (Color-coded)"""
        n_features = len(dual_result.intersection)

        fig, ax = plt.subplots(figsize=(16, 6))

        # Color coding: 정상=gray, AE만=orange, Raw만=purple, 교집합=red
        colors = []
        for j in range(n_features):
            if dual_result.intersection[j]:
                colors.append("#d62728")  # 교집합 (최종 유의)
            elif dual_result.ae_only[j]:
                colors.append("#ff7f0e")  # AE만
            elif dual_result.raw_only[j]:
                colors.append("#9467bd")  # Raw만
            else:
                colors.append("#cccccc")  # 정상

        # KS statistic을 y축으로 사용
        ks_vals = dual_result.raw_ks_statistics
        ax.scatter(range(n_features), ks_vals, c=colors, s=10, alpha=0.7)

        # Ground truth anomaly 표시
        for j in range(n_features):
            if dataset.labels[j] == 1:
                ax.scatter(j, ks_vals[j], facecolors='none', edgecolors='black',
                          s=40, linewidths=0.8)

        # Legend
        legend_elements = [
            Patch(facecolor="#d62728", label=f"교집합 (최종): {int(dual_result.intersection.sum())}"),
            Patch(facecolor="#ff7f0e", label=f"AE만: {int(dual_result.ae_only.sum())}"),
            Patch(facecolor="#9467bd", label=f"Raw만: {int(dual_result.raw_only.sum())}"),
            Patch(facecolor="#cccccc", label="정상"),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
                   markerfacecolor='none', markersize=8, label="Ground Truth Anomaly"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

        ax.set_xlabel("Feature Index")
        ax.set_ylabel("KS Statistic (Raw)")
        ax.set_title("Feature별 유의차 검출 결과 (AE Dual-Path Pipeline)", fontsize=13)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "feature_significance.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_all_methods_detected_features(self, all_method_detections, dataset, evaluator):
        """모든 방법별 유의차 검출 Feature 시각화 (6개 서브플롯)"""
        method_names = list(all_method_detections.keys())
        n_methods = len(method_names)
        n_features = len(dataset.labels)

        # 전체 비교 차트 (6 서브플롯)
        fig, axes = plt.subplots(n_methods, 1, figsize=(16, 3.5 * n_methods), sharex=True)
        if n_methods == 1:
            axes = [axes]

        for idx, method_name in enumerate(method_names):
            ax = axes[idx]
            detected = all_method_detections[method_name]["detected"]
            confidences = all_method_detections[method_name]["confidences"]

            # TP/FP/FN/TN 분류
            colors = []
            for j in range(n_features):
                if detected[j] and dataset.labels[j] == 1:
                    colors.append("#2ca02c")   # TP (정확 검출) - 초록
                elif detected[j] and dataset.labels[j] == 0:
                    colors.append("#d62728")   # FP (오탐) - 빨강
                elif not detected[j] and dataset.labels[j] == 1:
                    colors.append("#ff7f0e")   # FN (미탐) - 주황
                else:
                    colors.append("#cccccc")   # TN (정상) - 회색

            ax.scatter(range(n_features), confidences, c=colors, s=8, alpha=0.7)

            # Ground truth anomaly 위치 표시
            anomaly_indices = np.where(dataset.labels == 1)[0]
            for j in anomaly_indices:
                ax.scatter(j, confidences[j], facecolors='none', edgecolors='black',
                          s=30, linewidths=0.5)

            # 성능 정보 가져오기
            tp = int(np.sum(detected & (dataset.labels == 1)))
            fp = int(np.sum(detected & (dataset.labels == 0)))
            fn = int(np.sum(~detected & (dataset.labels == 1)))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            title = (f"{method_name}  |  "
                     f"검출: {int(detected.sum())}개  |  "
                     f"TP={tp}, FP={fp}, FN={fn}  |  "
                     f"P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_ylabel("Confidence")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

        axes[-1].set_xlabel("Feature Index")

        # 공통 범례
        legend_elements = [
            Patch(facecolor="#2ca02c", label="TP (정확 검출)"),
            Patch(facecolor="#d62728", label="FP (오탐)"),
            Patch(facecolor="#ff7f0e", label="FN (미탐)"),
            Patch(facecolor="#cccccc", label="TN (정상)"),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
                   markerfacecolor='none', markersize=8, label="Ground Truth Anomaly"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=5,
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle("방법별 유의차 검출 Feature 비교", fontsize=15, y=1.01)
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "all_methods_feature_detection.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

        # 방법 간 검출 일치도 히트맵
        self._plot_detection_agreement(all_method_detections, dataset)

    def _plot_detection_agreement(self, all_method_detections, dataset):
        """방법 간 검출 일치도 히트맵"""
        method_names = list(all_method_detections.keys())
        n_methods = len(method_names)
        n_features = len(dataset.labels)

        # 검출 매트릭스 (methods x features)
        detection_matrix = np.zeros((n_methods, n_features), dtype=int)
        for i, name in enumerate(method_names):
            detection_matrix[i] = all_method_detections[name]["detected"].astype(int)

        # 방법 간 Jaccard 유사도
        jaccard_matrix = np.zeros((n_methods, n_methods))
        for i in range(n_methods):
            for j in range(n_methods):
                intersection = np.sum(detection_matrix[i] & detection_matrix[j])
                union = np.sum(detection_matrix[i] | detection_matrix[j])
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            jaccard_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=method_names, yticklabels=method_names,
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={"label": "Jaccard Similarity"}
        )
        ax.set_title("방법 간 검출 일치도 (Jaccard Similarity)", fontsize=14)

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "detection_agreement_heatmap.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_cost_analysis(self, evaluator):
        """계산비용 비교 분석 시각화"""
        methods = [r.method_name for r in evaluator.results]
        times = [r.execution_time for r in evaluator.results]
        f1s = [r.overall_metrics.get("f1", 0) for r in evaluator.results]
        precisions = [r.overall_metrics.get("precision", 0) for r in evaluator.results]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # 1) 실행 시간 vs F1 Score 산점도
        ax = axes[0]
        for i, (t, f1, p, name) in enumerate(zip(times, f1s, precisions, methods)):
            color = self.colors[i % len(self.colors)]
            ax.scatter(t, f1, s=150, c=color, edgecolors='black', linewidths=0.5,
                       zorder=3)
            ax.annotate(name, (t, f1), fontsize=8,
                        xytext=(8, 5), textcoords='offset points')

        ax.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="F1 = 0.8")
        ax.set_xlabel("Execution Time (seconds)", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title("계산비용 vs 성능 Trade-off", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # 2) Feature당 처리 시간 비교
        ax2 = axes[1]
        n_features = 500
        per_feature_ms = [(t / n_features) * 1000 for t in times]

        sorted_idx = np.argsort(per_feature_ms)
        sorted_methods = [methods[i] for i in sorted_idx]
        sorted_pf = [per_feature_ms[i] for i in sorted_idx]
        sorted_f1 = [f1s[i] for i in sorted_idx]

        bar_colors = ["#2ca02c" if f1 >= 0.8 else "#ff7f0e" if f1 >= 0.5 else "#d62728"
                      for f1 in sorted_f1]
        bars = ax2.barh(range(len(sorted_methods)), sorted_pf,
                        color=bar_colors, edgecolor="black", linewidth=0.5)
        ax2.set_yticks(range(len(sorted_methods)))
        ax2.set_yticklabels(sorted_methods, fontsize=10)
        ax2.set_xlabel("Per Feature Processing Time (ms)", fontsize=11)
        ax2.set_title("Feature당 처리 시간", fontsize=13)
        ax2.grid(True, alpha=0.3, axis="x")

        for bar, pf, f1 in zip(bars, sorted_pf, sorted_f1):
            ax2.text(pf + max(sorted_pf) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{pf:.2f}ms | F1={f1:.2f}",
                     va="center", fontsize=9)

        legend_elements = [
            Patch(facecolor="#2ca02c", label="F1 >= 0.8"),
            Patch(facecolor="#ff7f0e", label="0.5 <= F1 < 0.8"),
            Patch(facecolor="#d62728", label="F1 < 0.5"),
        ]
        ax2.legend(handles=legend_elements, fontsize=9, loc="lower right")

        fig.suptitle("계산비용 비교 분석", fontsize=15, y=1.02)
        fig.tight_layout()
        fig.savefig(
            self.output_dir / "cost_analysis.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

    def plot_performance_heatmap(self, evaluator):
        """방법 x anomaly 유형 Recall 히트맵"""
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
            row.append(r.overall_metrics.get("f1", 0))
            data.append(row)

        columns = anomaly_types + ["Overall F1"]
        df = pd.DataFrame(data, index=methods, columns=columns)

        fig, ax = plt.subplots(figsize=(14, max(6, len(methods) * 0.8)))
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

        ax.axhline(y=0.8, color="red", linestyle="--", linewidth=2,
                    alpha=0.7, label="Precision 0.8 target")

        ax.set_xlabel("Detection Method")
        ax.set_ylabel("Score")
        ax.set_title("Method Performance Comparison (Precision / Recall / F1)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

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
        ax.set_title("ROC Curves", fontsize=14)
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
        """anomaly 유형별 방법 성능 비교"""
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

            sorted_idx = np.argsort(recalls)[::-1]
            sorted_methods = [methods[i] for i in sorted_idx]
            sorted_recalls = [recalls[i] for i in sorted_idx]

            bar_colors = ["#2ca02c" if v >= 0.8 else "#d62728" if v < 0.5 else "#ff7f0e"
                          for v in sorted_recalls]
            bars = ax.barh(range(len(sorted_methods)), sorted_recalls, color=bar_colors)
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
        """실행 시간 비교"""
        methods = [r.method_name for r in evaluator.results]
        times = [r.execution_time for r in evaluator.results]
        f1s = [r.overall_metrics.get("f1", 0) for r in evaluator.results]

        sorted_idx = np.argsort(times)
        methods = [methods[i] for i in sorted_idx]
        times = [times[i] for i in sorted_idx]
        f1s = [f1s[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(methods)), times, color="#1f77b4", alpha=0.8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=10)
        ax.set_xlabel("Execution Time (seconds)")
        ax.set_title("실행 시간 비교 (F1 Score 표시)", fontsize=14)
        if max(times) / (min(t for t in times if t > 0) + 1e-10) > 10:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3, axis="x")

        for bar, t, f1 in zip(bars, times, f1s):
            ax.text(
                t * 1.1 if t > 0 else 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.3f}s | F1={f1:.2f}",
                va="center", fontsize=9
            )

        fig.tight_layout()
        fig.savefig(
            self.output_dir / "execution_time.png",
            dpi=self.dpi, bbox_inches="tight"
        )
        plt.close(fig)

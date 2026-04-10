"""
Visualization for LSTM+CatBoost Walk-Forward Results

Usage:
    uv run python scripts/plot_lstm_results.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
# UseDejaVu Sans which supports Latin characters well and avoids CJK font issues
plt.rcParams["font.family"] = "DejaVu Sans"

from pathlib import Path


def plot_results(results: list, output_prefix: str = "walkforward_lstm") -> None:
    """绘制结果图表"""
    if not results:
        print("没有结果可绘图")
        return

    windows = range(len(results))

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("LSTM+CatBoost Walk-Forward Results", fontsize=16, fontweight="bold")

    # 1. Macro F1 (主指标)
    ax1 = axes[0, 0]
    macro_f1 = [r["macro_f1"] for r in results]
    macro_f1_post = [r["macro_f1_post"] for r in results]
    x = list(windows)
    width = 0.35
    ax1.bar([i - width / 2 for i in x], macro_f1, width, label="Raw", color="steelblue", alpha=0.8)
    ax1.bar([i + width / 2 for i in x], macro_f1_post, width, label="Post", color="forestgreen", alpha=0.8)
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Macro F1")
    ax1.set_title("Macro F1 Score (Primary Metric)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim([0, 1.0])
    ax1.set_xticks(list(x))
    for i, v in enumerate(macro_f1_post):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # 2. Accuracy
    ax2 = axes[0, 1]
    acc = [r["accuracy"] for r in results]
    acc_post = [r["accuracy_post"] for r in results]
    ax2.bar([i - width / 2 for i in x], acc, width, label="Raw", color="steelblue", alpha=0.8)
    ax2.bar([i + width / 2 for i in x], acc_post, width, label="Post", color="forestgreen", alpha=0.8)
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim([0, 1.0])
    ax2.set_xticks(list(x))

    # 3. Long/Short Recall
    ax3 = axes[1, 0]
    long_r = [r["long_recall"] for r in results]
    short_r = [r["short_recall"] for r in results]
    ax3.bar([i - width / 2 for i in x], long_r, width, label="Long Recall", color="crimson", alpha=0.8)
    ax3.bar([i + width / 2 for i in x], short_r, width, label="Short Recall", color="royalblue", alpha=0.8)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("Recall")
    ax3.set_title("Long/Short Recall")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim([0, 1.1])
    ax3.set_xticks(list(x))
    for i, (lr, sr) in enumerate(zip(long_r, short_r)):
        ax3.text(i - width / 2, lr + 0.02, f"{lr:.2f}", ha="center", fontsize=8)
        ax3.text(i + width / 2, sr + 0.02, f"{sr:.2f}", ha="center", fontsize=8)

    # 4. Long/Short Precision
    ax4 = axes[1, 1]
    long_p = [r["long_precision"] for r in results]
    short_p = [r["short_precision"] for r in results]
    ax4.bar([i - width / 2 for i in x], long_p, width, label="Long Precision", color="crimson", alpha=0.8)
    ax4.bar([i + width / 2 for i in x], short_p, width, label="Short Precision", color="royalblue", alpha=0.8)
    ax4.set_xlabel("Fold")
    ax4.set_ylabel("Precision")
    ax4.set_title("Long/Short Precision")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_ylim([0, 1.1])
    ax4.set_xticks(list(x))
    for i, (lp, sp) in enumerate(zip(long_p, short_p)):
        ax4.text(i - width / 2, lp + 0.02, f"{lp:.2f}", ha="center", fontsize=8)
        ax4.text(i + width / 2, sp + 0.02, f"{sp:.2f}", ha="center", fontsize=8)

    # 5. F1 by Class
    ax5 = axes[2, 0]
    f1_down = [r["f1_down"] for r in results]
    f1_neutral = [r["f1_neutral"] for r in results]
    f1_up = [r["f1_up"] for r in results]
    w = 0.25
    ax5.bar([i - w for i in x], f1_down, w, label="DOWN F1", color="royalblue", alpha=0.8)
    ax5.bar(x, f1_neutral, w, label="NEUTRAL F1", color="gray", alpha=0.8)
    ax5.bar([i + w for i in x], f1_up, w, label="UP F1", color="crimson", alpha=0.8)
    ax5.set_xlabel("Fold")
    ax5.set_ylabel("F1 Score")
    ax5.set_title("F1 Score by Class")
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")
    ax5.set_ylim([0, 1.0])
    ax5.set_xticks(list(x))

    # 6. Confusion Matrix (Average)
    ax6 = axes[2, 1]
    cm_raw = np.array(results[0]["confusion_matrix_raw"])
    for r in results[1:]:
        cm_raw += np.array(r["confusion_matrix_raw"])
    cm_avg = cm_raw / len(results)

    im = ax6.imshow(cm_avg, cmap="Blues", aspect="auto")
    ax6.set_xticks([0, 1, 2])
    ax6.set_yticks([0, 1, 2])
    ax6.set_xticklabels(["DOWN", "NEUTRAL", "UP"])
    ax6.set_yticklabels(["DOWN", "NEUTRAL", "UP"])
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Actual")
    ax6.set_title("Confusion Matrix (Average)")

    for i in range(3):
        for j in range(3):
            color = "white" if cm_avg[i, j] > cm_avg.max() / 2 else "black"
            ax6.text(j, i, f"{cm_avg[i, j]:.0f}", ha="center", va="center", color=color, fontsize=12)

    plt.colorbar(im, ax=ax6)

    plt.tight_layout()
    plot_path = f"{output_prefix}_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存到 {plot_path}")


def main() -> None:
    # 读取结果
    result_path = Path("models/walkforward_lstm_results.json")
    if not result_path.exists():
        print(f"结果文件不存在: {result_path}")
        return

    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]
    config = data["config"]

    print("=" * 60)
    print("LSTM+CatBoost Walk-Forward Results")
    print("=" * 60)
    print(f"\n配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\n结果汇总:")
    for i, r in enumerate(results):
        print(f"  Fold {i}: Macro F1={r['macro_f1_post']:.4f}, Acc={r['accuracy_post']:.4f}")

    avg_macro_f1 = np.mean([r["macro_f1_post"] for r in results])
    avg_acc = np.mean([r["accuracy_post"] for r in results])
    avg_long_recall = np.mean([r["long_recall"] for r in results])
    avg_short_recall = np.mean([r["short_recall"] for r in results])

    print(f"\n平均值:")
    print(f"  Macro F1: {avg_macro_f1:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  Long Recall: {avg_long_recall:.4f}")
    print(f"  Short Recall: {avg_short_recall:.4f}")

    # 绘图
    plot_results(results)


if __name__ == "__main__":
    main()

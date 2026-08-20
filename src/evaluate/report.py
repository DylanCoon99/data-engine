import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def generate_report(mined_metrics: dict, random_metrics: dict, output_dir: Path):
	"""Generate comparison plots and summary."""

	output_dir.mkdir(parents=True, exist_ok=True)

	# --- summary table ---
	summary = {
		"mined_mAP50": mined_metrics["mAP50"],
		"random_mAP50": random_metrics["mAP50"],
		"mined_mAP50-95": mined_metrics["mAP50-95"],
		"random_mAP50-95": random_metrics["mAP50-95"],
		"delta_mAP50": mined_metrics["mAP50"] - random_metrics["mAP50"],
		"delta_mAP50-95": mined_metrics["mAP50-95"] - random_metrics["mAP50-95"],
	}

	# print summary
	print("\n" + "=" * 60)
	print("  MINED vs RANDOM — EVALUATION RESULTS")
	print("=" * 60)
	print(f"  {'Metric':<20} {'Mined':>10} {'Random':>10} {'Delta':>10}")
	print("-" * 60)
	print(f"  {'mAP50':<20} {summary['mined_mAP50']:>10.4f} {summary['random_mAP50']:>10.4f} {summary['delta_mAP50']:>+10.4f}")
	print(f"  {'mAP50-95':<20} {summary['mined_mAP50-95']:>10.4f} {summary['random_mAP50-95']:>10.4f} {summary['delta_mAP50-95']:>+10.4f}")
	print("=" * 60)

	if summary["delta_mAP50"] > 0:
		print("  ✓ Mined selection outperforms random selection.")
	elif summary["delta_mAP50"] < 0:
		print("  ✗ Random selection outperforms mined selection.")
	else:
		print("  – No difference between mined and random selection.")
	print()

	# save summary as JSON
	summary_path = output_dir / "summary.json"
	with open(summary_path, "w") as f:
		json.dump(summary, f, indent=2)
	logger.info("Saved summary to %s", summary_path)

	# --- per-class AP comparison ---
	mined_ap = mined_metrics["per_class_ap"]
	random_ap = random_metrics["per_class_ap"]

	classes = sorted(mined_ap.keys())
	mined_vals = [mined_ap[c] for c in classes]
	random_vals = [random_ap[c] for c in classes]

	# print per-class table
	print(f"  {'Class':<20} {'Mined':>10} {'Random':>10} {'Delta':>10}")
	print("-" * 60)
	for c in classes:
		delta = mined_ap[c] - random_ap[c]
		print(f"  {c:<20} {mined_ap[c]:>10.4f} {random_ap[c]:>10.4f} {delta:>+10.4f}")
	print()

	# --- bar chart ---
	fig, ax = plt.subplots(figsize=(12, 6))
	x = range(len(classes))
	width = 0.35

	bars_mined = ax.bar([i - width / 2 for i in x], mined_vals, width, label="Mined", color="#2196F3")
	bars_random = ax.bar([i + width / 2 for i in x], random_vals, width, label="Random", color="#FF9800")

	ax.set_ylabel("AP (mAP50-95)")
	ax.set_title("Per-Class AP: Mined vs Random Selection")
	ax.set_xticks(x)
	ax.set_xticklabels(classes, rotation=45, ha="right")
	ax.legend()
	ax.set_ylim(0, 1)
	plt.tight_layout()

	chart_path = output_dir / "per_class_ap.png"
	plt.savefig(chart_path, dpi=150)
	plt.close()
	logger.info("Saved per-class chart to %s", chart_path)

	# --- overall mAP bar chart ---
	fig, ax = plt.subplots(figsize=(6, 4))
	arms = ["Mined", "Random"]
	map50 = [summary["mined_mAP50"], summary["random_mAP50"]]
	map50_95 = [summary["mined_mAP50-95"], summary["random_mAP50-95"]]

	x = range(len(arms))
	ax.bar([i - width / 2 for i in x], map50, width, label="mAP50", color="#2196F3")
	ax.bar([i + width / 2 for i in x], map50_95, width, label="mAP50-95", color="#FF9800")

	ax.set_ylabel("mAP")
	ax.set_title("Overall mAP: Mined vs Random")
	ax.set_xticks(x)
	ax.set_xticklabels(arms)
	ax.legend()
	ax.set_ylim(0, 1)
	plt.tight_layout()

	overall_path = output_dir / "overall_map.png"
	plt.savefig(overall_path, dpi=150)
	plt.close()
	logger.info("Saved overall chart to %s", overall_path)

	return summary

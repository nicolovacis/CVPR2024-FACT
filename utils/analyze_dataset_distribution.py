#!/usr/bin/env python3
"""
Analyze class distribution for a FACT-style dataset.

Inputs:
  --data_dir: path to preprocessed dataset (contains groundTruth/, mapping.txt)
  --output_dir: base output dir where <dataset_name>_analysis/ will be created
Optional:
  --dataset_name: name used for output folder (default: basename of data_dir)
  --fps: labels-per-second (default: 1.0). If each line is 1 second -> fps=1.
         If each line is a frame at 25fps -> fps=25.

Outputs (inside <output_dir>/<dataset_name>_analysis/):
  - summary_global.csv
  - summary_per_video.csv
  - plots:
      * global_seconds_per_class.png
      * global_percent_per_class.png
      * per_video_total_duration.png
      * per_video_top_class.png
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_mapping(mapping_path: Path):
    """
    mapping.txt format expected: "<id> <label>" per line
    returns:
      id_to_label: dict[int,str]
      label_to_id: dict[str,int]
      labels_in_id_order: list[str]
    """
    id_to_label = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            label = " ".join(parts[1:])
            id_to_label[idx] = label

    if not id_to_label:
        raise RuntimeError(f"mapping.txt parsed empty or invalid: {mapping_path}")

    labels_in_id_order = [id_to_label[i] for i in sorted(id_to_label.keys())]
    label_to_id = {v: k for k, v in id_to_label.items()}
    return id_to_label, label_to_id, labels_in_id_order


def iter_groundtruth_files(gt_dir: Path):
    for p in sorted(gt_dir.glob("*.txt")):
        yield p


def read_labels_file(gt_file: Path):
    # Each line is a label string
    with gt_file.open("r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip() != ""]
    return labels


def safe_seconds(count: int, fps: float) -> float:
    # seconds = timesteps / fps
    return count / fps if fps > 0 else 0.0


def write_csv(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def plot_bar(x_labels, values, title, ylabel, out_path: Path, rotate=45):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(x_labels)), values)
    plt.xticks(range(len(x_labels)), x_labels, rotation=rotate, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_barh(x_labels, values, title, xlabel, out_path: Path):
    plt.figure(figsize=(10, max(4, 0.35 * len(x_labels))))
    plt.barh(range(len(x_labels)), values)
    plt.yticks(range(len(x_labels)), x_labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to preprocessed dataset directory")
    ap.add_argument("--output_dir", required=True, help="Base output directory")
    ap.add_argument("--dataset_name", default=None, help="Dataset name for output folder")
    ap.add_argument("--fps", type=float, default=1.0, help="Labels-per-second (default 1.0)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_base = Path(args.output_dir).resolve()
    dataset_name = args.dataset_name or data_dir.name

    gt_dir = data_dir / "groundTruth"
    mapping_path = data_dir / "mapping.txt"

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing groundTruth dir: {gt_dir}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing mapping.txt: {mapping_path}")

    analysis_dir = out_base / f"{dataset_name}_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    id_to_label, label_to_id, labels_in_id_order = read_mapping(mapping_path)

    # Global + per-video stats
    global_counts = Counter()
    per_video_counts = {}
    per_video_total = {}
    unknown_labels = Counter()

    gt_files = list(iter_groundtruth_files(gt_dir))
    if not gt_files:
        raise RuntimeError(f"No .txt files found in {gt_dir}")

    for gt_file in gt_files:
        vid = gt_file.stem
        labels = read_labels_file(gt_file)
        c = Counter(labels)

        # Track unknown labels (not in mapping)
        for lbl, cnt in c.items():
            if lbl not in label_to_id:
                unknown_labels[lbl] += cnt

        per_video_counts[vid] = c
        per_video_total[vid] = sum(c.values())
        global_counts.update(c)

    # If mapping misses some label, still include them at the end (so you SEE the bug)
    all_labels = list(labels_in_id_order)
    for lbl in sorted(global_counts.keys()):
        if lbl not in label_to_id and lbl not in all_labels:
            all_labels.append(lbl)

    # Build global summary
    total_steps = sum(global_counts.values())
    total_seconds = safe_seconds(total_steps, args.fps)

    global_rows = []
    seconds_per_class = []
    percent_per_class = []

    for lbl in all_labels:
        cnt = global_counts.get(lbl, 0)
        sec = safe_seconds(cnt, args.fps)
        pct = (cnt / total_steps * 100.0) if total_steps > 0 else 0.0
        global_rows.append([lbl, cnt, f"{sec:.3f}", f"{pct:.3f}"])
        seconds_per_class.append(sec)
        percent_per_class.append(pct)

    write_csv(
        analysis_dir / "summary_global.csv",
        header=["class", "timesteps", "seconds", "percent"],
        rows=global_rows,
    )

    # Per-video summary (total + top class)
    per_video_rows = []
    per_video_duration_sec = []
    per_video_names = []
    per_video_top_class = []
    per_video_top_pct = []

    for vid in sorted(per_video_counts.keys()):
        c = per_video_counts[vid]
        steps = per_video_total[vid]
        sec = safe_seconds(steps, args.fps)

        if steps > 0:
            top_lbl, top_cnt = c.most_common(1)[0]
            top_pct = top_cnt / steps * 100.0
        else:
            top_lbl, top_pct = "NA", 0.0

        per_video_rows.append([vid, steps, f"{sec:.3f}", top_lbl, f"{top_pct:.3f}"])
        per_video_names.append(vid)
        per_video_duration_sec.append(sec)
        per_video_top_class.append(top_lbl)
        per_video_top_pct.append(top_pct)

    write_csv(
        analysis_dir / "summary_per_video.csv",
        header=["video", "timesteps", "seconds", "top_class", "top_class_percent"],
        rows=per_video_rows,
    )

    # Plots
    plot_bar(
        x_labels=all_labels,
        values=seconds_per_class,
        title=f"{dataset_name}: total seconds per class (fps={args.fps})",
        ylabel="seconds",
        out_path=analysis_dir / "global_seconds_per_class.png",
        rotate=45,
    )

    plot_bar(
        x_labels=all_labels,
        values=percent_per_class,
        title=f"{dataset_name}: class percentage",
        ylabel="percent",
        out_path=analysis_dir / "global_percent_per_class.png",
        rotate=45,
    )

    # Per-video duration plot (sorted)
    order = sorted(range(len(per_video_names)), key=lambda i: per_video_duration_sec[i], reverse=True)
    names_sorted = [per_video_names[i] for i in order]
    dur_sorted = [per_video_duration_sec[i] for i in order]
    plot_barh(
        x_labels=names_sorted,
        values=dur_sorted,
        title=f"{dataset_name}: video durations (seconds, fps={args.fps})",
        xlabel="seconds",
        out_path=analysis_dir / "per_video_total_duration.png",
    )

    # Top class per video (count how many videos have each top class)
    top_class_counter = Counter(per_video_top_class)
    top_labels = [lbl for lbl in all_labels if lbl in top_class_counter] + [
        lbl for lbl in sorted(top_class_counter.keys()) if lbl not in all_labels
    ]
    top_vals = [top_class_counter[lbl] for lbl in top_labels]
    plot_bar(
        x_labels=top_labels,
        values=top_vals,
        title=f"{dataset_name}: number of videos where class is the top label",
        ylabel="#videos",
        out_path=analysis_dir / "per_video_top_class.png",
        rotate=45,
    )

    # Write a small warnings file if labels not in mapping
    if unknown_labels:
        warn_path = analysis_dir / "WARN_unknown_labels.txt"
        with warn_path.open("w", encoding="utf-8") as f:
            f.write("Labels present in groundTruth but missing in mapping.txt:\n")
            for lbl, cnt in unknown_labels.most_common():
                f.write(f"- {lbl}: {cnt} timesteps\n")

    # Print final pointer (useful in logs)
    print(f"[OK] Wrote analysis to: {analysis_dir}")
    print(f"[OK] Total timesteps: {total_steps}, total seconds: {total_seconds:.3f} (fps={args.fps})")


if __name__ == "__main__":
    main()

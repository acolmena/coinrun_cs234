#!/usr/bin/env python3
"""
Export TensorBoard scalar summaries to CSV.

Produces:
1) a long/raw CSV with one row per scalar event
2) an aggregated CSV grouped by (run_id, tag, step), averaging across files/ranks
"""

from __future__ import print_function

import argparse
import csv
import glob
import math
import os

import tensorflow as tf


def parse_run_and_rank(log_subdir):
    """
    TB log subdirectories are typically named like "<run_id>_<rank>".
    If suffix is not an int, keep entire name as run_id and rank=-1.
    """
    if "_" not in log_subdir:
        return log_subdir, -1

    run_id, maybe_rank = log_subdir.rsplit("_", 1)
    try:
        return run_id, int(maybe_rank)
    except ValueError:
        return log_subdir, -1


def maybe_tensor_to_float(summary_value):
    if summary_value.HasField("simple_value"):
        return float(summary_value.simple_value)

    if summary_value.HasField("tensor"):
        try:
            arr = tf.make_ndarray(summary_value.tensor)
            if arr.size >= 1:
                return float(arr.reshape(-1)[0])
        except Exception:
            return None

    return None


def stddev(values):
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / float(n)
    var = sum((x - mean) ** 2 for x in values) / float(n)
    return math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default="/tmp/tensorflow",
        help="TensorBoard log root directory",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="",
        help="Comma-separated run IDs to include (e.g. baseline_s1,grok_s1). Empty means all.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="",
        help="Comma-separated scalar tags to include. Empty means all scalar tags.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/scalars_long.csv",
        help="Output CSV path for raw/long rows",
    )
    parser.add_argument(
        "--out-agg",
        type=str,
        default="results/scalars_agg.csv",
        help="Output CSV path for aggregated rows",
    )
    args = parser.parse_args()

    runs_filter = set([x.strip() for x in args.runs.split(",") if x.strip()])
    tags_filter = set([x.strip() for x in args.tags.split(",") if x.strip()])

    event_files = glob.glob(os.path.join(args.logdir, "**", "events.out.tfevents*"), recursive=True)
    if not event_files:
        raise RuntimeError("No TensorBoard event files found under {}".format(args.logdir))

    rows = []

    for event_file in event_files:
        log_subdir = os.path.basename(os.path.dirname(event_file))
        run_id, rank = parse_run_and_rank(log_subdir)

        if runs_filter and run_id not in runs_filter:
            continue

        for event in tf.train.summary_iterator(event_file):
            if not event.summary or not event.summary.value:
                continue

            step = int(event.step)
            wall_time = float(event.wall_time)

            for sv in event.summary.value:
                tag = sv.tag
                if tags_filter and tag not in tags_filter:
                    continue

                value = maybe_tensor_to_float(sv)
                if value is None:
                    continue

                rows.append(
                    {
                        "run_id": run_id,
                        "rank": rank,
                        "tag": tag,
                        "step": step,
                        "value": value,
                        "wall_time": wall_time,
                        "event_file": event_file,
                    }
                )

    if not rows:
        raise RuntimeError("No scalar rows matched your filters.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_agg_dir = os.path.dirname(args.out_agg)
    if out_agg_dir:
        os.makedirs(out_agg_dir, exist_ok=True)

    # Raw/long CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "rank", "tag", "step", "value", "wall_time", "event_file"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Aggregated CSV: mean/std/count over files/ranks for each run/tag/step
    grouped = {}
    for r in rows:
        key = (r["run_id"], r["tag"], r["step"])
        grouped.setdefault(key, []).append(r["value"])

    agg_rows = []
    for (run_id, tag, step), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        mean_val = sum(vals) / float(len(vals))
        agg_rows.append(
            {
                "run_id": run_id,
                "tag": tag,
                "step": step,
                "mean": mean_val,
                "std": stddev(vals),
                "count": len(vals),
            }
        )

    with open(args.out_agg, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "tag", "step", "mean", "std", "count"])
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)

    print("Wrote {} raw rows to {}".format(len(rows), args.out))
    print("Wrote {} aggregated rows to {}".format(len(agg_rows), args.out_agg))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Filter and split bridge_orig_lerobot dataset.

Step 1 (filter):
  - Load task instructions from tasks.parquet, normalize text, classify each
    task as clean or flagged (gibberish / single-word / question / non-English /
    pattern-matched), merge duplicate texts, then write:
      bridge_tasks_clean.json   — clean tasks
      bridge_tasks_flagged.json — flagged tasks

Step 2 (split, unless --filter_only):
  - Create {name}_clean dataset dir (symlinked data/videos, filtered meta).
  - Optionally create {name}_dirty dir with --save_dirty.
  - Optionally trim static head/tail frames with --static_threshold.

Usage:
    CMD="python filter_bridge_dataset.py"

    # Filter only (produce JSONs, no dataset split)
    $CMD --root /path/to/dataset --filter_only

    # Filter + split (clean only, default)
    $CMD --root /path/to/dataset

    # Filter + split (clean + dirty)
    $CMD --root /path/to/dataset --save_dirty

    # Trim static frames (action[:6] near zero) from clean episodes
    $CMD --root /path/to/dataset --static_threshold 5e-4

    # Custom output locations
    $CMD --root /path/to/dataset --output_dir /tmp/results --output_name my_bridge

Arguments:
    --root               Path to the original LeRobot dataset
    --output_dir          Dir for JSON outputs (default: script dir)
    --output_name         Base name for split dirs (default: derived from --root)
    --static_threshold    Trim static head/tail frames; 0 = off (default), try 5e-4
    --save_dirty          Also create the dirty (flagged) split
    --filter_only         Only run filter step, skip split
"""

import argparse
import glob
import json
import os
import re
import shutil

import nltk
import numpy as np
import pandas as pd

nltk.download("words", quiet=True)
from nltk.corpus import words as nltk_words  # noqa: E402

ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())
TASK_WORD_ALLOWLIST = {
    # Common robotics task words that are valid English but missing from nltk.words.
    "fridge",
}

FLAGGED_PATTERNS = [
    r".*image.*",
    r".*https?.*",
    r".*\bnot\b.*",
    r".*\bnothing\b.*",
    r".*\banything\b.*",
    r".*\bdidn.?t\b.*",
    r".*\bdoesn.?t\b.*",
    r".*\baren.?t\b.*",
    r".*\bupload\b.*",
    r".*\bpicture\b.*",
    r"\(.*\)",
]


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_task(text: str) -> str:
    t = text.strip()
    t = t.strip("\"'''")
    t = t.replace("\u201a\u00c4\u00f4", "'")
    t = re.sub(r"\s{2,}", " ", t)
    if "." in t and " " not in t:
        t = t.replace(".", " ")
    if "_" in t and " " not in t:
        t = t.replace("_", " ")
    return t.strip()


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def _clean_word(word: str) -> str:
    return word.lower().strip(".,!?;:'\"()-/\\")


def _is_english(word: str) -> bool:
    w = _clean_word(word)
    if not w:
        return True
    if w in TASK_WORD_ALLOWLIST:
        return True
    if not w.isalpha():
        alpha_part = "".join(c for c in w if c.isalpha())
        if len(alpha_part) >= 3 and alpha_part not in ENGLISH_WORDS:
            return False
        return True
    if len(w) <= 2:
        return True
    if w in ENGLISH_WORDS:
        return True
    for suffix in ("s", "ed", "ing", "ly", "er", "est", "tion", "ness"):
        stem = w.removesuffix(suffix)
        if len(stem) >= 2 and stem in ENGLISH_WORDS:
            return True
    return False


def is_gibberish(text: str) -> tuple[bool, str]:
    t = text.strip()
    if not t:
        return False, ""
    words = t.split()
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if not alpha_words:
        return True, "no_alphabetic_words"
    non_english = [w for w in alpha_words if not _is_english(w)]
    ratio = len(non_english) / len(alpha_words)
    if len(alpha_words) <= 2 and len(non_english) >= 1:
        return (
            True,
            f"short_non_english({len(non_english)}/{len(alpha_words)}): {non_english[:8]}",
        )
    if len(alpha_words) >= 3 and ratio >= 0.6:
        return (
            True,
            f"non_english_words({len(non_english)}/{len(alpha_words)}): {non_english[:8]}",
        )
    if len(alpha_words) >= 6 and ratio >= 0.5:
        return (
            True,
            f"many_non_english({len(non_english)}/{len(alpha_words)}): {non_english[:8]}",
        )
    if re.search(r"(.)\1{4,}", t.lower()):
        return True, "excessive_char_repetition"
    short_non_eng = [w for w in alpha_words if len(_clean_word(w)) <= 3 and not _is_english(w)]
    if len(alpha_words) >= 5 and len(short_non_eng) / len(alpha_words) >= 0.6:
        return (
            True,
            f"many_short_nonsense({len(short_non_eng)}/{len(alpha_words)}): {short_non_eng[:8]}",
        )
    return False, ""


def is_pattern_flagged(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return True
    return any(re.match(pat, t) for pat in FLAGGED_PATTERNS)


def is_single_word(text: str) -> bool:
    words = text.strip().split()
    return sum(1 for w in words if any(c.isalpha() for c in w)) == 1


def is_question(text: str) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    t_lower = t.lower()
    q_starts = (
        "what ",
        "how ",
        "why ",
        "where ",
        "when ",
        "who ",
        "which ",
        "did ",
        "does ",
        "do ",
        "can ",
        "could ",
        "would ",
        "should ",
        "will ",
    )
    return any(t_lower.startswith(q) for q in q_starts)


def is_non_english(text: str) -> bool:
    return sum(1 for c in text.strip() if not c.isascii() and c.isalpha()) >= 2


def classify_task(text: str) -> list[str]:
    reasons = []
    if is_pattern_flagged(text):
        reasons.append("pattern")
    gib, _ = is_gibberish(text)
    if gib:
        reasons.append("gibberish")
    if is_question(text):
        reasons.append("question")
    if is_non_english(text):
        reasons.append("non_english")
    if is_single_word(text) and not reasons:
        reasons.append("single_word")
    return reasons


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_compact_json(path, description, num_tasks, num_episodes, tasks):
    with open(path, "w") as f:
        f.write("{\n")
        f.write(f'  "description": {json.dumps(description)},\n')
        f.write(f'  "num_tasks": {num_tasks},\n')
        f.write(f'  "num_episodes": {num_episodes},\n')
        f.write('  "tasks": [\n')
        for i, t in enumerate(tasks):
            line = json.dumps(t, ensure_ascii=False)
            comma = "," if i < len(tasks) - 1 else ""
            f.write(f"    {line}{comma}\n")
        f.write("  ]\n")
        f.write("}\n")


def load_task_text_col(root):
    tasks_df = pd.read_parquet(os.path.join(root, "meta", "tasks.parquet")).reset_index()
    text_candidates = [c for c in tasks_df.columns if c != "task_index"]
    if text_candidates:
        return tasks_df, text_candidates[0]
    if tasks_df["task_index"].dtype == object:
        tasks_df["task_text"] = tasks_df["task_index"]
        tasks_df["task_index"] = range(len(tasks_df))
        return tasks_df, "task_text"
    raise RuntimeError(f"Cannot find task text column: {tasks_df.columns.tolist()}")


# ---------------------------------------------------------------------------
# Step 1: Filter tasks
# ---------------------------------------------------------------------------


def filter_tasks(root, output_dir):
    print(f"nltk dictionary: {len(ENGLISH_WORDS)} words")
    tasks_df, text_col = load_task_text_col(root)
    print(f"Total unique tasks: {len(tasks_df)}")

    data_files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    print(f"Reading {len(data_files)} data parquet files ...")
    dfs = [pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in data_files]
    data_df = pd.concat(dfs, ignore_index=True)

    ep_tasks = data_df.drop_duplicates("episode_index")[["episode_index", "task_index"]]
    task_counts = ep_tasks.groupby("task_index").size().reset_index(name="num_episodes")
    total_episodes = ep_tasks["episode_index"].nunique()
    print(f"Total episodes: {total_episodes}")

    merged = task_counts.merge(tasks_df, on="task_index", how="left")
    merged[text_col] = merged[text_col].apply(normalize_task)
    merged["flag_reasons"] = merged[text_col].apply(classify_task)
    merged["flagged"] = merged["flag_reasons"].apply(lambda r: len(r) > 0)

    # Merge duplicate task texts
    merged["task_normalized"] = merged[text_col].str.strip().str.lower()

    def _merge_group(group):
        return {
            "task": group[text_col].iloc[0].strip(),
            "task_indices": sorted(group["task_index"].tolist()),
            "num_episodes": int(group["num_episodes"].sum()),
            "flagged": bool(group["flagged"].any()),
            "flag_reasons": sorted(set(r for reasons in group["flag_reasons"] for r in reasons)),
        }

    grouped = merged.groupby("task_normalized").apply(_merge_group, include_groups=False).tolist()

    clean_tasks = sorted([t for t in grouped if not t["flagged"]], key=lambda x: -x["num_episodes"])
    flagged_tasks = sorted([t for t in grouped if t["flagged"]], key=lambda x: -x["num_episodes"])

    total_clean_eps = sum(t["num_episodes"] for t in clean_tasks)
    total_flagged_eps = sum(t["num_episodes"] for t in flagged_tasks)

    os.makedirs(output_dir, exist_ok=True)
    clean_path = os.path.join(output_dir, "bridge_tasks_clean.json")
    save_compact_json(
        clean_path,
        "Clean task instructions",
        len(clean_tasks),
        total_clean_eps,
        clean_tasks,
    )

    flagged_path = os.path.join(output_dir, "bridge_tasks_flagged.json")
    save_compact_json(
        flagged_path,
        "Flagged task instructions",
        len(flagged_tasks),
        total_flagged_eps,
        flagged_tasks,
    )

    print(
        f"\nClean:   {len(clean_tasks)} tasks / {total_clean_eps} episodes ({total_clean_eps / total_episodes * 100:.1f}%)"
    )
    print(
        f"Flagged: {len(flagged_tasks)} tasks / {total_flagged_eps} episodes ({total_flagged_eps / total_episodes * 100:.1f}%)"
    )
    for label in ("pattern", "gibberish", "single_word", "question", "non_english"):
        n = sum(1 for t in flagged_tasks if label in t["flag_reasons"])
        if n:
            print(f"  - {label}: {n}")
    print(f"Saved: {clean_path}")
    print(f"Saved: {flagged_path}")

    return clean_path, flagged_path


# ---------------------------------------------------------------------------
# Step 2: Split dataset (symlinked data/videos, filtered meta)
# ---------------------------------------------------------------------------


def build_episode_task_map(root):
    data_files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    dfs = [pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in data_files]
    data_df = pd.concat(dfs, ignore_index=True)
    return data_df.drop_duplicates("episode_index").set_index("episode_index")["task_index"].to_dict()


def create_split(orig_root, output_root, episode_set, orig_info, orig_episodes_df, label, trims=None):
    """Create a dataset split by zeroing out frame ranges for excluded episodes.

    LeRobot uses episode_index as a positional index into the episodes table,
    so we must keep ALL rows. Instead of removing rows, we zero out
    dataset_from_index and dataset_to_index for excluded episodes, which makes
    build_episode_spans compute valid_len <= 0 and skip them.

    If *trims* is provided, head/tail static frames are trimmed by adjusting
    dataset_from_index / dataset_to_index for included episodes.
    """
    os.makedirs(output_root, exist_ok=True)
    for subdir in ["data", "videos"]:
        src = os.path.join(orig_root, subdir)
        dst = os.path.join(output_root, subdir)
        if os.path.islink(dst):
            os.remove(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        elif os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    modified_eps = orig_episodes_df.copy()
    excluded_mask = ~modified_eps["episode_index"].isin(episode_set)
    has_range = "dataset_from_index" in modified_eps.columns and "dataset_to_index" in modified_eps.columns

    if has_range:
        modified_eps.loc[excluded_mask, "dataset_from_index"] = 0
        modified_eps.loc[excluded_mask, "dataset_to_index"] = 0

        if trims:
            for idx in modified_eps.index:
                ep_id = modified_eps.at[idx, "episode_index"]
                if ep_id in episode_set and ep_id in trims:
                    head_trim, tail_trim = trims[ep_id]
                    modified_eps.at[idx, "dataset_from_index"] += head_trim
                    modified_eps.at[idx, "dataset_to_index"] -= tail_trim
                    if modified_eps.at[idx, "dataset_from_index"] >= modified_eps.at[idx, "dataset_to_index"]:
                        modified_eps.at[idx, "dataset_from_index"] = 0
                        modified_eps.at[idx, "dataset_to_index"] = 0

    total_frames = 0
    included = modified_eps[~excluded_mask]
    if has_range:
        total_frames = int((included["dataset_to_index"] - included["dataset_from_index"]).sum())

    meta_dir = os.path.join(output_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    ep_meta_dir = os.path.join(meta_dir, "episodes", "chunk-000")
    os.makedirs(ep_meta_dir, exist_ok=True)
    modified_eps.to_parquet(os.path.join(ep_meta_dir, "file-000.parquet"), index=False)

    shutil.copy2(
        os.path.join(orig_root, "meta", "tasks.parquet"),
        os.path.join(meta_dir, "tasks.parquet"),
    )
    stats_src = os.path.join(orig_root, "meta", "stats.json")
    if os.path.exists(stats_src):
        shutil.copy2(stats_src, os.path.join(meta_dir, "stats.json"))

    new_info = orig_info.copy()
    new_info["total_episodes"] = len(orig_episodes_df)
    if total_frames > 0:
        new_info["total_frames"] = total_frames
    new_info["splits"] = {"train": f"0:{len(orig_episodes_df)}"}
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(new_info, f, indent=4)

    print(
        f"  {label}: {len(episode_set)} valid episodes (of {len(orig_episodes_df)} total), "
        f"{total_frames} frames -> {output_root}"
    )


def compute_static_trims(root, episode_set, threshold):
    """Compute per-episode head/tail static frame counts for trimming.

    A frame is "static" when max(abs(action[:6])) < threshold, where the
    first 6 action dims are translation + rotation.

    Returns (trims_dict, details_list):
      - trims_dict: {episode_index: (head_trim, tail_trim)}
      - details_list: list of dicts for the JSON report
    """
    data_files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    print(f"  Reading action data from {len(data_files)} files for static frame detection (threshold={threshold}) ...")
    dfs = [pd.read_parquet(f, columns=["episode_index", "action"]) for f in data_files]
    data_df = pd.concat(dfs, ignore_index=True)

    trims = {}
    details = []
    total_head, total_tail = 0, 0
    fully_static = 0
    total_checked = 0

    for ep_id, ep_df in data_df.groupby("episode_index"):
        if ep_id not in episode_set:
            continue
        total_checked += 1

        actions = np.stack(ep_df["action"].values)
        trans_rot = actions[:, :6]
        static_mask = np.abs(trans_rot).max(axis=1) < threshold
        n_static = int(static_mask.sum())
        ep_len = len(ep_df)

        if n_static == 0:
            continue

        head_trim = 0
        for i in range(ep_len):
            if static_mask[i]:
                head_trim += 1
            else:
                break

        tail_trim = 0
        for i in range(ep_len - 1, -1, -1):
            if static_mask[i]:
                tail_trim += 1
            else:
                break

        if head_trim + tail_trim >= ep_len:
            fully_static += 1
            trims[int(ep_id)] = (0, ep_len)
            details.append(
                {
                    "episode_index": int(ep_id),
                    "ep_len": ep_len,
                    "head": 0,
                    "tail": ep_len,
                    "fully_static": True,
                }
            )
            total_tail += ep_len
            continue

        if head_trim > 0 or tail_trim > 0:
            trims[int(ep_id)] = (head_trim, tail_trim)
            details.append(
                {
                    "episode_index": int(ep_id),
                    "ep_len": ep_len,
                    "head": head_trim,
                    "tail": tail_trim,
                    "fully_static": False,
                }
            )
            total_head += head_trim
            total_tail += tail_trim

    print(
        f"  {total_checked} eps checked, {len(trims)} trimmed (fully_static={fully_static}), frames: head={total_head} tail={total_tail} sum={total_head + total_tail}"
    )
    return trims, details


def save_static_report(path, threshold, details):
    """Save a JSON report of static frame trimming."""
    details_sorted = sorted(details, key=lambda x: -(x["head"] + x["tail"]))
    total_head = sum(d["head"] for d in details)
    total_tail = sum(d["tail"] for d in details)
    fully_static = sum(1 for d in details if d["fully_static"])

    with open(path, "w") as f:
        f.write("{\n")
        f.write('  "description": "Static frame trimming report",\n')
        f.write(f'  "threshold": {threshold},\n')
        f.write(f'  "episodes_trimmed": {len(details)},\n')
        f.write(f'  "episodes_fully_static": {fully_static},\n')
        f.write(
            f'  "total_frames_trimmed": {{"head": {total_head}, "tail": {total_tail}, "sum": {total_head + total_tail}}},\n'
        )
        f.write('  "episodes": [\n')
        for i, d in enumerate(details_sorted):
            line = json.dumps(d, ensure_ascii=False)
            comma = "," if i < len(details_sorted) - 1 else ""
            f.write(f"    {line}{comma}\n")
        f.write("  ]\n")
        f.write("}\n")
    print(f"Saved: {path}")


def split_dataset(
    orig_root,
    flagged_json,
    output_base,
    output_dir,
    output_name=None,
    static_threshold=0,
    save_dirty=False,
):
    with open(os.path.join(orig_root, "meta", "info.json")) as f:
        orig_info = json.load(f)
    ep_meta_files = sorted(
        glob.glob(
            os.path.join(orig_root, "meta", "episodes", "**", "*.parquet"),
            recursive=True,
        )
    )
    orig_episodes_df = pd.concat([pd.read_parquet(f) for f in ep_meta_files], ignore_index=True)

    with open(flagged_json) as f:
        flagged_indices = set()
        for t in json.load(f)["tasks"]:
            flagged_indices.update(t["task_indices"])

    ep_task_map = build_episode_task_map(orig_root)

    if "task_index" not in orig_episodes_df.columns:
        task_series = pd.Series(ep_task_map, name="task_index")
        task_series.index.name = "episode_index"
        orig_episodes_df = orig_episodes_df.merge(task_series.reset_index(), on="episode_index", how="left")

    all_episodes = set(orig_episodes_df["episode_index"].tolist())
    clean_eps, dirty_eps = set(), set()
    for ep in all_episodes:
        ti = ep_task_map.get(ep, -1)
        if ti in flagged_indices:
            dirty_eps.add(ep)
        else:
            clean_eps.add(ep)

    trims = None
    if static_threshold > 0:
        print("\nStep 2b: Detecting static head/tail frames ...")
        trims, details = compute_static_trims(orig_root, clean_eps, static_threshold)
        report_path = os.path.join(output_dir, "bridge_static_trims.json")
        save_static_report(report_path, static_threshold, details)

    base_name = output_name or os.path.basename(orig_root.rstrip("/"))
    print(f"\nSplitting: {len(clean_eps)} clean, {len(dirty_eps)} dirty (save_dirty={save_dirty})")
    create_split(
        orig_root,
        os.path.join(output_base, f"{base_name}_clean"),
        clean_eps,
        orig_info,
        orig_episodes_df,
        "CLEAN",
        trims,
    )
    if save_dirty:
        create_split(
            orig_root,
            os.path.join(output_base, f"{base_name}_dirty"),
            dirty_eps,
            orig_info,
            orig_episodes_df,
            "DIRTY",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Filter and split bridge dataset")
    parser.add_argument(
        "--root",
        default="/lustre/fsw/portfolios/dir/projects/dir_cosmos_base_lustre/bilyang/datasets/bridge_orig_lerobot_20260225",
    )
    parser.add_argument(
        "--output_json_dir",
        default=None,
        help="Dir for JSON outputs (default: same as this script)",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Base name for split dirs (default: derived from --root)",
    )
    parser.add_argument(
        "--output_data_dir",
        default=None,
        help="Dir for output split dataset (default: parent of root)",
    )
    parser.add_argument(
        "--static_threshold",
        type=float,
        default=0,
        help="Trim static head/tail frames where max(abs(action[:6])) < threshold (0=off, try 5e-4)",
    )
    parser.add_argument(
        "--save_dirty",
        action="store_true",
        help="Also create the dirty (flagged) split (default: off)",
    )
    parser.add_argument("--filter_only", action="store_true", help="Only run filter step, skip split")
    args = parser.parse_args()

    output_dir = args.output_json_dir or os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("Step 1: Filter tasks")
    print("=" * 80)
    clean_json, flagged_json = filter_tasks(args.root, output_dir)

    if args.static_threshold > 0 and args.filter_only:
        print()
        print("=" * 80)
        print("Step 2: Static frame detection (dry-run)")
        print("=" * 80)
        with open(flagged_json) as f:
            flagged_indices = set()
            for t in json.load(f)["tasks"]:
                flagged_indices.update(t["task_indices"])
        ep_task_map = build_episode_task_map(args.root)
        clean_eps = {ep for ep, ti in ep_task_map.items() if ti not in flagged_indices}
        _, details = compute_static_trims(args.root, clean_eps, args.static_threshold)
        report_path = os.path.join(output_dir, "bridge_static_trims.json")
        save_static_report(report_path, args.static_threshold, details)

    if not args.filter_only:
        print()
        print("=" * 80)
        print("Step 2: Split dataset")
        print("=" * 80)
        output_base = args.output_data_dir if args.output_data_dir else os.path.dirname(args.root.rstrip("/"))
        split_dataset(
            args.root,
            flagged_json,
            output_base,
            output_dir,
            args.output_name,
            args.static_threshold,
            args.save_dirty,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()

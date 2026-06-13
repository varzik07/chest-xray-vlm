"""
M1 — Data preparation for the chest X-ray VLM.

Reads the Indiana University Chest X-ray dataset:
  - reports : data/ecgen-radiology/*.xml   (3,955 doctor reports)
  - images  : data/images/*.png            (~7,470 X-ray images)

Each XML report has FINDINGS / IMPRESSION text and lists its image(s) via
<parentImage id="..."> where the id == the PNG filename (without .png).

Output:
  data/dataset.jsonl        one line per (image, report) pair
  data/train.jsonl
  data/val.jsonl
  data/test.jsonl

We split at the REPORT level (not the image level) so that two views of the
same patient never end up in different splits (that would leak information).

Run:  python src/prepare_data.py
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from sklearn.model_selection import train_test_split

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # project root (VLM/)
REPORTS_DIR = ROOT / "data" / "ecgen-radiology"
IMAGES_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "data"

# ---- text cleaning ---------------------------------------------------------
# "XXXX" is the anonymization placeholder the dataset uses to mask names,
# dates, and measurements. We drop it and tidy whitespace/punctuation.
_XXXX = re.compile(r"X{2,}")
_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:])")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = _XXXX.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _WS.sub(" ", text).strip()
    return text


def get_label(root: ET.Element, label: str) -> str:
    """Return the <AbstractText Label="..."> text for a given label."""
    for node in root.iter("AbstractText"):
        if node.attrib.get("Label", "").upper() == label.upper():
            return clean_text(node.text)
    return ""


def parse_report(xml_path: Path) -> dict | None:
    """Parse one XML file into {uid, findings, impression, image_ids}."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    findings = get_label(root, "FINDINGS")
    impression = get_label(root, "IMPRESSION")

    # Collect image ids that actually have a PNG on disk.
    image_ids = []
    for img in root.iter("parentImage"):
        img_id = img.attrib.get("id", "")
        if img_id and (IMAGES_DIR / f"{img_id}.png").exists():
            image_ids.append(img_id)

    return {
        "uid": xml_path.stem,            # e.g. "1000"
        "findings": findings,
        "impression": impression,
        "image_ids": image_ids,
    }


def build_records() -> list[dict]:
    """Build one record per (image, report) pair, keeping report uid for split."""
    xml_files = sorted(REPORTS_DIR.glob("*.xml"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    records: list[dict] = []
    n_reports = n_no_image = n_no_text = 0

    for xml_path in xml_files:
        rep = parse_report(xml_path)
        if rep is None:
            continue
        n_reports += 1

        # The "report" we train on = FINDINGS (fall back to IMPRESSION).
        report_text = rep["findings"] or rep["impression"]
        if not report_text:
            n_no_text += 1
            continue
        if not rep["image_ids"]:
            n_no_image += 1
            continue

        for img_id in rep["image_ids"]:
            records.append({
                "uid": rep["uid"],
                "image": f"data/images/{img_id}.png",
                "findings": rep["findings"],
                "impression": rep["impression"],
                "report": report_text,
            })

    print(f"Parsed reports        : {n_reports}")
    print(f"  dropped (no text)   : {n_no_text}")
    print(f"  dropped (no image)  : {n_no_image}")
    print(f"Usable (image,report) : {len(records)}")
    return records


def split_by_report(records: list[dict], seed: int = 42):
    """Split at report level (uid) so paired views stay together."""
    uids = sorted({r["uid"] for r in records})
    train_uids, temp_uids = train_test_split(uids, test_size=0.2, random_state=seed)
    val_uids, test_uids = train_test_split(temp_uids, test_size=0.5, random_state=seed)
    train_uids, val_uids, test_uids = set(train_uids), set(val_uids), set(test_uids)

    def subset(uid_set):
        return [r for r in records if r["uid"] in uid_set]

    return subset(train_uids), subset(val_uids), subset(test_uids)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def report_stats(name: str, rows: list[dict]) -> None:
    lengths = [len(r["report"].split()) for r in rows]
    avg = sum(lengths) / len(lengths) if lengths else 0
    print(f"  {name:5s}: {len(rows):5d} samples | avg report length {avg:5.1f} words")


def main() -> None:
    print("=== Building dataset ===")
    records = build_records()

    write_jsonl(OUT_DIR / "dataset.jsonl", records)

    train, val, test = split_by_report(records)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    print("\n=== Splits (report-level, no leakage) ===")
    report_stats("train", train)
    report_stats("val", val)
    report_stats("test", test)

    print("\n=== Example record ===")
    if records:
        ex = records[0]
        print(json.dumps({**ex, "report": ex["report"][:160] + "..."}, indent=2))

    print(f"\nWrote: dataset.jsonl, train.jsonl, val.jsonl, test.jsonl -> {OUT_DIR}")


if __name__ == "__main__":
    main()

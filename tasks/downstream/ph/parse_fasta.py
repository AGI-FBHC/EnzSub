#!/usr/bin/env python3
"""
Step 0: 解析 EpHod FASTA 文件为 CSV

处理 phopt_training/validation/testing.fasta → train/val/test.csv

用法:
  python parse_fasta.py \
      --input-dir /path/to/raw \
      --output-dir /path/to/processed \
      --splits train val test
"""

import argparse
import logging
from pathlib import Path

from utils import parse_fasta_to_dataframe, filter_ph_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# FASTA 文件名 → 输出 CSV stem 的映射
FASTA_STEM_MAP = {
    "phopt_training": "train",
    "phopt_validation": "val",
    "phopt_testing": "test",
}

def main():
    parser = argparse.ArgumentParser(description="Parse EpHod FASTA to CSV")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ph-min", type=float, default=2.0)
    parser.add_argument("--ph-max", type=float, default=12.0)
    parser.add_argument("--max-seq-length", type=int, default=4000)
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["train", "val", "test"],
                        help="Output splits to generate (train/val/test)")
    args = parser.parse_args()

    raw_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_outputs = set(args.splits)

    # 构建反向映射: 需要哪些 FASTA 文件
    all_fasta = list(raw_dir.glob("*.fasta")) + list(raw_dir.glob("*.fa"))
    fasta_files = []
    for f in all_fasta:
        out_stem = FASTA_STEM_MAP.get(f.stem, f.stem)
        if out_stem in allowed_outputs:
            fasta_files.append((f, out_stem))

    if not fasta_files:
        log.error(f"No matching FASTA files in {raw_dir}")
        log.error(f"  Available: {[f.stem for f in all_fasta]}")
        log.error(f"  Known mappings: {FASTA_STEM_MAP}")
        return False

    log.info(f"Processing {len(fasta_files)} file(s): "
             f"{[(f.name, out) for f, out in fasta_files]}")

    total_initial, total_final = 0, 0

    for fasta_file, out_stem in fasta_files:
        log.info(f"\n--- {fasta_file.name} → {out_stem}.csv ---")

        df = parse_fasta_to_dataframe(str(fasta_file))
        log.info(f"  Parsed {len(df)} records")

        df_filtered, stats = filter_ph_dataframe(
            df,
            ph_min=args.ph_min,
            ph_max=args.ph_max,
            max_seq_length=args.max_seq_length,
        )
        log.info(f"  Removed: {stats['removed_missing_pH']} missing pH, "
                 f"{stats['removed_pH_range']} out of range, "
                 f"{stats['removed_invalid_seq']} invalid seq")
        log.info(f"  Kept {stats['final']} records")

        output_csv = output_dir / f"{out_stem}.csv"
        df_filtered.to_csv(output_csv, index=False)
        log.info(f"  Saved: {output_csv}")

        total_initial += stats['initial']
        total_final += stats['final']

    log.info(f"\nTotal: {total_final}/{total_initial} "
             f"({total_final / total_initial * 100:.1f}%)")
    return True

if __name__ == "__main__":
    main()
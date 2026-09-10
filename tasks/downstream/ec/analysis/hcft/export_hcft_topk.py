

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a labeled Top-k embedding neighbourhood for one HCFT anchor.

Edit CONFIG below, then run:

    python export_hcft_topk.py

The script loads an existing merged ``*_all.pt`` embedding cache and an EC
label table, finds the nearest neighbours of one anchor, and immediately runs
MMseqs2 against those Top-k proteins. The final CSV therefore contains both
embedding cosine similarity and freshly computed sequence identity. By
default, the search gallery is restricted to the anchor's EC3 family.

The helper functions in this file are also imported by
``scan_hcft_topk_case_candidates.py``. Keep both files in the same directory.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd

# =============================================================================
# CONFIG: edit only this section
# =============================================================================
CONFIG = {
    # Embedding directory for the model state currently being inspected.
    # The directory must contain one or more merged *_all.pt files.
    "embedding_dir": (
        "artifacts/downstream/ec/embeddings/esm2_t33_650M/paper/esm2_t33_650M_base"
    ),

    # EC label table, for example split100.csv.
    "label_csv": "data/ec/split100.csv",

    # Anchor protein selected by scan_hcft_topk_case_candidates.py.
    "anchor": "Q55928",

    # Output for this model state. Use different filenames for Base and Final.
    "output": (
        "artifacts/downstream/ec/hcft/case_selection/Q55928_base_topk.csv"
    ),

    # Number of neighbours to export.
    "k": 20,

    # "family": search only inside the anchor's EC3 family.
    # "global": search across every labeled protein.
    "scope": "global",

    # Normally leave as None to infer EC3 from the anchor label.
    # Set a string such as "3.5.1" only when an explicit override is needed.
    "ec3_override": None,

    # Number of gallery embeddings processed at once.
    "embedding_chunk_size": 8192,

    # Sequence table containing both the anchor and gallery sequences.
    # It may be the same table as label_csv if that file contains Sequence.
    "sequence_csv": "data/ec/split100.csv",

    # MMseqs2 settings. The work directory stores FASTA files, databases,
    # raw alignments and the MMseqs log for this anchor/model-state run.
    "mmseqs_bin": "mmseqs",
    "mmseqs_threads": 32,
    "mmseqs_work_dir": (
        "artifacts/downstream/ec/hcft/case_selection/mmseqs_work/Q55928_base"
    ),

    # False is safer: an existing work directory raises an error.
    # Set True only when this exact run directory may be replaced.
    "force_recreate_work_dir": False,

    # Same-EC4 neighbours at or below this identity are marked as
    # HCFT-eligible low-identity positives. Identity is expected in percent.
    "positive_max_identity": 40.0,
}
# =============================================================================

# Keep this module importable on machines without the model runtime.
# PyTorch is loaded only when an embedding file is actually read.
torch = None
F = None

def require_torch() -> None:
    """Import PyTorch lazily with a clear environment error."""
    global torch, F
    if torch is not None:
        return
    try:
        import torch as torch_module
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to read the .pt embedding cache. Run this "
            "with the same Python/conda environment used for HCFT evaluation."
        ) from error
    torch = torch_module
    F = functional

def as_path(
    value: str | Path,
    name: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Convert one configured path and optionally require it to exist."""
    path = Path(value).expanduser()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"CONFIG[{name!r}] does not exist: {path}")
    return path

def validate_config() -> None:
    """Validate values that can be checked before loading large files."""
    if str(CONFIG["scope"]) not in {"family", "global"}:
        raise ValueError("CONFIG['scope'] must be 'family' or 'global'")

    for key in ("k", "embedding_chunk_size", "mmseqs_threads"):
        if int(CONFIG[key]) < 1:
            raise ValueError(f"CONFIG[{key!r}] must be positive")

    threshold = float(CONFIG["positive_max_identity"])
    if not 0.0 <= threshold <= 100.0:
        raise ValueError(
            "CONFIG['positive_max_identity'] must be between 0 and 100"
        )

    anchor = str(CONFIG["anchor"]).strip()
    if not anchor:
        raise ValueError("CONFIG['anchor'] cannot be empty")

def read_label_table(path: Path) -> dict[str, list[str]]:
    """Load EC labels from the TSV/CSV formats used by the EC task."""
    for separator in ("\t", ","):
        frame = pd.read_csv(
            path,
            sep=separator,
            dtype=str,
            na_filter=False,
        )
        if len(frame.columns) >= 2:
            break
    else:  # pragma: no cover - malformed input
        raise ValueError(f"Could not parse at least two columns from {path}")

    id_column = next(
        (
            column
            for column in (
                "Entry",
                "entry",
                "id",
                "ID",
                "seq_id",
                "uniprot_id",
            )
            if column in frame
        ),
        frame.columns[0],
    )
    ec_column = next(
        (
            column
            for column in (
                "EC number",
                "ec_number",
                "EC",
                "ec",
                "label",
            )
            if column in frame
        ),
        frame.columns[1],
    )

    labels: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        identifier = str(row[id_column]).strip()
        ec_values = [
            value.strip()
            for value in str(row[ec_column]).split(";")
            if value.strip()
        ]
        if identifier and ec_values:
            labels[identifier] = ec_values
    return labels

def load_embeddings(directory: Path) -> dict[str, "torch.Tensor"]:
    """Load all merged embedding mappings found in a directory."""
    require_torch()
    files = sorted(directory.glob("*_all.pt"))
    if not files:
        raise FileNotFoundError(
            f"No merged *_all.pt embedding file found in {directory}"
        )

    embeddings: dict[str, "torch.Tensor"] = {}
    for path in files:
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:  # PyTorch versions before weights_only existed
            payload = torch.load(path, map_location="cpu")

        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a dictionary payload")
        values = payload.get("embeddings", {})
        if not isinstance(values, dict):
            raise ValueError(f"{path} does not contain an embeddings mapping")

        embeddings.update(
            {
                str(identifier): vector.detach().float().cpu()
                for identifier, vector in values.items()
            }
        )
    return embeddings

def ec_at_level(labels: Iterable[str], level: int) -> set[str]:
    """Return complete EC prefixes at the requested hierarchy level."""
    return {
        ".".join(parts[:level])
        for label in labels
        if len(parts := label.split(".")) >= level
        and all(part and part != "-" for part in parts[:level])
    }

def read_sequence_table(path: Path) -> dict[str, str]:
    """Read common ID/sequence CSV or TSV layouts."""
    for separator in ("\t", ","):
        frame = pd.read_csv(
            path,
            sep=separator,
            dtype=str,
            na_filter=False,
        )
        if len(frame.columns) >= 2:
            break
    else:  # pragma: no cover - malformed input
        raise ValueError(
            f"Could not parse a two-column sequence table from {path}"
        )

    id_column = next(
        (
            column
            for column in (
                "Entry",
                "entry",
                "id",
                "ID",
                "seq_id",
                "uniprot_id",
            )
            if column in frame
        ),
        frame.columns[0],
    )
    sequence_column = next(
        (
            column
            for column in (
                "Sequence",
                "sequence",
                "seq",
                "protein_sequence",
            )
            if column in frame
        ),
        None,
    )
    if sequence_column is None:
        raise ValueError(
            f"No sequence column found in {path}. Expected one of "
            "Sequence, sequence, seq, protein_sequence; "
            f"found {list(frame.columns)}"
        )

    sequences: dict[str, str] = {}
    for identifier, sequence in zip(
        frame[id_column],
        frame[sequence_column],
    ):
        identifier = str(identifier).strip()
        cleaned = "".join(str(sequence).upper().split())
        if identifier and cleaned:
            sequences[identifier] = cleaned
    return sequences

def write_fasta(
    path: Path,
    records: list[tuple[str, str]],
) -> None:
    """Write compact FASTA records for the temporary MMseqs databases."""
    with path.open("w", encoding="utf-8") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n{sequence}\n")

def run_command(command: list[str], log_path: Path) -> None:
    """Run one MMseqs command and append stdout/stderr to a log."""
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"MMseqs command failed ({completed.returncode}); "
            f"inspect {log_path}"
        )

def prepare_work_directory(
    path: Path,
    force_recreate: bool,
) -> Path:
    """Create an isolated MMseqs directory with guarded replacement."""
    work_dir = path.expanduser().resolve()
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }
    if work_dir in protected:
        raise ValueError(
            f"Refusing to use broad protected path as MMseqs work directory: "
            f"{work_dir}"
        )

    if work_dir.exists():
        if not force_recreate:
            raise FileExistsError(
                f"MMseqs work directory already exists: {work_dir}. "
                "Set CONFIG['force_recreate_work_dir']=True to replace "
                "this exact directory."
            )
        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True)
    return work_dir

def recompute_mmseqs_identities(
    anchor: str,
    target_ids: list[str],
    sequences: dict[str, str],
    work_dir: Path,
    mmseqs_bin: str,
    threads: int,
) -> pd.DataFrame:
    """Run exhaustive anchor-to-Top-k MMseqs2 alignments."""
    if anchor not in sequences:
        raise KeyError(
            f"Anchor {anchor!r} has no non-empty sequence"
        )
    missing = [
        identifier
        for identifier in target_ids
        if identifier not in sequences
    ]
    if missing:
        raise KeyError(
            f"Missing sequences for {len(missing)} Top-k targets, "
            f"e.g. {missing[:5]}"
        )

    tmp_dir = work_dir / "mmseqs_tmp"
    tmp_dir.mkdir()
    query_fasta = work_dir / "query.fasta"
    target_fasta = work_dir / "targets.fasta"
    query_db = work_dir / "queryDB"
    target_db = work_dir / "targetDB"
    result_db = work_dir / "resultDB"
    alignment_path = work_dir / "anchor_vs_topk.m8"
    log_path = work_dir / "mmseqs.log"

    proxy_to_identifier = {
        f"t{index:04d}": identifier
        for index, identifier in enumerate(target_ids, start=1)
    }
    write_fasta(
        query_fasta,
        [("q0000", sequences[anchor])],
    )
    write_fasta(
        target_fasta,
        [
            (proxy, sequences[identifier])
            for proxy, identifier in proxy_to_identifier.items()
        ],
    )

    run_command(
        [
            mmseqs_bin,
            "createdb",
            str(query_fasta),
            str(query_db),
        ],
        log_path,
    )
    run_command(
        [
            mmseqs_bin,
            "createdb",
            str(target_fasta),
            str(target_db),
        ],
        log_path,
    )
    run_command(
        [
            mmseqs_bin,
            "search",
            str(query_db),
            str(target_db),
            str(result_db),
            str(tmp_dir),
            "--threads",
            str(threads),
            "--prefilter-mode",
            "2",
            "-e",
            "1e30",
            "--max-seqs",
            str(len(target_ids) + 10),
            "--min-seq-id",
            "0.0",
        ],
        log_path,
    )
    run_command(
        [
            mmseqs_bin,
            "convertalis",
            str(query_db),
            str(target_db),
            str(result_db),
            str(alignment_path),
            "--threads",
            str(threads),
            "--format-output",
            (
                "query,target,fident,alnlen,qstart,qend,qlen,"
                "tstart,tend,tlen,evalue,bits"
            ),
        ],
        log_path,
    )

    columns = [
        "query",
        "target",
        "fident",
        "alnlen",
        "qstart",
        "qend",
        "qlen",
        "tstart",
        "tend",
        "tlen",
        "evalue",
        "bits",
    ]
    try:
        hits = pd.read_csv(
            alignment_path,
            sep="\t",
            names=columns,
        )
    except pd.errors.EmptyDataError as error:
        raise RuntimeError(
            f"MMseqs produced no alignments; inspect {log_path}"
        ) from error

    numeric_columns = [
        "fident",
        "alnlen",
        "qstart",
        "qend",
        "qlen",
        "tstart",
        "tend",
        "tlen",
        "evalue",
        "bits",
    ]
    for column in numeric_columns:
        hits[column] = pd.to_numeric(
            hits[column],
            errors="coerce",
        )
    if hits[numeric_columns].isna().any().any():
        raise RuntimeError(
            f"MMseqs emitted non-numeric alignment fields; inspect {log_path}"
        )

    if hits["target"].duplicated().any():
        hits = (
            hits.sort_values(
                ["target", "evalue", "bits"],
                ascending=[True, True, False],
            )
            .drop_duplicates("target")
            .copy()
        )

    hits["neighbor_id"] = hits["target"].map(proxy_to_identifier)
    if hits["neighbor_id"].isna().any():
        raise RuntimeError(
            "MMseqs emitted an unexpected target proxy ID"
        )

    hits["sequence_identity"] = hits["fident"].astype(float)
    fraction_mask = hits["sequence_identity"].le(1.0)
    hits.loc[fraction_mask, "sequence_identity"] *= 100.0
    hits["alignment_length"] = hits["alnlen"].astype(int)
    hits["query_coverage"] = (
        (hits["qend"] - hits["qstart"] + 1) / hits["qlen"]
    )
    hits["target_coverage"] = (
        (hits["tend"] - hits["tstart"] + 1) / hits["tlen"]
    )

    returned = set(hits["neighbor_id"])
    missing_hits = sorted(set(target_ids) - returned)
    if missing_hits:
        raise RuntimeError(
            f"Exhaustive MMseqs search returned "
            f"{len(hits)}/{len(target_ids)} targets; "
            f"missing {missing_hits[:10]}. Inspect {log_path}."
        )

    return hits[
        [
            "neighbor_id",
            "sequence_identity",
            "alignment_length",
            "query_coverage",
            "target_coverage",
            "evalue",
            "bits",
        ]
    ].copy()

def resolve_anchor_family(
    labels: dict[str, list[str]],
    anchor: str,
) -> tuple[set[str], str]:
    """Resolve one complete EC4 set and one EC3 gallery family."""
    if anchor not in labels:
        raise KeyError(f"Anchor {anchor!r} has no EC label")

    anchor_ec4 = ec_at_level(labels[anchor], 4)
    if not anchor_ec4:
        raise ValueError(
            f"Anchor {anchor!r} has no complete four-level EC label"
        )

    override = CONFIG.get("ec3_override")
    if override is not None:
        anchor_ec3 = str(override).strip()
        if len(anchor_ec3.split(".")) != 3:
            raise ValueError(
                "CONFIG['ec3_override'] must contain exactly three EC levels"
            )
    else:
        ec3_values = sorted(ec_at_level(labels[anchor], 3))
        if not ec3_values:
            raise ValueError(
                f"Anchor {anchor!r} has no complete EC3 label"
            )
        if len(ec3_values) > 1:
            raise ValueError(
                f"Anchor {anchor!r} belongs to multiple EC3 families "
                f"{ec3_values}; set CONFIG['ec3_override'] explicitly"
            )
        anchor_ec3 = ec3_values[0]

    return anchor_ec4, anchor_ec3

def collect_topk(
    anchor: str,
    embeddings: dict[str, "torch.Tensor"],
    gallery_ids: list[str],
    k: int,
    chunk_size: int,
) -> list[tuple[float, str]]:
    """Find exact global Top-k while bounding temporary matrix memory."""
    query = F.normalize(
        embeddings[anchor].reshape(1, -1),
        dim=1,
    )
    candidates: list[tuple[float, str]] = []

    for start in range(0, len(gallery_ids), chunk_size):
        identifiers = gallery_ids[start : start + chunk_size]
        matrix = F.normalize(
            torch.stack(
                [embeddings[identifier] for identifier in identifiers]
            ),
            dim=1,
        )
        similarities = (query @ matrix.T).squeeze(0)
        take = min(k, len(identifiers))
        values, indices = torch.topk(similarities, k=take)
        candidates.extend(
            (float(value), identifiers[int(index)])
            for value, index in zip(values, indices)
        )

    candidates.sort(
        key=lambda item: (-item[0], item[1])
    )
    return candidates[:k]

def main() -> None:
    validate_config()
    require_torch()

    embedding_dir = as_path(
        CONFIG["embedding_dir"],
        "embedding_dir",
    )
    label_csv = as_path(CONFIG["label_csv"], "label_csv")
    output_path = as_path(
        CONFIG["output"],
        "output",
        must_exist=False,
    )
    sequence_csv = as_path(
        CONFIG["sequence_csv"],
        "sequence_csv",
    )
    mmseqs_work_dir = as_path(
        CONFIG["mmseqs_work_dir"],
        "mmseqs_work_dir",
        must_exist=False,
    )

    anchor = str(CONFIG["anchor"]).strip()
    k = int(CONFIG["k"])
    scope = str(CONFIG["scope"])
    embedding_chunk_size = int(CONFIG["embedding_chunk_size"])
    mmseqs_bin = str(CONFIG["mmseqs_bin"])
    mmseqs_threads = int(CONFIG["mmseqs_threads"])
    positive_max_identity = float(CONFIG["positive_max_identity"])
    force_recreate = bool(CONFIG["force_recreate_work_dir"])

    if shutil.which(mmseqs_bin) is None:
        raise FileNotFoundError(
            f"Could not find MMseqs executable: {mmseqs_bin}"
        )

    labels = read_label_table(label_csv)
    anchor_ec4, anchor_ec3 = resolve_anchor_family(labels, anchor)

    embeddings = load_embeddings(embedding_dir)
    if anchor not in embeddings:
        raise KeyError(
            f"Anchor {anchor!r} has no embedding in {embedding_dir}"
        )

    gallery_ids = [
        identifier
        for identifier, entry_labels in labels.items()
        if identifier != anchor
        and identifier in embeddings
        and (
            scope == "global"
            or anchor_ec3 in ec_at_level(entry_labels, 3)
        )
    ]
    if len(gallery_ids) < k:
        raise ValueError(
            f"Only {len(gallery_ids)} eligible gallery proteins for k={k}"
        )

    topk = collect_topk(
        anchor=anchor,
        embeddings=embeddings,
        gallery_ids=gallery_ids,
        k=k,
        chunk_size=embedding_chunk_size,
    )

    records: list[dict[str, object]] = []
    for rank, (similarity, identifier) in enumerate(topk, start=1):
        neighbour_labels = labels[identifier]
        neighbour_ec4 = ec_at_level(neighbour_labels, 4)
        same_ec4 = bool(anchor_ec4 & neighbour_ec4)

        records.append(
            {
                "rank": rank,
                "anchor": anchor,
                "anchor_ec4": ";".join(sorted(anchor_ec4)),
                "neighbor_id": identifier,
                "neighbor_ec4": ";".join(sorted(neighbour_ec4)),
                "cosine_similarity": similarity,
                "same_ec4_positive": same_ec4,
                "relation": (
                    "same EC4 positive"
                    if same_ec4
                    else "different EC4"
                ),
                "scope": scope,
                "ec3_family": anchor_ec3,
            }
        )

    topk_frame = pd.DataFrame(records)
    sequences = read_sequence_table(sequence_csv)
    target_ids = list(topk_frame["neighbor_id"].astype(str))

    # The work directory is only touched after all input tables, labels,
    # embeddings and required sequences have passed validation.
    missing_sequence_ids = [
        identifier
        for identifier in [anchor, *target_ids]
        if identifier not in sequences
    ]
    if missing_sequence_ids:
        raise KeyError(
            f"Missing sequences for {len(missing_sequence_ids)} proteins, "
            f"e.g. {missing_sequence_ids[:5]}"
        )
    work_dir = prepare_work_directory(
        mmseqs_work_dir,
        force_recreate,
    )
    identity_frame = recompute_mmseqs_identities(
        anchor=anchor,
        target_ids=target_ids,
        sequences=sequences,
        work_dir=work_dir,
        mmseqs_bin=mmseqs_bin,
        threads=mmseqs_threads,
    )

    output = topk_frame.merge(
        identity_frame,
        on="neighbor_id",
        how="left",
        validate="one_to_one",
    )
    if output["sequence_identity"].isna().any():
        raise RuntimeError(
            "Unexpected missing MMseqs identities after Top-k merge"
        )
    output["identity_available"] = True
    output["hcft_positive_eligible"] = (
        output["same_ec4_positive"]
        & output["sequence_identity"].le(positive_max_identity)
    )

    ordered_columns = [
        "rank",
        "anchor",
        "anchor_ec4",
        "neighbor_id",
        "neighbor_ec4",
        "cosine_similarity",
        "sequence_identity",
        "alignment_length",
        "query_coverage",
        "target_coverage",
        "evalue",
        "bits",
        "identity_available",
        "same_ec4_positive",
        "hcft_positive_eligible",
        "relation",
        "scope",
        "ec3_family",
    ]
    output = output[ordered_columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    first_positive = output.loc[
        output["same_ec4_positive"],
        "rank",
    ]
    rank_text = (
        str(int(first_positive.iloc[0]))
        if not first_positive.empty
        else f">{k}"
    )
    print(
        f"anchor={anchor} | "
        f"EC4={';'.join(sorted(anchor_ec4))} | "
        f"scope={scope} | "
        f"gallery={len(gallery_ids):,}"
    )
    print(f"first_same_EC4_positive_rank={rank_text}")
    print(f"mmseqs_identity_coverage={len(output)}/{len(output)}")
    print(f"wrote {output_path}")
    print(f"MMseqs work/log directory: {work_dir}")
    print(output.to_string(index=False))

if __name__ == "__main__":
    main()
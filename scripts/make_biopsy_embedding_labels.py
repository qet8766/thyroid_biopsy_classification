#!/usr/bin/env python3
"""Build slide-level label files for biopsy UNI embeddings.

The script intentionally uses only the Python standard library so it can run in
the current inference environment without installing Excel dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

SINGLE_SLIDE_TOKEN = "\ub2e8"
LABEL_ORDER = [
    "I",
    "II",
    "III",
    "IIIa",
    "IIIb",
    "IIIc",
    "IIId",
    "IIIe",
    "IV",
    "IVa",
    "IVb",
    "IVc",
    "IVd",
    "V",
    "VI",
    "LYMPHOMA",
    "_",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
MAJOR_TO_ID = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}


def spreadsheet_column_index(cell_ref: str) -> int | None:
    match = re.match(r"([A-Z]+)", cell_ref or "")
    if not match:
        return None
    index = 0
    for char in match.group(1):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def read_xlsx_first_sheet(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", XLSX_NS):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS))
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}

        first_sheet = workbook.find("a:sheets/a:sheet", XLSX_NS)
        if first_sheet is None:
            raise ValueError(f"{path} has no worksheet")
        rel_id = first_sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = rel_targets[rel_id].lstrip("/")
        sheet_path = target if target.startswith("xl/") else posixpath.normpath(f"xl/{target}")

        sheet = ET.fromstring(archive.read(sheet_path))
        raw_rows: list[list[str]] = []
        for row in sheet.findall("a:sheetData/a:row", XLSX_NS):
            values: list[str] = []
            for cell in row.findall("a:c", XLSX_NS):
                index = spreadsheet_column_index(cell.attrib.get("r", ""))
                if index is None:
                    index = len(values)
                while len(values) <= index:
                    values.append("")

                value_node = cell.find("a:v", XLSX_NS)
                inline_node = cell.find("a:is", XLSX_NS)
                cell_type = cell.attrib.get("t")
                if cell_type == "s" and value_node is not None:
                    value = shared_strings[int(value_node.text or "0")]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(
                        text.text or "" for text in inline_node.findall(".//a:t", XLSX_NS)
                    )
                elif value_node is not None:
                    value = value_node.text or ""
                else:
                    value = ""
                values[index] = value.strip() if isinstance(value, str) else str(value)
            raw_rows.append(values)

    if not raw_rows:
        raise ValueError(f"{path} has no rows")

    headers = [item.strip() for item in raw_rows[0]]
    records: list[dict[str, str]] = []
    for row_number, values in enumerate(raw_rows[1:], start=2):
        padded = values + [""] * (len(headers) - len(values))
        record = {header: padded[index].strip() for index, header in enumerate(headers)}
        record["label_row"] = str(row_number)
        records.append(record)
    return records


def normalize_case_id(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).upper()


def normalize_label_key(value: str) -> str:
    key = (value or "").strip()
    if key == SINGLE_SLIDE_TOKEN:
        return ""
    if key.startswith("#"):
        key = key[1:]
    return key.replace(" ", "")


def expand_key(key: str) -> list[str]:
    key = (key or "").strip()
    if not key:
        return [""]
    if "," in key:
        return [item.strip() for item in key.split(",") if item.strip()]
    if "-" in key:
        start, end = key.split("-", 1)
        if start.isdigit() and end.isdigit():
            return [str(value) for value in range(int(start), int(end) + 1)]
    return [key]


def normalize_category(value: str) -> str:
    category = (value or "").strip()
    if not category:
        return ""
    if category == "_":
        return "_"
    if category.upper() == "LYMPHOMA":
        return "LYMPHOMA"

    match = re.match(r"^(VI|IV|V|III|II|I)([A-Za-z]?)$", category, flags=re.IGNORECASE)
    if not match:
        return category
    return f"{match.group(1).upper()}{match.group(2).lower()}"


def category_major(category: str) -> str:
    if category in {"", "_", "LYMPHOMA"}:
        return category
    for major in ("VI", "IV", "V", "III", "II", "I"):
        if category.startswith(major):
            return major
    return ""


def extract_case_id(rel_path: str) -> str:
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return ""
    return normalize_case_id(parts[0])


def extract_slide_key(rel_path: str) -> str:
    stem = PurePosixPath(rel_path).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def extract_scan_id(rel_path: str) -> str:
    return PurePosixPath(rel_path).stem.split("_", 1)[0]


def unique_join(values: list[str]) -> str:
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            kept.append(value)
    return "|".join(kept)


def candidate_summary(candidates: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for item in candidates:
        parts.append(
            "row {row}: slide={slide}, lesion={lesion}, side={side}, label={label}".format(
                row=item.get("label_row", ""),
                slide=item.get("Slide number", ""),
                lesion=item.get("lesion", ""),
                side=item.get("Side", ""),
                label=item.get("category_norm", ""),
            )
        )
    return "; ".join(parts)


def label_fields(
    status: str,
    candidates: list[dict[str, str]],
    note: str = "",
) -> dict[str, str]:
    if not candidates:
        return {
            "label_status": status,
            "label": "",
            "label_id": "",
            "label_major": "",
            "label_major_id": "",
            "label_raw": "",
            "lesion": "",
            "side": "",
            "ground_truth_slide_number": "",
            "label_row": "",
            "candidate_count": "0",
            "candidate_summary": "",
            "exception_note": note,
        }

    categories = [item.get("category_norm", "") for item in candidates]
    category = categories[0] if len(set(categories)) == 1 else ""
    major = category_major(category)
    return {
        "label_status": status,
        "label": category,
        "label_id": str(LABEL_TO_ID[category]) if category in LABEL_TO_ID else "",
        "label_major": major,
        "label_major_id": str(MAJOR_TO_ID[major]) if major in MAJOR_TO_ID else "",
        "label_raw": unique_join([item.get("H&E category", "") for item in candidates]),
        "lesion": unique_join([item.get("lesion", "") for item in candidates]),
        "side": unique_join([item.get("Side", "") for item in candidates]),
        "ground_truth_slide_number": unique_join(
            [item.get("Slide number", "") for item in candidates]
        ),
        "label_row": unique_join([item.get("label_row", "") for item in candidates]),
        "candidate_count": str(len(candidates)),
        "candidate_summary": candidate_summary(candidates),
        "exception_note": note,
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"rel_path", "source_file", "output_path", "status"}
    missing = required.difference(rows[0].keys() if rows else set())
    if missing:
        raise ValueError(f"{path} is missing required manifest columns: {sorted(missing)}")
    return rows


def load_exceptions(path: Path | None) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    if path is None or not path.exists():
        return {}, {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    by_rel_path: dict[str, dict] = {}
    by_case_slide: dict[tuple[str, str], dict] = {}
    for override in data.get("overrides", []):
        if not isinstance(override, dict):
            continue
        if not override_has_action(override):
            continue
        rel_path = (override.get("rel_path") or "").strip()
        if rel_path:
            by_rel_path[PurePosixPath(rel_path).as_posix()] = override
            continue
        case_id = normalize_case_id(override.get("case_id", ""))
        slide_key = (override.get("slide_key") or "").strip()
        if case_id:
            by_case_slide[(case_id, slide_key)] = override
    return by_rel_path, by_case_slide


def override_has_action(override: dict) -> bool:
    return bool(
        override.get("exclude")
        or override.get("label_row")
        or (override.get("label") or "").strip()
    )


def apply_override(
    base: dict[str, str],
    override: dict,
    labels_by_row: dict[str, dict[str, str]],
) -> dict[str, str]:
    result = dict(base)
    note = (override.get("note") or "").strip()

    if override.get("exclude"):
        result.update(label_fields("excluded", [], note=note))
        return result

    label_row = str(override.get("label_row") or "").strip()
    if label_row:
        if label_row not in labels_by_row:
            raise ValueError(f"exception references unknown label_row={label_row}")
        result.update(label_fields("manual_override", [labels_by_row[label_row]], note=note))
        return result

    label = normalize_category(override.get("label", ""))
    if label:
        manual = {
            "label_row": "",
            "Slide number": override.get("ground_truth_slide_number", ""),
            "lesion": override.get("lesion", ""),
            "Side": override.get("side", ""),
            "H&E category": override.get("label_raw", override.get("label", "")),
            "category_norm": label,
        }
        result.update(label_fields("manual_override", [manual], note=note))
        return result

    return result


def build_label_rows(
    labels: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    manifest_path: Path,
    exceptions_path: Path | None,
) -> tuple[list[dict[str, str]], Counter]:
    labels_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    labels_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    labels_by_row: dict[str, dict[str, str]] = {}

    for raw in labels:
        record = dict(raw)
        case_id = normalize_case_id(record.get("pathology_id", ""))
        slide_key = normalize_label_key(record.get("Slide number", ""))
        record["case_id"] = case_id
        record["slide_key_norm"] = slide_key
        record["category_norm"] = normalize_category(record.get("H&E category", ""))
        labels_by_case[case_id].append(record)
        labels_by_row[record["label_row"]] = record
        for expanded_key in expand_key(slide_key):
            labels_by_key[(case_id, expanded_key)].append(record)

    manifest_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        manifest_by_case[extract_case_id(row["rel_path"])].append(row)

    exceptions_by_rel, exceptions_by_case_slide = load_exceptions(exceptions_path)

    output_rows: list[dict[str, str]] = []
    counts: Counter = Counter()
    for manifest in manifest_rows:
        rel_path = PurePosixPath(manifest["rel_path"]).as_posix()
        case_id = extract_case_id(rel_path)
        slide_key = extract_slide_key(rel_path)

        candidates_by_row: dict[str, dict[str, str]] = {}
        for expanded_key in expand_key(slide_key):
            for candidate in labels_by_key.get((case_id, expanded_key), []):
                candidates_by_row[candidate["label_row"]] = candidate
        candidates = list(candidates_by_row.values())

        if candidates:
            categories = {item["category_norm"] for item in candidates}
            if len(categories) == 1:
                auto_status = "matched" if len(candidates) == 1 else "multi_row_same_category"
            else:
                auto_status = "ambiguous"
        elif len(labels_by_case.get(case_id, [])) == 1 and len(manifest_by_case.get(case_id, [])) == 1:
            candidates = labels_by_case[case_id]
            auto_status = "single_label_case"
        else:
            auto_status = "unmatched"

        embedding_name = Path(manifest["output_path"]).name
        sibling_embedding = manifest_path.parent / embedding_name
        embedding_path = str(sibling_embedding if sibling_embedding.exists() else manifest["output_path"])

        output = {
            "embedding_path": embedding_path,
            "embedding_file": embedding_name,
            "manifest_output_path": manifest["output_path"],
            "rel_path": rel_path,
            "case_id": case_id,
            "scan_id": extract_scan_id(rel_path),
            "slide_key": slide_key,
            "source_file": manifest["source_file"],
            "manifest_status": manifest["status"],
            "selected_tiles": manifest.get("selected_tiles", ""),
            "embedded_tiles": manifest.get("embedded_tiles", ""),
            "total_tiles": manifest.get("total_tiles", ""),
            "source_mpp": manifest.get("source_mpp", ""),
            "target_mpp": manifest.get("target_mpp", ""),
        }
        output.update(label_fields(auto_status, candidates))

        override = exceptions_by_rel.get(rel_path) or exceptions_by_case_slide.get((case_id, slide_key))
        if override:
            output = apply_override(output, override, labels_by_row)

        counts[output["label_status"]] += 1
        output_rows.append(output)

    return output_rows, counts


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_exception_template(path: Path, issue_rows: list[dict[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    overrides: list[dict[str, object]] = []
    for row in issue_rows:
        if row["label_status"] not in {"ambiguous", "unmatched"}:
            continue
        overrides.append(
            {
                "rel_path": row["rel_path"],
                "case_id": row["case_id"],
                "slide_key": row["slide_key"],
                "label_row": None,
                "label": None,
                "exclude": False,
                "note": row["candidate_summary"] or "No matching label row found.",
            }
        )
    data = {
        "instructions": [
            "Fill exactly one of label_row, label, or exclude=true for each unresolved rel_path.",
            "Use label_row when one of the candidate spreadsheet rows is correct.",
            "Use label for a manual normalized label such as IIIb, IVa, V, or VI.",
        ],
        "overrides": overrides,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("/mnt/fastpath_d/thyroid/excel files/biopsy label.xlsx"),
        help="Ground-truth biopsy label workbook.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/manifest.csv"),
        help="Embedding manifest generated by fastpath_uni_infer.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_labels.csv"
        ),
        help="Destination CSV containing only labeled, non-excluded embedding rows.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="Optional destination CSV with every manifest row, including excluded rows.",
    )
    parser.add_argument(
        "--issues-output",
        type=Path,
        default=Path(
            "/mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_embedding_label_issues.csv"
        ),
        help="Destination CSV for rows that need review or an exception.",
    )
    parser.add_argument(
        "--exceptions",
        type=Path,
        default=Path(
            "/mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_label_exceptions.json"
        ),
        help="Optional JSON overrides for ambiguous or unmatched rows.",
    )
    parser.add_argument(
        "--exception-template",
        type=Path,
        default=Path(
            "/mnt/fastpath_d/thyroid/_pipeline/uni_thyroid_patch256/biopsy_label_exceptions.json"
        ),
        help="Template JSON to write for unresolved rows.",
    )
    parser.add_argument(
        "--overwrite-exception-template",
        action="store_true",
        help="Overwrite the exception template even if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = read_xlsx_first_sheet(args.workbook)
    required_columns = {"lesion", "pathology_id", "Slide number", "Side", "H&E category"}
    missing_columns = required_columns.difference(labels[0].keys() if labels else set())
    if missing_columns:
        raise ValueError(f"{args.workbook} is missing required columns: {sorted(missing_columns)}")

    manifest_rows = load_manifest(args.manifest)
    output_rows, counts = build_label_rows(labels, manifest_rows, args.manifest, args.exceptions)

    label_fields_order = [
        "embedding_path",
        "embedding_file",
        "manifest_output_path",
        "rel_path",
        "case_id",
        "scan_id",
        "slide_key",
        "source_file",
        "manifest_status",
        "selected_tiles",
        "embedded_tiles",
        "total_tiles",
        "source_mpp",
        "target_mpp",
        "label_status",
        "label",
        "label_id",
        "label_major",
        "label_major_id",
        "label_raw",
        "lesion",
        "side",
        "ground_truth_slide_number",
        "label_row",
        "candidate_count",
        "candidate_summary",
        "exception_note",
    ]
    usable_rows = [
        row
        for row in output_rows
        if row["label"] and row["label_status"] not in {"excluded", "ambiguous", "unmatched"}
    ]
    write_csv(args.output, usable_rows, label_fields_order)
    if args.audit_output is not None:
        write_csv(args.audit_output, output_rows, label_fields_order)

    issue_statuses = {"single_label_case", "multi_row_same_category", "ambiguous", "unmatched"}
    issue_rows = [row for row in output_rows if row["label_status"] in issue_statuses]
    write_csv(args.issues_output, issue_rows, label_fields_order)
    write_exception_template(args.exception_template, issue_rows, args.overwrite_exception_template)

    print(f"wrote {args.output}")
    if args.audit_output is not None:
        print(f"wrote {args.audit_output}")
    print(f"wrote {args.issues_output}")
    if args.exception_template:
        print(f"exception template: {args.exception_template}")
    print("label_status counts:")
    for status, count in counts.most_common():
        print(f"  {status}: {count}")
    unresolved = counts["ambiguous"] + counts["unmatched"]
    print(f"unresolved rows requiring manual exception: {unresolved}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise

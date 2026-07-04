"""Export all 90 shared realization matrices as Excel and CSV tables."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np

try:
    import xlsxwriter
except ModuleNotFoundError:
    xlsxwriter = None


ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = ROOT / "results" / "processed" / "req4_shared_realizations"
OUTPUT_DIR = ROOT / "results" / "exports" / "shared_realization_tables"
CSV_DIR = OUTPUT_DIR / "csv"
XLSX_PATH = OUTPUT_DIR / "req4_shared_realizations_90_matrices.xlsx"
INDEX_PATH = OUTPUT_DIR / "matrix_index.csv"
ZIP_PATH = OUTPUT_DIR / "req4_shared_realizations_90_csv.zip"

FREQ_CODES = {
    "low_frequency": "LF",
    "medium_frequency": "MF",
    "high_frequency": "HF",
}
SKL_CODES = {
    "low_skl": "S20",
    "medium_skl": "S200",
    "high_skl": "S2000",
}


def collect_records():
    records = []
    for setting_dir in sorted(path for path in MATRIX_DIR.iterdir() if path.is_dir()):
        config = json.loads((setting_dir / "config.json").read_text(encoding="utf-8"))
        for realization, matrix_path in enumerate(sorted(setting_dir.glob("realization_*.npy"))):
            sheet_name = (
                f"{FREQ_CODES[config['frequency_label']]}_"
                f"{SKL_CODES[config['skl_label']]}_R{realization:02d}"
            )
            records.append(
                {
                    "sheet_name": sheet_name,
                    "setting_id": config["setting_id"],
                    "frequency_label": config["frequency_label"],
                    "change_frequency": config["change_frequency"],
                    "skl_label": config["skl_label"],
                    "target_skl": config["target_skl"],
                    "realization": realization,
                    "sample_seed": config["sample_seeds"][realization],
                    "schedule_seed": config["schedule_seed"],
                    "constellation_seed": config["constellation_seed"],
                    "rows": config["horizon"],
                    "campaigns": config["number_of_campaigns"],
                    "matrix_file": str(matrix_path.relative_to(ROOT)),
                    "matrix_path": matrix_path,
                }
            )
    if len(records) != 90:
        raise ValueError(f"Expected 90 matrices, found {len(records)}.")
    return records


def export_index(records):
    fields = [key for key in records[0] if key != "matrix_path"]
    with INDEX_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in records)


def export_csv_files(records):
    for record in records:
        matrix = np.load(record["matrix_path"])
        expected_shape = (record["rows"], record["campaigns"])
        if matrix.shape != expected_shape:
            raise ValueError(f"Unexpected shape {matrix.shape} in {record['matrix_path']}")
        header = ["round"] + [f"campaign_{campaign}" for campaign in range(record["campaigns"])]
        output = CSV_DIR / f"{record['sheet_name']}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for round_index, values in enumerate(matrix):
                writer.writerow([round_index, *[format(float(value), ".17g") for value in values]])


def export_excel(records):
    if xlsxwriter is None:
        if XLSX_PATH.exists():
            XLSX_PATH.unlink()
        print("xlsxwriter is not installed; skipped Excel export.")
        return

    workbook = xlsxwriter.Workbook(XLSX_PATH, {"constant_memory": True})
    header_format = workbook.add_format(
        {"bold": True, "font_color": "#FFFFFF", "bg_color": "#2F6DF6", "border": 1}
    )
    integer_format = workbook.add_format({"num_format": "0"})
    value_format = workbook.add_format({"num_format": "0.000000000000000"})

    index = workbook.add_worksheet("INDEX")
    index_fields = [key for key in records[0] if key != "matrix_path"]
    index.write_row(0, 0, index_fields, header_format)
    for row_index, record in enumerate(records, start=1):
        index.write_row(row_index, 0, [record[field] for field in index_fields])
    index.freeze_panes(1, 0)
    index.autofilter(0, 0, len(records), len(index_fields) - 1)
    index.set_column(0, 0, 16)
    index.set_column(1, 5, 24)
    index.set_column(6, 11, 16)
    index.set_column(12, 12, 72)

    for record in records:
        matrix = np.load(record["matrix_path"])
        headers = ["round"] + [f"campaign_{campaign}" for campaign in range(record["campaigns"])]
        worksheet = workbook.add_worksheet(record["sheet_name"])
        worksheet.write_row(0, 0, headers, header_format)
        for round_index, values in enumerate(matrix, start=1):
            worksheet.write_number(round_index, 0, round_index - 1, integer_format)
            for campaign, value in enumerate(values, start=1):
                worksheet.write_number(round_index, campaign, float(value), value_format)
        worksheet.freeze_panes(1, 1)
        worksheet.autofilter(0, 0, record["rows"], record["campaigns"])
        worksheet.set_column(0, 0, 10)
        worksheet.set_column(1, record["campaigns"], 19)
    workbook.close()


def export_zip():
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(INDEX_PATH, INDEX_PATH.name)
        for csv_path in sorted(CSV_DIR.glob("*.csv")):
            archive.write(csv_path, f"csv/{csv_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for old_csv in CSV_DIR.glob("*.csv"):
        old_csv.unlink()

    records = collect_records()
    export_index(records)
    export_csv_files(records)
    export_excel(records)
    export_zip()
    print(f"Exported {len(records)} matrices to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

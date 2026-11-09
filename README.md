# CS 236 Project – Phase 1: EDA and Dataset Processing

Exploratory data analysis and processing of **NYC 311** and **LA MyLA311** service requests (800,000 records each) with PySpark.

## Requirements

| Component | Version |
|---|---|
| Python | 3.12 |
| PySpark | 3.5.1 |
| Java | 11 (required by Spark) |
| matplotlib, pandas | for the EDA plots |

```bash
pip install pyspark==3.5.1 matplotlib pandas
java -version        # must report version 11
```

## Folder structure

```
Project/
├── CS236-Fall26-Project-Dataset/     # raw input (not modified)
│   ├── nyc_311_800k.csv
│   └── la_311_800k.csv
├── phase1/
│   ├── eda.py                        # 1.2 Exploratory Data Analysis
│   ├── process.py                    # 1.3 Dataset Processing
│   ├── README.md
│   └── output/
│       ├── eda_output.txt            # EDA console log
│       ├── process_output.txt        # processing console log
│       └── eda/
│           ├── schema_comparison.csv
│           └── plots/                # 9 distribution plots (3_*.png)
└── processed/
    ├── nyc_clean.parquet
    ├── la_clean.parquet
    ├── unified.parquet
    └── csv/
        ├── nyc_clean.csv
        ├── la_clean.csv
        └── unified.csv
```

## How to run

Run all commands from the **Project** folder, because both scripts find the data relative to it.

```bash
cd "/path/to/Project"
```

### 1. Exploratory Data Analysis (`eda.py`)

```bash
python -u phase1/eda.py | tee phase1/output/eda_output.txt
```

The script runs all four parts by default. To run some parts only, pass their numbers:

```bash
python phase1/eda.py 1        # Part 1: load and scan (schema, dimensions, types, date ranges, samples)
python phase1/eda.py 2        # Part 2: data quality (missing values, duplicates, invalid values)
python phase1/eda.py 3        # Part 3: distributions (writes 9 plots)
python phase1/eda.py 4        # Part 4: schema comparison (writes schema_comparison.csv)
python phase1/eda.py 3 4      # several parts
```

EDA reads only the raw CSVs and does not change them.

### 2. Dataset Processing (`process.py`)

```bash
python -u phase1/process.py | tee phase1/output/process_output.txt
```

Steps, one function per step:
1. Clean both datasets: trim text, turn placeholders into NULL, validate ZIP codes, merge case variants, and fill or label empty cells by rule.
2. Adjust the schema: parse timestamps and cast numeric and boolean columns.
3. Standardize values: `status_std`, `channel_std`.
4. Create the required attributes: `source_city`, `request_id`, `resolution_hours`, `resolution_valid`.
5. Save the cleaned datasets to `processed/nyc_clean.*` and `processed/la_clean.*`.
6. Merge both into a 23-column unified schema, saved to `processed/unified.*`.
7. Validate the output with 22 checks; the run ends with `22/22 checks passed`.

To write Parquet only and skip the CSV files (faster):

```bash
python -u phase1/process.py --skip-csv
```

Each run rebuilds all three datasets from the raw files and overwrites the previous outputs.

## Outputs

| File | Rows | Columns | Description |
|---|---|---|---|
| `processed/csv/nyc_clean.csv` | 800,000 | 46 | All 44 NYC columns + 2 derived |
| `processed/csv/la_clean.csv` | 800,000 | 36 | All 34 LA columns + 2 derived |
| `processed/csv/unified.csv` | 1,600,000 | 23 | Shared schema for both cities |

The Parquet versions keep the column types (timestamps, integers, booleans) and are the input for Phase 2. The CSV versions are for viewing.

## Notes

- **No rows or original columns are removed.** Missing values stay NULL, or become "Not Applicable" where the attribute does not apply to that request type. Missing values are never replaced with 0.
- Timestamps are local wall-clock times; the Spark session time zone is set to UTC so they are not shifted.
- Both scripts set `PYSPARK_PYTHON` to the running interpreter, which avoids Python version mismatches between the Spark driver and its workers.
- If a run looks stuck when piped to `tee`, use `python -u` (unbuffered output), as in the commands above.

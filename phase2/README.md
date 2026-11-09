# CS 236 Project – Phase 2: Spark Analysis and Database Population

Phase 2 uses the outputs of Phase 1 (`processed/*.parquet`). Run `phase1/process.py` first.

## Files

| File | Section | Purpose |
|---|---|---|
| `analysis.py` | 2.1 | Spark analyses on `processed/unified.parquet` (2 required + 2 bonus) |
| `schema.sql` | 2.2 | PostgreSQL table design: types, primary keys, NOT NULL, CHECK constraints, indexes |
| `load_db.py` | 2.2 | Creates the tables from `schema.sql`, loads all three datasets with Spark JDBC, verifies them |

## 1. Spark analysis

```bash
cd "/path/to/Project"
mkdir -p phase2/output
python -u phase2/analysis.py | tee phase2/output/analysis_output.txt
```

| Analysis | Output tables (`phase2/output/analysis/`) | Plot (`phase2/output/plots/`) |
|---|---|---|
| A1 Weekly request volume | `a1_weekly_volume_by_period.csv`, `a1_weekly_volume_by_month.csv`, `a1_weekly_volume_highest_lowest.csv` | `a1_weekly_volume.png` |
| A2 Service-request hotspots | `a2_hotspots_top_zip_request_types.csv` | `a2_hotspots.png` |
| B1 Resolution performance (bonus) | `b1_resolution_by_city.csv`, `b1_slowest_request_types.csv` | `b1_slowest_request_types.png` |
| B2 Status and channels (bonus) | `b2_status_std_by_city.csv`, `b2_channel_std_by_city.csv` | `b2_status_and_channel.png` |

Each analysis prints its result tables and a "Findings" summary.

## 2. Database population

### Start PostgreSQL with Docker

```bash
docker run -d \
  --name cs236-postgres \
  -e POSTGRES_USER=cs236 \
  -e POSTGRES_PASSWORD=cs236 \
  -e POSTGRES_DB=cs236 \
  -p 5432:5432 \
  -v cs236-pgdata:/var/lib/postgresql/data \
  postgres:16

docker ps                      # cs236-postgres should be "Up"
```

The `-v` volume keeps the data when the container is stopped (`docker stop cs236-postgres`) and started again (`docker start cs236-postgres`).

### Load and verify

```bash
python -u phase2/load_db.py | tee phase2/output/load_db_output.txt
```

or, as in the project instructions:

```bash
PYSPARK_PYTHON=$(which python) spark-submit \
  --packages org.postgresql:postgresql:42.7.3 \
  phase2/load_db.py
```

The first run downloads the PostgreSQL JDBC driver (42.7.3). The script:
1. runs `schema.sql` (drops and recreates the three tables, so every run starts clean),
2. checks that the Parquet columns match the tables and that NOT NULL columns have no NULLs,
3. loads `nyc_requests` (800,000 rows), `la_requests` (800,000) and `unified_requests` (1,600,000),
4. verifies each table: row count, column types, primary key, non-NULL count of every column, timestamp ranges, numeric totals, and a 20-row sample compared column by column. It ends with `N of N checks passed`.

Connection settings default to the docker command above. Change them with environment variables if needed: `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD`.

### Look at the data

```bash
docker exec -it cs236-postgres psql -U cs236 -d cs236
```

```sql
\dt                                   -- list tables
\d unified_requests                   -- columns, types, constraints, indexes
SELECT source_city, COUNT(*) FROM unified_requests GROUP BY source_city;
```

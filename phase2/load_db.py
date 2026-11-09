"""
load_db.py - CS236 Phase 2, section 2.2: load the three processed datasets into PostgreSQL
with PySpark (JDBC) and verify the result.

Run from the Project folder, with the PostgreSQL container running:
    python -u phase2/load_db.py | tee phase2/output/load_db_output.txt
or, as in the project instructions:
    spark-submit --packages org.postgresql:postgresql:42.7.3 phase2/load_db.py

Inputs   processed/nyc_clean.parquet, processed/la_clean.parquet, processed/unified.parquet
Tables   nyc_requests, la_requests, unified_requests   (design: phase2/schema.sql)

Steps
    1. create   run phase2/schema.sql (drop + create tables, keys, constraints, indexes)
    2. precheck every Parquet column exists in its table, and NOT NULL columns have no NULLs
    3. load     DataFrame.write.jdbc(mode="append") into the empty tables
    4. verify   row counts, column types, non-NULL counts per column, timestamp ranges,
                numeric totals and a row-by-row comparison of a random sample

Connection settings come from environment variables (defaults match the docker command
in phase2/README.md): PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD.
"""
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE, "processed")
SCHEMA_SQL = os.path.join(BASE, "phase2", "schema.sql")

PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DB = os.environ.get("PG_DB", "cs236")
PG_USER = os.environ.get("PG_USER", "cs236")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "cs236")
# reWriteBatchedInserts: the driver sends multi-row INSERTs, which loads much faster.
JDBC_URL = f"jdbc:postgresql://{PG_HOST}:{PG_PORT}/{PG_DB}?reWriteBatchedInserts=true"
JDBC_PROPS = {"user": PG_USER, "password": PG_PASSWORD, "driver": "org.postgresql.Driver",
              "batchsize": "10000"}

# dataset name -> (table, primary-key column)
DATASETS = {
    "nyc_clean": ("nyc_requests", "unique_key"),
    "la_clean": ("la_requests", "srnumber"),
    "unified": ("unified_requests", "request_id"),
}

# Spark type -> accepted PostgreSQL types (information_schema.columns.data_type)
TYPE_MATCH = {
    "string": {"text", "character varying"},
    "timestamp": {"timestamp without time zone"},
    "bigint": {"bigint"},
    "int": {"integer"},
    "double": {"double precision"},
    "boolean": {"boolean"},
}

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (SparkSession.builder
         .appName("CS236-LoadDB")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")  # JDBC driver
         .config("spark.sql.session.timeZone", "UTC")      # same wall-clock times as Phase 1
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# The JDBC driver turns timestamps into text using the JVM's default time zone. Setting it to
# UTC (the Spark session time zone) stores exactly the wall-clock times of the Parquet files,
# e.g. 2024-01-01 00:00:06 stays 2024-01-01 00:00:06 in PostgreSQL.
_jvm = spark.sparkContext._jvm
_jvm.java.util.TimeZone.setDefault(_jvm.java.util.TimeZone.getTimeZone("UTC"))

results = []


def header(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def query(sql):
    """Run a SELECT in PostgreSQL and return the result as a Spark DataFrame."""
    return (spark.read.format("jdbc").option("url", JDBC_URL).option("query", sql)
            .option("user", PG_USER).option("password", PG_PASSWORD)
            .option("driver", "org.postgresql.Driver").load())


def execute_sql_file(path):
    """Execute every statement of a .sql file through the PostgreSQL JDBC driver.
    (Spark's JDBC writer cannot create primary keys or constraints, so the tables are
    created here from schema.sql and Spark only appends the rows.)"""
    lines = [ln for ln in open(path, encoding="utf-8") if not ln.strip().startswith("--")]
    statements = [s.strip() for s in "".join(lines).split(";") if s.strip()]
    loader = _jvm.java.lang.Thread.currentThread().getContextClassLoader()
    # Class.newInstance() takes no arguments, so Py4J can call it directly.
    # (getDeclaredConstructor() is a Java varargs method, which Py4J cannot call with no arguments.)
    driver = loader.loadClass("org.postgresql.Driver").newInstance()
    props = _jvm.java.util.Properties()
    props.setProperty("user", PG_USER)
    props.setProperty("password", PG_PASSWORD)
    conn = driver.connect(JDBC_URL, props)
    try:
        stmt = conn.createStatement()
        for s in statements:
            stmt.execute(s)
        stmt.close()
    finally:
        conn.close()
    return len(statements)


def table_columns(table):
    """{column: (data_type, is_nullable)} of a PostgreSQL table, in table order."""
    rows = query(f"SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                 f"WHERE table_schema = 'public' AND table_name = '{table}' "
                 f"ORDER BY ordinal_position").collect()
    return {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in rows}


# =============================================================================
# STEP 1 - CREATE TABLES
# =============================================================================
header("STEP 1 - CREATE TABLES (phase2/schema.sql)")
print(f"  database: {JDBC_URL.split('?')[0]}  user: {PG_USER}")
try:
    n = execute_sql_file(SCHEMA_SQL)
except Exception as e:                                         # connection / SQL problem
    print(f"  ERROR: could not run schema.sql: {e}")
    print("  Is the PostgreSQL container running (docker ps)? Do PG_* settings match it?")
    spark.stop()
    sys.exit(1)
print(f"  {n} SQL statements executed")

data = {name: spark.read.parquet(os.path.join(PROCESSED, f"{name}.parquet")) for name in DATASETS}
schemas = {name: table_columns(table) for name, (table, _) in DATASETS.items()}
for name, (table, pk) in DATASETS.items():
    print(f"  {table:<17} {len(schemas[name])} columns, primary key {pk}")


# =============================================================================
# STEP 2 - PRE-LOAD CHECKS (stop before writing anything if the data does not fit)
# =============================================================================
header("STEP 2 - PRE-LOAD CHECKS")
problems = []
for name, (table, _) in DATASETS.items():
    df, cols = data[name], schemas[name]
    missing = [c for c in df.columns if c not in cols]
    extra = [c for c in cols if c not in df.columns]
    if missing or extra:
        problems.append(f"{table}: columns only in Parquet {missing}, only in table {extra}")
        continue
    not_null = [c for c, (_, nullable) in cols.items() if nullable == "NO"]
    r = df.agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in not_null]).collect()[0]
    bad = {c: r[c] for c in not_null if r[c]}
    if bad:
        problems.append(f"{table}: NULLs in NOT NULL columns {bad}")
    print(f"  {table:<17} columns match; NOT NULL columns without NULLs: "
          f"{len(not_null) - len(bad)}/{len(not_null)}")
if problems:
    for p in problems:
        print(f"  ERROR: {p}")
    print("  Fix phase2/schema.sql (or the data) and run again. Nothing was loaded.")
    spark.stop()
    sys.exit(1)


# =============================================================================
# STEP 3 - LOAD
# =============================================================================
header("STEP 3 - LOAD WITH SPARK JDBC")
for name, (table, _) in DATASETS.items():
    # Select the columns in table order; 4 partitions = 4 parallel JDBC connections.
    (data[name].select(list(schemas[name])).repartition(4)
        .write.jdbc(JDBC_URL, table, mode="append", properties=JDBC_PROPS))
    print(f"  loaded processed/{name}.parquet -> {table}")


# =============================================================================
# STEP 4 - VERIFY
# =============================================================================
header("STEP 4 - VERIFY THE DATABASE AGAINST THE PARQUET FILES")
for name, (table, pk) in DATASETS.items():
    df, cols = data[name], schemas[name]
    spark_types = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    print(f"\n  --- {table} ---")

    # 1. record count
    n_parquet = df.count()
    n_db = query(f"SELECT COUNT(*) AS n FROM {table}").collect()[0]["n"]
    check("row count", n_db == n_parquet, f"database {n_db:,} / parquet {n_parquet:,}")

    # 2. column types
    wrong = [f"{c}: {spark_types[c]} -> {t}" for c, (t, _) in cols.items()
             if t not in TYPE_MATCH.get(spark_types[c], set())]
    check("column types match the Spark types", not wrong, "; ".join(wrong))
    type_summary = {}
    for c, (t, _) in cols.items():
        type_summary[t] = type_summary.get(t, 0) + 1
    print("         " + ", ".join(f"{t} x{k}" for t, k in type_summary.items()))

    # 3. primary key: unique and complete
    n_pk = query(f"SELECT COUNT(DISTINCT {pk}) AS n FROM {table}").collect()[0]["n"]
    check(f"{pk} unique", n_pk == n_parquet, f"{n_pk:,} distinct")

    # 4. non-NULL count of every column (no value lost or turned into NULL)
    db_nn = query("SELECT " + ", ".join(f'COUNT("{c}") AS "{c}"' for c in cols)
                  + f" FROM {table}").collect()[0]
    pq_nn = df.agg(*[F.count(c).alias(c) for c in cols]).collect()[0]
    diff = [f"{c} {db_nn[c]:,}/{pq_nn[c]:,}" for c in cols if db_nn[c] != pq_nn[c]]
    check("non-NULL count of every column", not diff, "; ".join(diff))

    # 5. timestamp ranges, compared as text (catches any time-zone shift)
    ts_cols = [c for c in cols if spark_types[c] == "timestamp"]
    db_ts = query("SELECT " + ", ".join(
        f"TO_CHAR(MIN({c}), 'YYYY-MM-DD HH24:MI:SS') AS min_{c}, "
        f"TO_CHAR(MAX({c}), 'YYYY-MM-DD HH24:MI:SS') AS max_{c}" for c in ts_cols)
        + f" FROM {table}").collect()[0]
    pq_ts = df.agg(*[e for c in ts_cols for e in (
        F.date_format(F.min(c), "yyyy-MM-dd HH:mm:ss").alias(f"min_{c}"),
        F.date_format(F.max(c), "yyyy-MM-dd HH:mm:ss").alias(f"max_{c}"))]).collect()[0]
    diff = [f"{k}: db {db_ts[k]} vs parquet {pq_ts[k]}" for k in db_ts.asDict() if db_ts[k] != pq_ts[k]]
    created = ts_cols[0]
    check("timestamp min/max unchanged", not diff,
          "; ".join(diff) or f"{created} {db_ts['min_' + created]} -> {db_ts['max_' + created]}")

    # 6. numeric and boolean totals of the derived attributes and coordinates
    db_num = query(f"SELECT SUM(resolution_hours) AS hours, "
                   f"SUM(CASE WHEN resolution_valid THEN 1 ELSE 0 END) AS valid, "
                   f"SUM(latitude) AS lat, SUM(longitude) AS lon FROM {table}").collect()[0]
    pq_num = df.agg(F.sum("resolution_hours").alias("hours"),
                    F.sum(F.col("resolution_valid").cast("int")).alias("valid"),
                    F.sum("latitude").alias("lat"), F.sum("longitude").alias("lon")).collect()[0]
    close = lambda a, b: abs(float(a or 0) - float(b or 0)) <= 1e-6 * max(1.0, abs(float(b or 0)))
    check("resolution_hours / resolution_valid / coordinates totals",
          all(close(db_num[k], pq_num[k]) for k in ["hours", "lat", "lon"]) and db_num["valid"] == pq_num["valid"],
          f"valid={db_num['valid']:,}, total hours={float(db_num['hours'] or 0):,.1f}")

    # 7. row-by-row comparison of a random sample of 20 records (every column)
    ids = [r[0] for r in df.select(pk).orderBy(F.rand(seed=236)).limit(20).collect()]
    id_list = ", ".join(str(i) if isinstance(i, int) else f"'{i}'" for i in ids)
    db_rows = query(f"SELECT * FROM {table} WHERE {pk} IN ({id_list})").select(list(cols))
    pq_rows = df.where(F.col(pk).isin(ids)).select(list(cols))
    n_diff = db_rows.exceptAll(pq_rows).count() + pq_rows.exceptAll(db_rows).count()
    check("random sample of 20 rows identical in every column", n_diff == 0 and db_rows.count() == 20,
          f"{n_diff} differing rows")

print("\n  Example rows from unified_requests:")
query("SELECT request_id, source_city, created_at, status_std, channel_std, request_type, "
      "postal_code, resolution_hours, resolution_valid FROM unified_requests "
      "ORDER BY request_id LIMIT 4").show(truncate=False)

print(f"\n  {sum(results)} of {len(results)} checks passed")
spark.stop()
sys.exit(0 if all(results) else 1)

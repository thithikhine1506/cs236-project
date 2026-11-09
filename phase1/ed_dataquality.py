"""
eda.py - CS236 Phase 1: exploratory data analysis of NYC 311 and LA MyLA311.
Run from the Project folder:  python phase1/eda.py

Part 1  Load & scan     : dimensions, schema, inferred types, date ranges, sample records
Part 2  Data quality    : missing values, placeholders, duplicates, ID uniqueness,
                          timestamp logic, postal codes, coordinates, text consistency
"""
import os
import sys
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "CS236-Fall26-Project-Dataset")
NYC_PATH = os.path.join(DATA, "nyc_311_800k.csv")
LA_PATH = os.path.join(DATA, "la_311_800k.csv")
TS_FMT = "yyyy-MM-dd'T'HH:mm:ss.SSS"   # e.g. 2026-10-05T02:05:08.000

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (SparkSession.builder
         .appName("CS236-EDA")
         .master("local[*]")                           # use all CPU cores
         .config("spark.driver.memory", "4g")
         .config("spark.sql.session.timeZone", "UTC")  # keep wall-clock times unchanged
         .config("spark.ui.showConsoleProgress", "false")  # no progress bars in the output
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")


def load(path):
    """Read a CSV with every column as a string, so no value is silently lost."""
    return (spark.read
            .option("header", True)
            .option("inferSchema", False)
            .option("quote", '"')
            .option("escape", '"')
            .csv(path))


# Which column plays which role in each dataset ---------------------------------
CONFIG = {
    "NYC 311": dict(
        path=NYC_PATH,
        id="unique_key", id_pattern=r"^\d+$",
        created="created_date", closed="closed_date", status="status",
        zip="incident_zip", zip_prefix=r"^1[01]",          # NYC ZIPs start 10xxx / 11xxx
        lat="latitude", lon="longitude",
        bbox=(40.45, 40.95, -74.30, -73.65),               # lat_min, lat_max, lon_min, lon_max
        ts_cols=["created_date", "closed_date", "due_date", "resolution_action_updated_date"],
        key_cols=["unique_key", "created_date", "closed_date", "agency", "complaint_type",
                  "status", "open_data_channel_type", "incident_zip", "borough"],
        categorical=["agency", "complaint_type", "descriptor", "location_type", "status",
                     "borough", "city", "open_data_channel_type"],
    ),
    "LA MyLA311": dict(
        path=LA_PATH,
        id="srnumber", id_pattern=r"^1-\d+$",
        created="createddate", closed="closeddate", status="status",
        zip="zipcode", zip_prefix=r"^9[0-3]",              # LA ZIPs start 90xxx-93xxx
        lat="latitude", lon="longitude",
        bbox=(33.65, 34.40, -118.70, -118.10),
        ts_cols=["createddate", "updateddate", "servicedate", "closeddate"],
        key_cols=["srnumber", "createddate", "closeddate", "owner", "requesttype",
                  "status", "requestsource", "zipcode", "apc"],
        categorical=["owner", "requesttype", "status", "requestsource", "actiontaken",
                     "createdbyuserorganization", "apc", "policeprecinct"],
    ),
}

PLACEHOLDERS = ["unspecified", "n/a", "na", "none", "null", "unknown", "-", "?", "0", "00000"]


def header(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def ts(col):
    """Parse a raw timestamp string; blanks and malformed values become NULL."""
    return F.to_timestamp(F.col(col), TS_FMT)


def show_table(rows, columns, n=60):
    """Print a small Python result table through Spark's table formatter."""
    if rows:
        spark.createDataFrame(rows, columns).show(n, truncate=False)
    else:
        print("  (none)")


# =============================================================================
# PART 1 - LOAD & SCAN
# =============================================================================
def load_and_scan(name, df, cfg):
    header(f"PART 1 - LOAD & SCAN: {name}")

    # Dimensions
    print(f"Rows: {df.count():,}   Columns: {len(df.columns)}")

    # Schema as read (all strings)
    df.printSchema()

    # Data types: what Spark would infer from a 10% sample
    inferred = (spark.read.option("header", True).option("inferSchema", True)
                .option("samplingRatio", 0.1).option("escape", '"').csv(cfg["path"]))
    print("Types Spark infers (10% sample):")
    for f in inferred.schema.fields:
        print(f"  {f.name:<32} {f.dataType.simpleString()}")

    # Date ranges. date_format() turns the result into text inside Spark,
    # so Python doesn't shift it into the Mac's local time zone when collecting.
    print("\nDate ranges:")
    for c in cfg["ts_cols"]:
        t = ts(c)
        r = df.agg(F.date_format(F.min(t), "yyyy-MM-dd HH:mm:ss").alias("min"),
                   F.date_format(F.max(t), "yyyy-MM-dd HH:mm:ss").alias("max"),
                   F.count(t).alias("non_null")).collect()[0]
        print(f"  {c:<32} {r['min']}  ->  {r['max']}   (non-null: {r['non_null']:,})")

    # Representative records
    print("\nFirst 5 rows (key columns):")
    df.select(cfg["key_cols"]).show(5, truncate=30)
    print("Random 5 rows (key columns):")
    df.select(cfg["key_cols"]).sample(fraction=0.001, seed=42).limit(5).show(truncate=30)
    print("One full record (all columns):")
    df.show(1, vertical=True, truncate=False)


# =============================================================================
# PART 2 - DATA QUALITY
# =============================================================================
def missing_values(df, n):
    """NULL vs empty-string counts per column.
    To decide which columns are usable or droppable, and whether the source
    encodes 'missing' consistently (NULL only, or also as '')."""
    print("\n[2.1] Missing values per column (sorted by % missing)")
    aggs = []
    for c in df.columns:
        aggs.append(F.sum(F.col(c).isNull().cast("int")).alias(f"{c}__null"))
        aggs.append(F.sum((F.trim(F.col(c)) == "").cast("int")).alias(f"{c}__blank"))
    r = df.agg(*aggs).collect()[0]
    rows = []
    for c in df.columns:
        nulls, blanks = r[f"{c}__null"] or 0, r[f"{c}__blank"] or 0
        rows.append((c, nulls, blanks, round(100.0 * (nulls + blanks) / n, 2)))
    rows.sort(key=lambda x: -x[3])
    show_table(rows, ["column", "null", "empty_string", "pct_missing"])


def placeholders(df, n):
    """Values like 'Unspecified', 'N/A', 'UNKNOWN', '0' are 'missing'."""
    
    print("\n[2.2] Placeholder values meaning 'unknown'")
    norm = lambda c: F.lower(F.trim(F.col(c)))
    r = df.agg(*[F.sum(norm(c).isin(PLACEHOLDERS).cast("int")).alias(c)
                 for c in df.columns]).collect()[0]
    hits = sorted([(c, r[c], round(100.0 * r[c] / n, 2)) for c in df.columns if r[c]],
                  key=lambda x: -x[1])
    show_table(hits, ["column", "placeholder_count", "pct"])
    for c, _, _ in hits[:6]:   # which placeholder strings, in the most affected columns
        vals = (df.where(norm(c).isin(PLACEHOLDERS)).groupBy(c).count()
                  .orderBy(F.desc("count")).limit(4).collect())
        print(f"   {c}: " + ", ".join(f"'{v[0]}'={v[1]:,}" for v in vals))


def duplicates_and_ids(df, n, cfg):
    """Exact duplicate rows, identifier uniqueness and identifier format.
    Every record must be uniquely identifiable after the two cities are merged."""
    print("\n[2.3] Duplicates and identifier uniqueness")
    idc = cfg["id"]
    exact_dups = n - df.dropDuplicates().count()
    distinct_ids = df.select(idc).distinct().count()
    null_ids = df.where(F.col(idc).isNull() | (F.trim(F.col(idc)) == "")).count()
    bad_format = df.where(~F.col(idc).rlike(cfg["id_pattern"])).count()
    lengths = df.agg(F.min(F.length(idc)), F.max(F.length(idc))).collect()[0]
    print(f"  exact duplicate rows        : {exact_dups:,}")
    print(f"  distinct {idc:<18} : {distinct_ids:,} of {n:,} rows")
    print(f"  NULL / blank ids            : {null_ids:,}")
    print(f"  ids not matching {cfg['id_pattern']:<10}: {bad_format:,}")
    print(f"  id length range             : {lengths[0]}-{lengths[1]} characters")


def timestamp_logic(df, n, cfg, name):
    """Contradictions between timestamps, and between status and closed date."""
    print("\n[2.4] Timestamp logic")
    cr, cl = ts(cfg["created"]), ts(cfg["closed"])
    hours = (F.unix_timestamp(cl) - F.unix_timestamp(cr)) / 3600.0
    checks = {
        "created missing or unparseable": cr.isNull(),
        "closed before created (negative duration)": cl < cr,
        "closed == created (zero duration)": cl == cr,
        "closed exactly at 00:00:00 (date-only time)": F.date_format(cl, "HH:mm:ss") == "00:00:00",
        "resolution longer than 365 days": hours > 365 * 24,
    }
    if name.startswith("NYC"):
        checks["resolution_action_updated before created"] = ts("resolution_action_updated_date") < cr
    else:
        checks["servicedate before created"] = ts("servicedate") < cr
        checks["updateddate before created"] = ts("updateddate") < cr
    r = df.agg(*[F.sum(cond.cast("int")).alias(k) for k, cond in checks.items()]).collect()[0]
    show_table([(k, r[k] or 0, round(100.0 * (r[k] or 0) / n, 3)) for k in checks],
               ["check", "count", "pct"])

    print("  Status vs. closed date (does 'Closed' always have a closed date?)")
    (df.groupBy(cfg["status"], cl.isNotNull().alias("has_closed_date")).count()
       .orderBy(cfg["status"], "has_closed_date").show(truncate=False))


def postal_codes(df, n, cfg):
    """Format of the postal-code column. Postal code must be uniform 5-digit string"""
    print("\n[2.5] Postal code validity")
    z = F.trim(F.col(cfg["zip"]))
    fmt = (F.when(F.col(cfg["zip"]).isNull() | (z == ""), "missing")
            .when(z.rlike(r"^\d{5}$") & z.rlike(cfg["zip_prefix"]), "valid 5-digit, in city range")
            .when(z.rlike(r"^\d{5}$"), "5-digit but outside city range")
            .when(z.rlike(r"^\d{5}-\d{4}$"), "ZIP+4")
            .otherwise("malformed"))
    (df.groupBy(fmt.alias("zip_format")).count()
       .withColumn("pct", F.round(100.0 * F.col("count") / n, 3))
       .orderBy(F.desc("count")).show(truncate=False))
    print("  Examples of non-valid values:")
    (df.where(~(z.rlike(r"^\d{5}$") & z.rlike(cfg["zip_prefix"])) & z.isNotNull() & (z != ""))
       .groupBy(cfg["zip"]).count().orderBy(F.desc("count")).show(10, truncate=False))


def coordinates(df, cfg):
    """Latitude/longitude parseability and plausibility.
    Zero or out-of-city coordinates are errors that would misplace requests."""
    print("\n[2.6] Coordinate validity")
    lat = F.col(cfg["lat"]).cast("double")
    lon = F.col(cfg["lon"]).cast("double")
    lat0, lat1, lon0, lon1 = cfg["bbox"]
    outside = (lat < lat0) | (lat > lat1) | (lon < lon0) | (lon > lon1)
    (df.agg(F.sum(lat.isNull().cast("int")).alias("missing"),
            F.sum(((lat == 0) | (lon == 0)).cast("int")).alias("zero"),
            F.sum((lat.isNotNull() & outside).cast("int")).alias("outside_city_box"),
            F.min(lat).alias("min_lat"), F.max(lat).alias("max_lat"),
            F.min(lon).alias("min_lon"), F.max(lon).alias("max_lon"))
       .show(truncate=False))


def text_consistency(df, cfg):
    """Case / whitespace variants inside categorical columns.
    'PLUMBING' and 'Plumbing' are one category; normalizing before grouping
    prevents the same category being counted twice."""
    print("\n[2.7] Text consistency in categorical columns")
    norm = lambda c: F.upper(F.regexp_replace(F.trim(F.col(c)), r"\s+", " "))
    r = df.agg(*[F.countDistinct(c).alias(f"{c}__raw") for c in cfg["categorical"]],
               *[F.countDistinct(norm(c)).alias(f"{c}__norm") for c in cfg["categorical"]]).collect()[0]
    show_table([(c, r[f"{c}__raw"], r[f"{c}__norm"], r[f"{c}__raw"] - r[f"{c}__norm"])
                for c in cfg["categorical"]],
               ["column", "distinct_raw", "distinct_normalized", "variants"])
    for c in cfg["categorical"]:
        if r[f"{c}__raw"] != r[f"{c}__norm"]:
            ex = (df.groupBy(norm(c).alias("normalized"))
                    .agg(F.collect_set(c).alias("raw_variants"))
                    .where(F.size("raw_variants") > 1).limit(5).collect())
            for e in ex:
                print(f"   {c}: {e['raw_variants']}")


def data_quality(name, df, cfg):
    header(f"PART 2 - DATA QUALITY: {name}")
    n = df.count()
    missing_values(df, n)
    placeholders(df, n)
    duplicates_and_ids(df, n, cfg)
    timestamp_logic(df, n, cfg, name)
    postal_codes(df, n, cfg)
    coordinates(df, cfg)
    text_consistency(df, cfg)


# =============================================================================
# MAIN
# =============================================================================
raw = {}
for name, cfg in CONFIG.items():
    # persist: the data is scanned many times below, so keep it in memory (spill to disk)
    raw[name] = load(cfg["path"]).persist(StorageLevel.MEMORY_AND_DISK)

for name, cfg in CONFIG.items():
    load_and_scan(name, raw[name], cfg)

for name, cfg in CONFIG.items():
    data_quality(name, raw[name], cfg)

# Cross-dataset check: could a NYC id and a LA id ever be the same string?
header("CROSS-DATASET: identifier overlap")
overlap = (raw["NYC 311"].select(F.col("unique_key").alias("id"))
           .intersect(raw["LA MyLA311"].select(F.col("srnumber").alias("id"))).count())
print(f"Identical id strings in both datasets: {overlap}")

spark.stop()
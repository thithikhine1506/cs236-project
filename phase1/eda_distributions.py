"""
eda.py - CS236 Phase 1: exploratory data analysis of NYC 311 and LA MyLA311.
Run from the Project folder:  python phase1/eda.py

Part 1  Load & scan     : dimensions, schema, inferred types, date ranges, sample records
Part 2  Data quality    : missing values, placeholders, duplicates, ID uniqueness,
                          timestamp logic, postal codes, coordinates, text consistency
Part 3  Distributions   : status / channel / agency, request types, time patterns,
                          resolution time, geography (plots -> phase1/output/eda/plots/)

"""
import os
import sys

import matplotlib
matplotlib.use("Agg")                 # save figures to files, never open a window
import matplotlib.pyplot as plt
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "CS236-Fall26-Project-Dataset")
NYC_PATH = os.path.join(DATA, "nyc_311_800k.csv")
LA_PATH = os.path.join(DATA, "la_311_800k.csv")
TS_FMT = "yyyy-MM-dd'T'HH:mm:ss.SSS"   # e.g. 2026-10-05T02:05:08.000
PLOT_DIR = os.path.join(BASE, "phase1", "output", "eda", "plots")

# Make Spark's worker processes use the same Python interpreter as this script.
# Without this, Spark starts the first 'python3' on the PATH (here 3.14),
# which differs from the driver (miniforge 3.12) and crashes.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (SparkSession.builder
         .appName("CS236-EDA")
         .master("local[*]")                               # use all CPU cores
         .config("spark.driver.memory", "4g")
         .config("spark.sql.session.timeZone", "UTC")      # keep wall-clock times unchanged
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


# Which column plays which role in each dataset.
CONFIG = {
    "NYC 311": dict(
        path=NYC_PATH,
        id="unique_key", id_pattern=r"^\d+$",
        created="created_date", closed="closed_date", status="status",
        channel="open_data_channel_type", agency="agency", rtype="complaint_type",
        area="borough",
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
        channel="requestsource", agency="owner", rtype="requesttype",
        area="apc",
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
    # so Python doesn't shift it into the computer's local time zone.
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
    """NULL vs empty-string counts per column."""
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
    """Values like 'Unspecified', 'N/A', 'UNKNOWN', '0' that mean 'missing'."""
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
    Every record must stay uniquely identifiable after the two cities are merged."""
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
    """Format of the postal-code column, must be a uniform 5-digit string."""
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
# PART 3 - DISTRIBUTIONS
# =============================================================================
COLORS = {"NYC 311": "#2a6fdb", "LA MyLA311": "#e8833a"}
WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]   # Spark dayofweek: 1 = Sunday


def save_plot(fig, filename):
    """Save a matplotlib figure into folder phase1/output/eda/plots/."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=130)
    plt.close(fig)
    print(f"  plot saved: phase1/output/eda/plots/{filename}")


def value_counts(df, col, n, total):
    """Top-n values of a column with count and % of all rows (NULL shown as '<NULL>')."""
    rows = (df.groupBy(F.coalesce(F.col(col), F.lit("<NULL>")).alias("value")).count()
              .orderBy(F.desc("count")).limit(n).collect())
    return [(r["value"], r["count"], round(100.0 * r["count"] / total, 2)) for r in rows]


def categorical_distributions(raw, totals):
    """Status, submission channel and responsible agency."""
    header("PART 3.1 - STATUS, CHANNEL AND AGENCY")
    for role, title in [("status", "Request status"),
                        ("channel", "Submission channel"),
                        ("agency", "Responsible agency")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, (name, cfg) in zip(axes, CONFIG.items()):
            col = cfg[role]
            n_distinct = raw[name].select(F.countDistinct(col)).collect()[0][0]
            rows = value_counts(raw[name], col, 15, totals[name])
            print(f"\n{name} - {col} ({n_distinct} distinct values)")
            show_table(rows, [col, "count", "pct"])
            ax.barh([r[0] for r in rows][::-1], [r[2] for r in rows][::-1], color=COLORS[name])
            ax.set_title(f"{name}: {col}")
            ax.set_xlabel("% of requests")
        fig.suptitle(title)
        save_plot(fig, f"3_1_{role}.png")


def request_types(raw, totals):
    """Request type (complaint_type / requesttype)."""
    header("PART 3.2 - REQUEST TYPES")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (name, cfg) in zip(axes, CONFIG.items()):
        col = cfg["rtype"]
        n_distinct = raw[name].select(F.countDistinct(col)).collect()[0][0]
        rows = value_counts(raw[name], col, 15, totals[name])
        print(f"\n{name} - {col}: {n_distinct} distinct types, "
              f"top 15 cover {sum(r[2] for r in rows):.1f}% of requests")
        show_table(rows, [col, "count", "pct"])
        ax.barh([r[0] for r in rows][::-1], [r[2] for r in rows][::-1], color=COLORS[name])
        ax.set_title(f"{name}: top {len(rows)} of {n_distinct} request types")
        ax.set_xlabel("% of requests")
        ax.tick_params(axis="y", labelsize=8)
    save_plot(fig, "3_2_request_types.png")


def temporal_patterns(raw, totals):
    """Requests by month, weekday, hour and within-month period.
    It shows coverage (partial months), daily rhythms and recording artefacts."""
    header("PART 3.3 - TEMPORAL PATTERNS (by created time)")
    period_label = (F.when(F.dayofmonth("t") <= 7, "P1: days 1-7")
                     .when(F.dayofmonth("t") <= 14, "P2: days 8-14")
                     .when(F.dayofmonth("t") <= 21, "P3: days 15-21")
                     .otherwise("P4: days 22-end"))
    results = {}
    for name, cfg in CONFIG.items():
        t = raw[name].select(ts(cfg["created"]).alias("t"))
        month = (t.groupBy(F.date_format("t", "yyyy-MM").alias("month"))
                  .agg(F.count("*").alias("requests"),
                       F.countDistinct(F.to_date("t")).alias("days_covered"))
                  .orderBy("month").collect())
        dow = t.groupBy(F.dayofweek("t").alias("d")).count().orderBy("d").collect()
        hour = t.groupBy(F.hour("t").alias("h")).count().orderBy("h").collect()
        period = (t.groupBy(period_label.alias("period"))
                   .agg(F.count("*").alias("requests"),
                        F.countDistinct(F.to_date("t")).alias("days_covered"))
                   .orderBy("period").collect())
        results[name] = dict(month=month, dow=dow, hour=hour, period=period)
        n = totals[name]

        print(f"\n{name} - requests per month (partial months have fewer days covered)")
        show_table([(r["month"], r["requests"], r["days_covered"],
                     round(r["requests"] / r["days_covered"])) for r in month],
                   ["month", "requests", "days_covered", "requests_per_day"])
        print(f"{name} - requests per within-month period "
              "(P4 has 7-10 days, so compare requests_per_day, not raw counts)")
        show_table([(r["period"], r["requests"], round(100.0 * r["requests"] / n, 2),
                     r["days_covered"], round(r["requests"] / r["days_covered"])) for r in period],
                   ["period", "requests", "pct", "days_covered", "requests_per_day"])
        print(f"{name} - requests per weekday")
        show_table([(WEEKDAYS[r["d"] - 1], r["count"], round(100.0 * r["count"] / n, 2)) for r in dow],
                   ["weekday", "requests", "pct"])
        peak = max(hour, key=lambda r: r["count"])
        low = min(hour, key=lambda r: r["count"])
        print(f"{name} - busiest hour {peak['h']:02d}:00 ({peak['count']:,}), "
              f"quietest hour {low['h']:02d}:00 ({low['count']:,})")

    # Figure 1: monthly volume
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name in zip(axes, CONFIG):
        m = results[name]["month"]
        bars = ax.bar([r["month"] for r in m], [r["requests"] for r in m], color=COLORS[name])
        for b, r in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r['days_covered']}d",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{name}: requests per month (label = days covered)")
        ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "3_3_monthly_volume.png")

    # Figure 2: hour of day and weekday, as % of each city's requests
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in CONFIG:
        n = totals[name]
        h = results[name]["hour"]
        d = results[name]["dow"]
        axes[0].plot([r["h"] for r in h], [100.0 * r["count"] / n for r in h],
                     marker="o", color=COLORS[name], label=name)
        axes[1].plot([WEEKDAYS[r["d"] - 1] for r in d], [100.0 * r["count"] / n for r in d],
                     marker="o", color=COLORS[name], label=name)
    axes[0].set_title("Requests by hour created (local time)")
    axes[0].set_xlabel("hour")
    axes[0].set_xticks(range(0, 24, 2))
    axes[1].set_title("Requests by weekday created")
    for ax in axes:
        ax.set_ylabel("% of city's requests")
        ax.legend()
        ax.grid(alpha=0.3)
    save_plot(fig, "3_3_hour_and_weekday.png")

    # Figure 3: within-month periods, normalized per day covered
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    for i, name in enumerate(CONFIG):
        p = results[name]["period"]
        per_day = [r["requests"] / r["days_covered"] for r in p]
        mean = sum(per_day) / len(per_day)
        xs = [j + (i - 0.5) * width for j in range(len(p))]
        ax.bar(xs, [100.0 * v / mean for v in per_day], width, color=COLORS[name], label=name)
    ax.axhline(100, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels([r["period"] for r in results["NYC 311"]["period"]])
    ax.set_ylabel("requests per day (% of city average)")
    ax.set_title("Within-month periods, normalized by days covered")
    ax.legend()
    save_plot(fig, "3_3_month_periods.png")


def resolution_time(raw):
    """Resolution time = closed - created, for requests that are Closed with a valid closed date.
    Resolution time is a required derived attribute; its spread decides how outliers are
    treated and whether mean or median is the better summary."""
    header("PART 3.4 - RESOLUTION TIME (status Closed, closed >= created)")
    rows, hists, by_type = [], {}, {}
    for name, cfg in CONFIG.items():
        cr, cl = ts(cfg["created"]), ts(cfg["closed"])
        hours = ((F.unix_timestamp(cl) - F.unix_timestamp(cr)) / 3600.0).alias("hours")
        valid = (raw[name]
                 .where((F.lower(F.trim(F.col(cfg["status"]))) == "closed") & cl.isNotNull() & (cl >= cr))
                 .select(F.col(cfg["rtype"]).alias("rtype"), hours))
        q = valid.agg(F.count("*").alias("n"),
                      F.mean("hours").alias("mean"),
                      F.percentile_approx("hours", [0.05, 0.25, 0.5, 0.75, 0.95, 0.99], 10000).alias("p"),
                      F.max("hours").alias("max"),
                      F.sum((F.col("hours") == 0).cast("int")).alias("zero")).collect()[0]
        rows.append((name, q["n"], q["zero"], round(q["mean"], 1), round(q["p"][0], 2),
                     round(q["p"][1], 2), round(q["p"][2], 2), round(q["p"][3], 1),
                     round(q["p"][4], 1), round(q["p"][5], 1), round(q["max"], 1)))
        # histogram on a log scale: 4 bins per factor of 10 (zero durations excluded)
        hists[name] = (valid.where("hours > 0")
                       .groupBy(F.floor(F.log10("hours") * 4).alias("b")).count()
                       .orderBy("b").collect())
        by_type[name] = (valid.groupBy("rtype")
                         .agg(F.count("*").alias("closed"),
                              F.round(F.percentile_approx("hours", 0.5, 10000), 1).alias("median_h"),
                              F.round(F.mean("hours"), 1).alias("mean_h"))
                         .orderBy(F.desc("closed")).limit(8).collect())

    print("Resolution time in hours")
    show_table(rows, ["dataset", "n_valid", "zero", "mean", "p05", "p25", "median",
                      "p75", "p95", "p99", "max"])
    for name in CONFIG:
        print(f"{name} - median / mean resolution hours for the 8 most common request types")
        show_table([(r["rtype"], r["closed"], r["median_h"], r["mean_h"]) for r in by_type[name]],
                   ["request_type", "closed", "median_h", "mean_h"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in CONFIG:
        h = hists[name]
        total = sum(r["count"] for r in h)
        ax.step([10 ** (r["b"] / 4.0) for r in h], [100.0 * r["count"] / total for r in h],
                where="post", color=COLORS[name], label=name)
    ax.set_xscale("log")
    for x, label in [(1, "1 hour"), (24, "1 day"), (168, "1 week"), (720, "30 days"), (8760, "1 year")]:
        ax.axvline(x, color="grey", lw=0.6, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.97, label, fontsize=8, ha="left", va="top")
    ax.set_xlabel("resolution time (hours, log scale)")
    ax.set_ylabel("% of closed requests")
    ax.set_title("Distribution of resolution time")
    ax.legend()
    save_plot(fig, "3_4_resolution_time.png")


def geography(raw, totals):
    """Area, postal codes and a point map."""
    header("PART 3.5 - GEOGRAPHY")
    for name, cfg in CONFIG.items():
        df, n = raw[name], totals[name]
        z = F.trim(F.col(cfg["zip"]))
        n_zip = df.where(z.rlike(r"^\d{5}$") & z.rlike(cfg["zip_prefix"])).select(
            F.countDistinct(z)).collect()[0][0]
        print(f"\n{name} - {cfg['area']} distribution")
        show_table(value_counts(df, cfg["area"], 10, n), [cfg["area"], "count", "pct"])
        top = value_counts(df, cfg["zip"], 10, n)
        print(f"{name} - {n_zip} distinct valid postal codes; top 10 hold "
              f"{sum(r[2] for r in top):.1f}% of requests")
        show_table(top, [cfg["zip"], "count", "pct"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (name, cfg) in zip(axes, CONFIG.items()):
        lat0, lat1, lon0, lon1 = cfg["bbox"]
        pts = (raw[name].select(F.col(cfg["lat"]).cast("double").alias("lat"),
                                F.col(cfg["lon"]).cast("double").alias("lon"))
               .where(f"lat between {lat0} and {lat1} and lon between {lon0} and {lon1}")
               .sample(fraction=0.05, seed=1).collect())
        ax.scatter([p["lon"] for p in pts], [p["lat"] for p in pts], s=0.2, alpha=0.3,
                   color=COLORS[name])
        ax.set_title(f"{name}: 5% sample of request locations")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect(1.3)
        ax.locator_params(axis="x", nbins=5)     # fewer longitude labels, so they don't overlap
    save_plot(fig, "3_5_request_locations.png")


def distributions(raw):
    totals = {name: raw[name].count() for name in raw}
    categorical_distributions(raw, totals)
    request_types(raw, totals)
    temporal_patterns(raw, totals)
    resolution_time(raw)
    geography(raw, totals)


# =============================================================================
# MAIN
# =============================================================================
# Which parts to run: all by default, or the ones given on the command line.
PARTS = set(sys.argv[1:]) or {"1", "2", "3"}

raw = {}
for name, cfg in CONFIG.items():
    # persist: the data is scanned many times below, so keep it in memory (spill to disk)
    raw[name] = load(cfg["path"]).persist(StorageLevel.MEMORY_AND_DISK)

if "1" in PARTS:
    for name, cfg in CONFIG.items():
        load_and_scan(name, raw[name], cfg)

if "2" in PARTS:
    for name, cfg in CONFIG.items():
        data_quality(name, raw[name], cfg)
    # Cross-dataset check: could a NYC id and a LA id ever be the same string?
    header("CROSS-DATASET: identifier overlap")
    overlap = (raw["NYC 311"].select(F.col("unique_key").alias("id"))
               .intersect(raw["LA MyLA311"].select(F.col("srnumber").alias("id"))).count())
    print(f"Identical id strings in both datasets: {overlap}")

if "3" in PARTS:
    distributions(raw)

spark.stop()
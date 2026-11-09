"""
eda.py - CS236 Phase 1: exploratory data analysis of NYC 311 and LA MyLA311.

Part 1(Load & scan) : dimensions, schema, inferred types, date ranges, sample records.
Part 2(Data quality): missing values, placeholders, duplicates, ID uniqueness, timestamp logic, postal codes, coordinates, text consistency.
Part 3(Distributions) : status / channel / agency, request types, time patterns, resolution time, geography.
Part 4(Schema comparison): attribute mapping, single-source attributes, type/format differences, categorical representations.

"""

import os
import sys

import matplotlib
matplotlib.use("Agg")                 
import matplotlib.pyplot as plt
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "CS236-Fall26-Project-Dataset")
NYC_PATH = os.path.join(DATA, "nyc_311_800k.csv")
LA_PATH = os.path.join(DATA, "la_311_800k.csv")
TS_FMT = "yyyy-MM-dd'T'HH:mm:ss.SSS"   
PLOT_DIR = os.path.join(BASE, "phase1", "output", "eda", "plots")

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (SparkSession.builder
         .appName("CS236-EDA")
         .master("local[*]")                               
         .config("spark.driver.memory", "4g")
         .config("spark.sql.session.timeZone", "UTC")      
         .config("spark.ui.showConsoleProgress", "false")  
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


# A dictionary, the request ID is unique_key in NYC and srnumber in LA, and the agency is agency vs owner. 
# It also holds each city’s valid ZIP prefix and coordinate bounding box.
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



# PART 1 - LOAD & SCAN

def load_and_scan(name, df, cfg):
    header(f"PART 1 - LOAD & SCAN: {name}")

    # Dimensions (row and column counts)
    print(f"Rows: {df.count():,}   Columns: {len(df.columns)}")

    # Schema as read (all strings)
    df.printSchema()

    # Data types: what Spark would infer from a 10% sample
    inferred = (spark.read.option("header", True).option("inferSchema", True)
                .option("samplingRatio", 0.1).option("escape", '"').csv(cfg["path"]))
    print("Types Spark infers (10% sample):")
    for f in inferred.schema.fields:
        print(f"  {f.name:<32} {f.dataType.simpleString()}")

    # Date ranges. date_format() turns the result into text inside Spark.
    
    print("\nDate ranges:")
    for c in cfg["ts_cols"]:
        t = ts(c)
        r = df.agg(F.date_format(F.min(t), "yyyy-MM-dd HH:mm:ss").alias("min"),
                   F.date_format(F.max(t), "yyyy-MM-dd HH:mm:ss").alias("max"),
                   F.count(t).alias("non_null")).collect()[0]
        print(f"  {c:<32} {r['min']}  ->  {r['max']}   (non-null: {r['non_null']:,})")

    # Representative records, the first 5 random rows, and one full record.
    print("\nFirst 5 rows (key columns):")
    df.select(cfg["key_cols"]).show(5, truncate=30)
    print("Random 5 rows (key columns):")
    df.select(cfg["key_cols"]).sample(fraction=0.001, seed=42).limit(5).show(truncate=30)
    print("One full record (all columns):")
    df.show(1, vertical=True, truncate=False)


# PART 2 - DATA QUALITY

# this function is to check NULL and empty-string counts per column, sorted by % missing
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

# this function is to check “missing” values such as “Unspecified” or “0”, and where they occur
def placeholders(df, n):
    """'Unspecified', 'N/A', 'UNKNOWN', '0' mean 'missing'."""
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

# to check duplicate rows, if IDs are unique. 
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

# to check impossible times (closed before created, durations over a year) and “Closed” requests with no closed date
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

#to check whether ZIPs are 5 digits and inside the city’s range.
def postal_codes(df, n, cfg):
    """Format of the postal-code column must be a uniform 5-digit string."""
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

# to checks whether each request’s latitude and longitude are usable and believable.
def coordinates(df, cfg):
    """Latitude/longitude parseability and plausibility.
    Zero or out-of-city coordinates are errors."""
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

# to check spelling variants such as “PLUMBING” vs “Plumbing” in category columns.
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

# to run all seven in order.
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



# PART 3 - DISTRIBUTIONS (to produce plots)

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

# output plot is status, channel and agency, as 3 side-by-side bar charts.
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

# output plot is top request types for each city.
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

# output plot is requests per month (with days covered), hour of day, weekday and within-month periods (normalized per day)
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

# output plot is time from created to closed: percentiles, median by type, and a log-scale histogram
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

# output plot is district and ZIP distribution, with a map of a 5% sample of request locations
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




# PART 4 - SCHEMA COMPARISON
# Each row: (concept, NYC column, LA column, proposed type in the unified schema, note).
# None means the source has no column.

# It's a hand-built table pairing each shared concept with its NYC column, its LA column, a proposed unified type and a note.
SCHEMA_MAP = [
    ("request id",               "unique_key",                     "srnumber",       "string",    "NYC 8 digits vs LA '1-' + 10 digits"),
    ("created timestamp",        "created_date",                   "createddate",    "timestamp", "same ISO format, local time"),
    ("closed timestamp",         "closed_date",                    "closeddate",     "timestamp", "same ISO format, local time"),
    ("last update timestamp",    "resolution_action_updated_date", "updateddate",    "timestamp", "similar meaning"),
    ("due / service date",       "due_date",                       "servicedate",    "timestamp", "DIFFERENT meaning: NYC target date vs LA service date"),
    ("status",                   "status",                         "status",         "string",    "same name, different vocabularies"),
    ("submission channel",       "open_data_channel_type",         "requestsource",  "string",    "different vocabularies"),
    ("agency code",              "agency",                         "owner",          "string",    "different agencies, no shared codes"),
    ("agency full name",         "agency_name",                    None,             "string",    "LA has codes only"),
    ("request type",             "complaint_type",                 "requesttype",    "string",    "different taxonomies"),
    ("request detail",           "descriptor",                     None,             "string",    "LA has no sub-type"),
    ("resolution / action",      "resolution_description",         "actiontaken",    "string",    "NYC free text vs LA short code"),
    ("postal code",              "incident_zip",                   "zipcode",        "string",    "keep as 5-char text (leading zeros)"),
    ("full address",             "incident_address",               "address",        "string",    "LA address ends with ', <zip>'"),
    ("street name",              "street_name",                    "streetname",     "string",    "LA splits direction / name / suffix"),
    ("city / locality",          "city",                           None,             "string",    "LA rows are all Los Angeles"),
    ("district",                 "borough",                        "apc",            "string",    "different units: 5 boroughs vs 7 APCs"),
    ("council district",         "council_district",               "cd",             "integer",   "NYC zero-padded ('09') vs LA ('9')"),
    ("police precinct",          "police_precinct",                "policeprecinct", "string",    "NYC 'Precinct 60' vs LA 'SOUTHWEST'"),
    ("neighborhood body",        "community_board",                "ncname",         "string",    "community board vs neighborhood council"),
    ("latitude",                 "latitude",                       "latitude",       "double",    "same name"),
    ("longitude",                "longitude",                      "longitude",      "double",    "same name"),
    ("point (WKT)",              "location",                       "location",       "string",    "same name, redundant with lat/lon"),
]
SCHEMA_CSV = os.path.join(BASE, "phase1", "output", "eda", "schema_comparison.csv")


def first_values(df):
    """One example (first non-NULL) value for every column, computed in a single pass."""
    r = df.agg(*[F.first(c, ignorenulls=True).alias(c) for c in df.columns]).collect()[0]
    return r.asDict()

# To shows the matched columns with example values and % missing, and saves schema_comparison.csv
def attribute_mapping(raw, totals):
    """Side-by-side mapping of the attributes both sources share.
    This table is the blueprint for the unified schema - which columns are merged,
    under what common name and type, and which differences have to be resolved."""
    header("PART 4.1 - ATTRIBUTE MAPPING (overlapping attributes)")
    nyc, la = raw["NYC 311"], raw["LA MyLA311"]
    ex = {"NYC 311": first_values(nyc), "LA MyLA311": first_values(la)}
    miss = {}
    for name, df in raw.items():
        r = df.agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).collect()[0]
        miss[name] = {c: round(100.0 * r[c] / totals[name], 1) for c in df.columns}

    rows = []
    for concept, a, b, target, note in SCHEMA_MAP:
        if a and b:
            match = "same name" if a == b else "renamed"
        else:
            match = "NYC only" if a else "LA only"
        rows.append((concept, match,
                     a or "-", str(ex["NYC 311"].get(a, "-"))[:28] if a else "-",
                     miss["NYC 311"][a] if a else None,
                     b or "-", str(ex["LA MyLA311"].get(b, "-"))[:28] if b else "-",
                     miss["LA MyLA311"][b] if b else None,
                     target, note))
    cols = ["concept", "match", "nyc_column", "nyc_example", "nyc_pct_null",
            "la_column", "la_example", "la_pct_null", "unified_type", "note"]
    show_table(rows, cols)

    # save the mapping.
    import csv
    os.makedirs(os.path.dirname(SCHEMA_CSV), exist_ok=True)
    with open(SCHEMA_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    print(f"  table saved: phase1/output/eda/schema_comparison.csv")

    both = sum(1 for _, a, b, _, _ in SCHEMA_MAP if a and b)
    same = sum(1 for _, a, b, _, _ in SCHEMA_MAP if a and b and a == b)
    print(f"\n  {both} concepts exist in both sources: {same} with the same column name, "
          f"{both - same} renamed.")

# To list the columns that exist in only one city.
def source_only_attributes(raw, totals):
    """Columns that exist in only one source."""
    header("PART 4.2 - ATTRIBUTES AVAILABLE IN ONLY ONE SOURCE")
    mapped = {"NYC 311": {a for _, a, _, _, _ in SCHEMA_MAP if a},
              "LA MyLA311": {b for _, _, b, _, _ in SCHEMA_MAP if b}}
    for name, df in raw.items():
        only = [c for c in df.columns if c not in mapped[name]]
        ex = first_values(df.select(only))
        r = df.agg(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in only]).collect()[0]
        print(f"\n{name}: {len(only)} columns with no counterpart in the other city")
        show_table([(c, str(ex[c])[:40], round(100.0 * r[c] / totals[name], 1)) for c in only],
                   ["column", "example", "pct_null"])
    for _, a, b, _, note in SCHEMA_MAP:
        if not (a and b):
            print(f"  also single-source by concept: {a or b} ({note})")

# To format differences such as NYC council district “09” vs LA “9”, and coordinate precision.
def type_and_format_differences(raw):
    """Columns that are the same thing but are stored in different formats."""
    header("PART 4.3 - TYPE AND FORMAT DIFFERENCES IN OVERLAPPING COLUMNS")
    nyc, la = raw["NYC 311"], raw["LA MyLA311"]

    def sample(df, col, n=6):
        return ", ".join(str(r[0]) for r in df.where(F.col(col).isNotNull())
                         .groupBy(col).count().orderBy(F.desc("count")).limit(n).collect())

    def pattern_summary(df, col):
        """Share of values that are digits only, have a leading zero, or contain letters."""
        v = F.trim(F.col(col))
        r = df.where(F.col(col).isNotNull()).agg(
            F.count("*").alias("n"),
            F.sum(v.rlike(r"^\d+$").cast("int")).alias("digits"),
            F.sum(v.rlike(r"^0\d").cast("int")).alias("lead0"),
            F.sum(v.rlike(r"[A-Za-z]").cast("int")).alias("letters"),
            F.min(F.length(v)).alias("minlen"), F.max(F.length(v)).alias("maxlen")).collect()[0]
        pct = lambda x: round(100.0 * (x or 0) / max(r["n"], 1), 1)
        return (pct(r["digits"]), pct(r["lead0"]), pct(r["letters"]), f"{r['minlen']}-{r['maxlen']}")

    pairs = [("request id", "unique_key", "srnumber"),
             ("postal code", "incident_zip", "zipcode"),
             ("council district", "council_district", "cd"),
             ("police precinct", "police_precinct", "policeprecinct"),
             ("district", "borough", "apc"),
             ("neighborhood body", "community_board", "ncname"),
             ("latitude", "latitude", "latitude")]
    rows = []
    for concept, a, b in pairs:
        pa, pb = pattern_summary(nyc, a), pattern_summary(la, b)
        rows.append((concept, f"NYC {a}", *pa))
        rows.append(("", f"LA  {b}", *pb))
    show_table(rows, ["concept", "column", "pct_digits_only", "pct_leading_zero",
                      "pct_has_letters", "length_range"])
    print("  Most common values side by side:")
    for concept, a, b in pairs[2:6]:
        print(f"  {concept:<18} NYC {a}: {sample(nyc, a)}")
        print(f"  {'':<18} LA  {b}: {sample(la, b)}")

    # decimal precision of coordinates
    for name, df in raw.items():
        r = df.where(F.col("latitude").isNotNull()).agg(
            F.max(F.length(F.split("latitude", r"\.").getItem(1))).alias("maxdec"),
            F.min(F.length(F.split("latitude", r"\.").getItem(1))).alias("mindec")).collect()[0]
        print(f"  {name} latitude decimals: {r['mindec']}-{r['maxdec']}")

# For different vocabularies.
def categorical_representations(raw):
    """How each source spells the values of shared categorical attributes.
    Identical meanings written differently ('PHONE' vs 'Call', 'Cancelled' only in LA)
    must be mapped to one standardized vocabulary."""
    header("PART 4.4 - CATEGORICAL REPRESENTATIONS")
    nyc, la = raw["NYC 311"], raw["LA MyLA311"]

    def vocab(df, col):
        return [(r[0], r[1]) for r in df.groupBy(col).count().orderBy(F.desc("count")).collect()]

    for concept, a, b in [("status", "status", "status"),
                          ("submission channel", "open_data_channel_type", "requestsource")]:
        va, vb = vocab(nyc, a), vocab(la, b)
        print(f"\n{concept}: NYC {len(va)} values vs LA {len(vb)} values")
        rows = [(va[i][0] if i < len(va) else "", va[i][1] if i < len(va) else None,
                 vb[i][0] if i < len(vb) else "", vb[i][1] if i < len(vb) else None)
                for i in range(max(len(va), len(vb)))]
        show_table(rows, [f"NYC {a}", "nyc_count", f"LA {b}", "la_count"])
        shared = {str(x[0]).upper() for x in va} & {str(x[0]).upper() for x in vb}
        print(f"  values spelled the same in both (ignoring case): {sorted(shared) or 'none'}")

    # agency codes and request types: how much literal overlap is there?
    for concept, a, b in [("agency code", "agency", "owner"),
                          ("request type", "complaint_type", "requesttype")]:
        sa = {r[0].strip().upper() for r in nyc.select(a).distinct().collect() if r[0]}
        sb = {r[0].strip().upper() for r in la.select(b).distinct().collect() if r[0]}
        print(f"\n{concept}: NYC {len(sa)} distinct, LA {len(sb)} distinct, "
              f"{len(sa & sb)} identical (ignoring case): {sorted(sa & sb) or 'none'}")

    # writing style (case) of the main text columns
    print("\nWriting style of categorical text (share of values written in ALL CAPS):")
    rows = []
    for name, df, cols in [("NYC 311", nyc, ["status", "open_data_channel_type", "agency",
                                              "complaint_type", "borough"]),
                           ("LA MyLA311", la, ["status", "requestsource", "owner",
                                               "requesttype", "apc"])]:
        r = df.agg(*[F.sum(((F.col(c) == F.upper(F.col(c))) &
                            F.col(c).rlike("[A-Za-z]")).cast("int")).alias(c) for c in cols],
                   *[F.count(c).alias(f"{c}__n") for c in cols]).collect()[0]
        for c in cols:
            rows.append((name, c, round(100.0 * r[c] / r[f"{c}__n"], 1)))
    show_table(rows, ["dataset", "column", "pct_all_caps"])

    # yes/no style flags exist only in LA
    print("LA-only flag columns and their codes:")
    for c in ["anonymous", "addressverified", "approximateaddress"]:
        print(f"  {c}: " + ", ".join(f"{v}={n:,}" for v, n in vocab(la, c)))


def schema_comparison(raw):
    totals = {name: raw[name].count() for name in raw}
    attribute_mapping(raw, totals)
    source_only_attributes(raw, totals)
    type_and_format_differences(raw)
    categorical_representations(raw)




# MAIN

PARTS = set(sys.argv[1:]) or {"1", "2", "3", "4"}

raw = {}
for name, cfg in CONFIG.items():
    # persist: the data is scanned many times below.
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

if "4" in PARTS:
    schema_comparison(raw)

spark.stop()
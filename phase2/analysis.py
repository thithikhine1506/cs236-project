"""
analysis.py - CS236 Phase 2, section 2.1: Spark analysis of the unified 311 dataset.
Run from the Project folder:  python -u phase2/analysis.py | tee phase2/output/analysis_output.txt
One analysis only:            python -u phase2/analysis.py A1      (A1, A2, B1, B2)

Input
    processed/unified.parquet          (written by phase1/process.py)
Outputs
    phase2/output/analysis/*.csv       one result table per analysis
    phase2/output/plots/*.png          one figure per analysis

Analyses
    Weekly request volume          requests per within-month period (days 1-7, 8-14, 15-21, 22-end) per city; highest / lowest period
    Service-request hotspots       top 10 postal codes per city; top 3 request types in each postal code, with counts and % of the postal code's requests.
    Resolution performance         average / median resolution time per city; 5 request types with the longest average resolution time per city
    Status and channels            count and % per standardized status and channel per city; most common status and channel per city

Main PySpark operations: withColumn + when (derived labels), groupBy + agg (counts, avg,
percentile), Window + row_number (top-N per group, highest / lowest), join (attach totals),
where / filter (valid rows only).
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")                 # save figures to files, never open a window
import matplotlib.pyplot as plt
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED_PATH = os.path.join(BASE, "processed", "unified.parquet")
OUT_DIR = os.path.join(BASE, "phase2", "output", "analysis")
PLOT_DIR = os.path.join(BASE, "phase2", "output", "plots")

CITIES = ["NYC", "LA"]
COLORS = {"NYC": "#2a6fdb", "LA": "#e8833a"}
TOP_ZIPS = 10                  # postal codes per city
TOP_TYPES_PER_ZIP = 3          # request types per postal code
TOP_SLOW_TYPES = 5             # slowest request types per city
# Minimum number of valid resolutions a request type needs to be ranked in B1.
# 1 = every type is ranked (literal reading of the task). Raise it (e.g. 30) to ignore
# request types whose average rests on only a handful of requests.
MIN_TYPE_REQUESTS = 30

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (SparkSession.builder
         .appName("CS236-Analysis")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.session.timeZone", "UTC")      # same wall-clock times as Phase 1
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")


# =============================================================================
# HELPERS
# =============================================================================
def header(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def save_csv(df, name):
    """Write a (small) result DataFrame to phase2/output/analysis/<name>.csv and return its rows."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = df.collect()
    with open(os.path.join(OUT_DIR, f"{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(df.columns)
        w.writerows([list(r) for r in rows])
    print(f"  table saved: phase2/output/analysis/{name}.csv")
    return rows


def save_plot(fig, filename):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=130)
    plt.close(fig)
    print(f"  plot saved: phase2/output/plots/{filename}")


def pct(part, whole):
    """Percentage column rounded to 2 decimals."""
    return F.round(100.0 * part / whole, 2)


# =============================================================================
# WEEKLY REQUEST VOLUME
# =============================================================================
def weekly_volume(uni):
    """Operations:
    1. withColumn + when: label every request with its within-month period from
       dayofmonth(created_at): P1 days 1-7, P2 8-14, P3 15-21, P4 22-end of month.
    2. groupBy(source_city, period).agg(count): requests per period per city, summed over
       all months in the data. countDistinct(date) gives the calendar days covered, so a
       requests-per-day rate can be shown next to the raw count (P4 has 7-10 days).
    3. Window partitioned by city + row_number: rank periods by count (descending for the
       highest, ascending for the lowest).
    4. A month x period breakdown (groupBy month, period) shows the same counts per month."""


    # Divide each month into four periods.
    header("WEEKLY REQUEST VOLUME (within-month periods)")
    period = (F.when(F.dayofmonth("created_at") <= 7, "P1: days 1-7")
               .when(F.dayofmonth("created_at") <= 14, "P2: days 8-14")
               .when(F.dayofmonth("created_at") <= 21, "P3: days 15-21")
               .otherwise("P4: days 22-end"))
    df = uni.select("source_city", F.col("created_at"),
                    period.alias("period"), F.date_format("created_at", "yyyy-MM").alias("month"))

    # Count requests per period per city. Makes 8 groups (2 cities x 4 periods)
    city_total = Window.partitionBy("source_city")
    by_period = (df.groupBy("source_city", "period")
                   .agg(F.count("*").alias("requests"),
                        F.countDistinct(F.to_date("created_at")).alias("days_covered"))
                   .withColumn("pct_of_city", pct(F.col("requests"), F.sum("requests").over(city_total)))
                   .withColumn("requests_per_day", F.round(F.col("requests") / F.col("days_covered"), 1))
                   .orderBy(F.desc("source_city"), "period"))
    by_period.show(truncate=False)
    rows = save_csv(by_period, "a1_weekly_volume_by_period")

    # The same counts, month by month.
    by_month = (df.groupBy("source_city", "month", "period").count()
                  .groupBy("source_city", "month").pivot("period").sum("count")
                  .orderBy(F.desc("source_city"), "month"))
    print("Requests per month and period (partial first / last months have fewer days):")
    by_month.show(truncate=False)
    save_csv(by_month, "a1_weekly_volume_by_month")

    hi = Window.partitionBy("source_city").orderBy(F.desc("requests"), "period")
    lo = Window.partitionBy("source_city").orderBy(F.asc("requests"), "period")
    # Find the highest and lowest period per city.
    extremes = (by_period
                .withColumn("_hi", F.row_number().over(hi))
                .withColumn("_lo", F.row_number().over(lo))
                .where("_hi = 1 OR _lo = 1")
                .select("source_city",
                        F.when(F.col("_hi") == 1, "highest").otherwise("lowest").alias("rank"),
                        "period", "requests", "pct_of_city", "requests_per_day")
                .orderBy(F.desc("source_city"), "rank"))
    extremes.show(truncate=False)
    ext = save_csv(extremes, "a1_weekly_volume_highest_lowest")

    # Summarize the findings.
    print("Findings:")
    for city in CITIES:
        h = [r for r in ext if r["source_city"] == city and r["rank"] == "highest"][0]
        l = [r for r in ext if r["source_city"] == city and r["rank"] == "lowest"][0]
        print(f"  {city}: highest {h['period']} ({h['requests']:,} requests, {h['pct_of_city']}%), "
              f"lowest {l['period']} ({l['requests']:,}, {l['pct_of_city']}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, city in zip(axes, CITIES):
        r = [x for x in rows if x["source_city"] == city]
        bars = ax.bar([x["period"] for x in r], [x["requests"] for x in r], color=COLORS[city])
        for b, x in zip(bars, r):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{x['requests']:,}\n({x['days_covered']} days)", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{city}: requests per within-month period")
        ax.set_ylabel("requests")
        ax.set_ylim(0, max(x["requests"] for x in r) * 1.18)
    save_plot(fig, "a1_weekly_volume.png")


# =============================================================================
# SERVICE-REQUEST HOTSPOTS
# =============================================================================
def hotspots(uni):
    """Operations:
    1. where(postal_code IS NOT NULL): requests without a ZIP cannot belong to a hotspot.
    2. groupBy(source_city, postal_code).count + Window(city) row_number: the 10 postal
       codes with the most requests per city (ties: lower postal code first).
    3. join the top-10 list back to the requests, then groupBy(city, postal_code,
       request_type).count + Window(city, postal_code) row_number: the 3 most frequent
       request types within each hotspot.
    4. pct_of_zip = type count / all requests in that postal code x 100."""
    header("SERVICE-REQUEST HOTSPOTS (top postal codes and their top request types)")
    # Keep only requests that have a ZIP code.
    with_zip = uni.where(F.col("postal_code").isNotNull())

    # Top 10 ZIP codes per city.
    zip_counts = (with_zip.groupBy("source_city", "postal_code")
                          .agg(F.count("*").alias("zip_requests")))
    w_zip = Window.partitionBy("source_city").orderBy(F.desc("zip_requests"), "postal_code")
    city_tot = (uni.groupBy("source_city").agg(F.count("*").alias("city_requests")))
    top_zips = (zip_counts.withColumn("zip_rank", F.row_number().over(w_zip))
                          .where(F.col("zip_rank") <= TOP_ZIPS)
                          .join(city_tot, "source_city")
                          .withColumn("pct_of_city", pct(F.col("zip_requests"), F.col("city_requests")))
                          .drop("city_requests"))

    # Count request types inside those ZIPs.
    type_counts = (with_zip.join(top_zips.select("source_city", "postal_code"),
                                 ["source_city", "postal_code"])
                           .groupBy("source_city", "postal_code", "request_type")
                           .agg(F.count("*").alias("type_requests")))
    w_type = Window.partitionBy("source_city", "postal_code").orderBy(F.desc("type_requests"),
                                                                      "request_type")
    
    # Top 3 types per ZIP + count + % of the ZIP.
    result = (type_counts.withColumn("type_rank", F.row_number().over(w_type))
                         .where(F.col("type_rank") <= TOP_TYPES_PER_ZIP)
                         .join(top_zips, ["source_city", "postal_code"])
                         .withColumn("pct_of_zip", pct(F.col("type_requests"), F.col("zip_requests")))
                         .select("source_city", "zip_rank", "postal_code", "zip_requests", "pct_of_city",
                                 "type_rank", "request_type", "type_requests", "pct_of_zip")
                         .orderBy(F.desc("source_city"), "zip_rank", "type_rank"))
    result.show(TOP_ZIPS * TOP_TYPES_PER_ZIP * 2, truncate=False)
    rows = save_csv(result, "a2_hotspots_top_zip_request_types")

    print("Findings:")
    for city in CITIES:
        r = [x for x in rows if x["source_city"] == city]
        top = [x for x in r if x["type_rank"] == 1]
        share = sum(x["pct_of_city"] for x in top)
        print(f"  {city}: top 10 postal codes hold {share:.1f}% of the city's requests; "
              f"#1 is {top[0]['postal_code']} ({top[0]['zip_requests']:,} requests), "
              f"led by '{top[0]['request_type']}' ({top[0]['pct_of_zip']}% of that ZIP)")
        leaders = {}
        for x in top:
            leaders[x["request_type"]] = leaders.get(x["request_type"], 0) + 1
        print(f"        most frequent #1 type across its hotspots: " +
              ", ".join(f"{k} ({v} ZIPs)" for k, v in sorted(leaders.items(), key=lambda kv: -kv[1])))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, city in zip(axes, CITIES):
        top = sorted([x for x in rows if x["source_city"] == city and x["type_rank"] == 1],
                     key=lambda x: x["zip_rank"], reverse=True)
        ax.barh([x["postal_code"] for x in top], [x["zip_requests"] for x in top], color=COLORS[city])
        for i, x in enumerate(top):
            ax.text(x["zip_requests"], i, f"  {x['request_type']} {x['pct_of_zip']}%",
                    va="center", fontsize=7)
        ax.set_title(f"{city}: top {TOP_ZIPS} postal codes (label = #1 request type)")
        ax.set_xlabel("requests")
        ax.set_xlim(0, max(x["zip_requests"] for x in top) * 1.6)
    save_plot(fig, "a2_hotspots.png")


# =============================================================================
# RESOLUTION PERFORMANCE
# =============================================================================
def resolution_performance(uni):
    """Operations:
    1. where(resolution_valid): only valid completed requests (status Closed, closed date
       present, closed >= created; flag created in Phase 1).
    2. groupBy(source_city).agg(avg, percentile 0.5, count): average and exact median
       resolution time per city, in hours and days.
    3. groupBy(source_city, request_type).agg(...) + Window(city) row_number ordered by
       average descending: the 5 request types with the longest average resolution time."""
    header("B1 - RESOLUTION PERFORMANCE (bonus, valid completed requests only)")
    # Use only valid completed requests.
    valid = uni.where(F.col("resolution_valid"))

    # Average and median per city
    per_city = (valid.groupBy("source_city")
                     .agg(F.count("*").alias("valid_requests"),
                          F.round(F.avg("resolution_hours"), 2).alias("avg_hours"),
                          F.round(F.expr("percentile(resolution_hours, 0.5)"), 2).alias("median_hours"))
                     .withColumn("avg_days", F.round(F.col("avg_hours") / 24, 2))
                     .withColumn("median_days", F.round(F.col("median_hours") / 24, 2))
                     .orderBy(F.desc("source_city")))
    per_city.show(truncate=False)
    city_rows = save_csv(per_city, "b1_resolution_by_city")

    # Average per request type.
    by_type = (valid.groupBy("source_city", "request_type")
                    .agg(F.count("*").alias("valid_requests"),
                         F.round(F.avg("resolution_hours"), 2).alias("avg_hours"),
                         F.round(F.expr("percentile(resolution_hours, 0.5)"), 2).alias("median_hours"))
                    .where(F.col("valid_requests") >= MIN_TYPE_REQUESTS))
    w = Window.partitionBy("source_city").orderBy(F.desc("avg_hours"), "request_type")

    # Top 5 longest average per city.
    slowest = (by_type.withColumn("rank", F.row_number().over(w))
                      .where(F.col("rank") <= TOP_SLOW_TYPES)
                      .withColumn("avg_days", F.round(F.col("avg_hours") / 24, 1))
                      .select("source_city", "rank", "request_type", "valid_requests",
                              "avg_hours", "avg_days", "median_hours")
                      .orderBy(F.desc("source_city"), "rank"))
    print(f"{TOP_SLOW_TYPES} request types with the longest average resolution time "
          f"(types with >= {MIN_TYPE_REQUESTS} valid requests):")
    slowest.show(truncate=False)
    rows = save_csv(slowest, "b1_slowest_request_types")

    # Findings.
    print("Findings:")
    for r in city_rows:
        skew = "mean far above median (right-skewed)" if r["avg_hours"] > 2 * r["median_hours"] else ""
        print(f"  {r['source_city']}: average {r['avg_hours']} h ({r['avg_days']} days), median "
              f"{r['median_hours']} h ({r['median_days']} days) over {r['valid_requests']:,} requests  {skew}")
    for city in CITIES:
        s = [x for x in rows if x["source_city"] == city][0]
        print(f"  {city}: slowest type '{s['request_type']}' averages {s['avg_days']} days "
              f"({s['valid_requests']:,} valid requests)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, city in zip(axes, CITIES):
        r = sorted([x for x in rows if x["source_city"] == city], key=lambda x: x["rank"], reverse=True)
        ax.barh([x["request_type"] for x in r], [x["avg_days"] for x in r], color=COLORS[city])
        ax.set_title(f"{city}: {TOP_SLOW_TYPES} slowest request types (average)")
        ax.set_xlabel("average resolution time (days)")
        ax.tick_params(axis="y", labelsize=8)
    save_plot(fig, "b1_slowest_request_types.png")


# =============================================================================
# STATUS AND SUBMISSION CHANNELS
# =============================================================================
def status_and_channels(uni):
    """Operations:
    1. groupBy(source_city, status_std).count and groupBy(source_city, channel_std).count.
    2. Window(city) sum: % of the city's requests in each category.
    3. Window(city) row_number ordered by count: the most common category per city."""
    header("B2 - STANDARDIZED STATUS AND SUBMISSION CHANNEL (bonus)")

    # Windows for city totals and ranking.
    city_total = Window.partitionBy("source_city")
    top = Window.partitionBy("source_city").orderBy(F.desc("requests"), "category")

    # Most common status / channel per city.
    results, most_common = {}, {}
    for dim, col in [("status", "status_std"), ("channel", "channel_std")]:
        t = (uni.groupBy("source_city", F.col(col).alias("category"))
                .agg(F.count("*").alias("requests"))
                .withColumn("pct_of_city", pct(F.col("requests"), F.sum("requests").over(city_total)))
                .withColumn("rank", F.row_number().over(top))
                .orderBy(F.desc("source_city"), "rank"))
        print(f"\n{col} per city:")
        t.show(truncate=False)
        results[dim] = save_csv(t.select("source_city", "rank", "category", "requests", "pct_of_city"),
                                f"b2_{col}_by_city")
        most_common[dim] = {r["source_city"]: r for r in results[dim] if r["rank"] == 1}

    # Findings.
    print("Findings:")
    for city in CITIES:
        s, c = most_common["status"][city], most_common["channel"][city]
        print(f"  {city}: most common status '{s['category']}' ({s['pct_of_city']}%), "
              f"most common channel '{c['category']}' ({c['pct_of_city']}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dim in zip(axes, ["status", "channel"]):
        cats = sorted({r["category"] for r in results[dim]})
        width = 0.4
        for i, city in enumerate(CITIES):
            vals = {r["category"]: r["pct_of_city"] for r in results[dim] if r["source_city"] == city}
            ax.bar([j + (i - 0.5) * width for j in range(len(cats))], [vals.get(c, 0) for c in cats],
                   width, color=COLORS[city], label=city)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_ylabel("% of city's requests")
        ax.set_title(f"Standardized {dim}")
        ax.legend()
    save_plot(fig, "b2_status_and_channel.png")


# =============================================================================
# MAIN
# =============================================================================
# Load the unified dataset.
uni = spark.read.parquet(UNIFIED_PATH).persist(StorageLevel.MEMORY_AND_DISK)

# Sanity check of the input.
counts = {r["source_city"]: r["count"] for r in uni.groupBy("source_city").count().collect()}
header("UNIFIED DATASET")
print(f"  {UNIFIED_PATH}")
print(f"  rows: {sum(counts.values()):,}  (" + ", ".join(f"{k} {v:,}" for k, v in counts.items()) + ")")

# Choose which analyses to run.
PARTS = {p.upper() for p in sys.argv[1:]} or {"A1", "A2", "B1", "B2"}
if "A1" in PARTS:
    weekly_volume(uni)
if "A2" in PARTS:
    hotspots(uni)
if "B1" in PARTS:
    resolution_performance(uni)
if "B2" in PARTS:
    status_and_channels(uni)

spark.stop()

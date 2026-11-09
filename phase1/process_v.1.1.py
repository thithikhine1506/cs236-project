"""
process.py - CS236 Phase 1, section 1.3: clean, standardize and integrate NYC 311 and LA MyLA311 into one unified service-request dataset.

Inputs
    nyc_311_800k.csv
    la_311_800k.csv
Outputs
    processed/csv/nyc_clean.csv   cleaned NYC, all 44 original columns + derived
    processed/csv/la_clean.csv    cleaned LA,  all 34 original columns + derived
    processed/csv/unified.csv     both cities in one common schema

Pipeline
    1. clean_nyc / clean_la    Clean the datasets (types, placeholders, ZIPs, case variants)
    2. add_resolution          Create required attributes (resolution_hours, resolution_valid)
    3. save                    Persist cleaned datasets
    4. to_unified_nyc / _la    Adjust the schema + standardize values + add source_city,
                               source_id and request_id
    5. merge                   Merge both cities into one dataset
    6. validate                Validate counts, uniqueness, types and derived attributes
"""
import glob
import os
import shutil
import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "CS236-Fall26-Project-Dataset")
NYC_PATH = os.path.join(DATA, "nyc_311_800k.csv")
LA_PATH = os.path.join(DATA, "la_311_800k.csv")
PROCESSED = os.path.join(BASE, "processed")
TS_FMT = "yyyy-MM-dd'T'HH:mm:ss.SSS"   # e.g. 2026-10-05T02:05:08.000
SKIP_CSV = "--skip-csv" in sys.argv


os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

spark = (SparkSession.builder
         .appName("CS236-Process")
         .master("local[*]")
         .config("spark.driver.memory", "4g")
         .config("spark.sql.session.timeZone", "UTC")      # keep wall-clock times unchanged
         .config("spark.ui.showConsoleProgress", "false")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")


def header(title):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)



PLACEHOLDERS = ["unspecified", "n/a", "na", "unknown", "none", "null"]

# Columns where EDA found such placeholders. status and channel are NOT listed: their 'Unspecified' / 'UNKNOWN' stay as recorded and become 'Unknown' in the standardized columns.
NYC_PLACEHOLDER_COLS = ["park_facility_name", "park_borough", "borough", "descriptor",
                        "descriptor_2", "facility_type", "location_type",
                        "police_precinct", "resolution_description"]

# NYC categorical columns with spelling variants such as “PLUMBING” / “Plumbing”
NYC_CASE_COLS = ["complaint_type", "descriptor", "location_type", "city"]

# Label for LA mobileos when the request did not come from the mobile app.
MOBILEOS_NOT_APPLICABLE = "Not Applicable"

# Label for LA address when the request has no location ('Feedback' requests).
ADDRESS_NOT_APPLICABLE = "Not Applicable"

# LA suffix: a street word maps to a suffix (STREET → ST, AVENUE → AVE)
SUFFIX_FROM_ADDRESS = {"ST": "ST", "STREET": "ST", "AVE": "AVE", "AVENUE": "AVE"}

# Label for NYC descriptor_2 when that kind of request never has a third-level detail.
DESCRIPTOR2_NOT_APPLICABLE = "Not Applicable"

# Label for NYC park_facility_name when the request is not located in a park.
PARK_NOT_APPLICABLE = "Not Applicable"

# Label for NYC descriptor when that complaint type (for that agency) never has a sub-type.
DESCRIPTOR_NOT_APPLICABLE = "Not Applicable"

# LA council district -> council member during the data period (Jan-Jul 2024).
# Matches the name already recorded for each district.
CD_MEMBER = {
    1: "Eunisses Hernandez", 2: "Paul Krekorian", 3: "Bob Blumenfield",
    4: "Nithya Raman", 5: "Katy Yaroslavsky", 6: "Imelda Padilla",
    7: "Monica Rodriguez", 8: "Marqueece Harris-Dawson", 9: "Curren D. Price Jr.",
    10: "Heather Hutt", 11: "Traci Park", 12: "John Lee",
    13: "Hugo Soto-Martinez", 14: "Kevin de Le\u00f3n", 15: "Tim McOsker",
}

# Valid ZIP ranges (NYC 10/11xxx, LA 90–93xxx).
NYC_ZIP_PREFIX = r"^1[01]"     # 10xxx, 11xxx
LA_ZIP_PREFIX = r"^9[0-3]"     # 90xxx - 93xxx

# Standardized status. Original values are kept in the 'status' column.
STATUS_MAP = {
    "NYC": {"Open": "Open", "Assigned": "In Progress", "In Progress": "In Progress",
            "Started": "In Progress", "Pending": "Pending", "Closed": "Closed",
            "Unspecified": "Unknown"},
    "LA": {"Open": "Open", "Pending": "Pending", "Closed": "Closed",
           "Cancelled": "Cancelled", "Referred Out": "Referred", "Forward": "Referred"},
}
STATUS_STD_VALUES = ["Open", "In Progress", "Pending", "Closed", "Cancelled", "Referred", "Unknown"]

# Standardized submission channel. Original values are kept in the 'channel' column.
CHANNEL_MAP = {
    "NYC": {"PHONE": "Phone", "ONLINE": "Online", "MOBILE": "Mobile App",
            "OTHER": "Other", "UNKNOWN": "Unknown"},
    "LA": {"Call": "Phone", "Voicemail": "Phone", "Queue Initiated Customer Call": "Phone",
           "Mobile App": "Mobile App",
           "Self Service": "Online", "Web Form": "Online", "Social": "Online",
           "Email": "Email",
           "Driver Self Report": "City Staff", "Council's Office": "City Staff",
           "City Attorney": "City Staff",
           "Fax": "Other", "Letter": "Other", "Walk-in": "Other", "Radio": "Other"},
}
CHANNEL_STD_VALUES = ["Phone", "Online", "Mobile App", "Email", "City Staff", "Other", "Unknown"]

# Unified schema: column name -> Spark type (order = column order of the table).
# Design of the merged table: 23 column names and their types.
UNIFIED_SCHEMA = [
    ("request_id", "string"),         # 'NYC-<unique_key>' or 'LA-<srnumber>', primary key
    ("source_city", "string"),        # 'NYC' or 'LA'
    ("source_id", "string"),          # original identifier in the source dataset
    ("created_at", "timestamp"),
    ("closed_at", "timestamp"),
    ("updated_at", "timestamp"),
    ("status", "string"),             # original status
    ("status_std", "string"),         # standardized status
    ("channel", "string"),            # original submission channel
    ("channel_std", "string"),        # standardized submission channel
    ("agency", "string"),             # responsible agency code
    ("agency_name", "string"),        # full agency name (NYC only, NULL for LA)
    ("request_type", "string"),
    ("request_detail", "string"),     # NYC descriptor (NULL for LA)
    ("postal_code", "string"),
    ("address", "string"),
    ("district", "string"),           # NYC borough / LA area planning commission
    ("council_district", "int"),
    ("police_precinct", "string"),
    ("latitude", "double"),
    ("longitude", "double"),
    ("resolution_hours", "double"),
    ("resolution_valid", "boolean"),
]


# =============================================================================
# HELPERS
# =============================================================================

# Reads a CSV with every column as text
def load(path):
    """Read a raw CSV with every column as a string."""
    return (spark.read.option("header", True).option("inferSchema", False)
            .option("quote", '"').option("escape", '"').csv(path))

# Removes extra spaces; empty text → NULL
def trim_all(df):
    """Trim surrounding whitespace in every column and turn empty strings into NULL.
    It makes a single representation of 'missing' (NULL) for every column."""
    return df.select([F.when(F.trim(F.col(c)) == "", None).otherwise(F.trim(F.col(c))).alias(c)
                      for c in df.columns])

# “Unspecified”, “N/A” … → NULL
def placeholders_to_null(df, cols):
    """Replace placeholder text ('Unspecified', 'N/A', ...) with NULL in the given columns."""
    for c in cols:
        df = df.withColumn(c, F.when(F.lower(F.col(c)).isin(PLACEHOLDERS), None)
                              .otherwise(F.col(c)))
    return df

# Text → timestamp, int, double
def to_timestamps(df, cols):
    """Parse ISO-8601 text ('2026-10-05T02:05:08.000') into TIMESTAMP (local wall-clock time)."""
    for c in cols:
        df = df.withColumn(c, F.to_timestamp(F.col(c), TS_FMT))
    return df

# Text → timestamp, int, double
def cast_columns(df, types):
    """Cast columns to SQL types, e.g. {'latitude': 'double', 'council_district': 'int'}.
    Codes such as ZIP, BBL and house number stay STRING so leading zeros are kept."""
    for c, t in types.items():
        df = df.withColumn(c, F.col(c).cast(t))
    return df

# Keeps only valid 5-digit ZIPs in the city’s range
def clean_zip(col, prefix):
    """Keep a ZIP only if it is exactly 5 digits and starts with the city's prefix;
    anything else ('0', 'No', 'VE, 0', out-of-state codes) becomes NULL."""
    z = F.col(col)
    return F.when(z.rlike(r"^\d{5}$") & z.rlike(prefix), z)

# Merges spelling variants into the most common spelling
def unify_case_variants(df, col):
    """Replace case variants of the same value (e.g. 'PLUMBING' / 'Plumbing') with the
    most frequent spelling."""
    key = F.upper(F.regexp_replace(F.col(col), r"\s+", " "))
    counts = df.groupBy(key.alias("_k"), F.col(col).alias("_v")).count().where("_k IS NOT NULL")
    w = Window.partitionBy("_k").orderBy(F.desc("count"), F.asc("_v"))
    canon = (counts.withColumn("_r", F.row_number().over(w))
                   .withColumn("_n", F.count("*").over(Window.partitionBy("_k"))))
    merged_groups = canon.where("_r = 1 AND _n > 1").count()
    canon = canon.where("_r = 1").select("_k", F.col("_v").alias("_canon"))
    out = (df.withColumn("_k", key)
             .join(F.broadcast(canon), "_k", "left")
             .withColumn(col, F.coalesce(F.col("_canon"), F.col(col)))
             .drop("_k", "_canon"))
    return out, merged_groups


def add_resolution(df, created, closed, status):
    """Derived attributes:
       resolution_hours = (closed - created) in hours; NULL if no closed time or if
                          closed < created (impossible, so not reported as a duration)
       resolution_valid = TRUE only if status is 'Closed' AND resolution_hours exists."""
    hours = (F.unix_timestamp(F.col(closed)) - F.unix_timestamp(F.col(created))) / 3600.0
    return (df.withColumn("resolution_hours", F.when(hours >= 0, F.round(hours, 4)))
              .withColumn("resolution_valid",
                          (F.lower(F.col(status)) == "closed") & F.col("resolution_hours").isNotNull()))

# Translates raw categories into the standard vocabulary.
def zip_from_address(address_col):
    """LA ZIP code found inside an address, or NULL.
    Takes the LAST 5-digit number in the LA ZIP range (90xxx-93xxx). A ZIP+4 ('91423-3603') gives its first 5 digits."""
    zip5 = F.regexp_extract(address_col, r"^.*[^\d-](9[0-3]\d{3})(?:-\d{4})?(?!\d)", 1)
    return F.when(zip5 != "", zip5)


# Strip the ZIP from the address; detect intersection addresses (“X ST AT Y AVE”)
def la_street_part(address_col):
    """LA address without its trailing ', <zip>', upper-case and trimmed."""
    return F.upper(F.trim(F.regexp_replace(address_col, r",\s*\d{5}\s*$", "")))


# Strip the ZIP from the address; detect intersection addresses (“X ST AT Y AVE”)
def is_intersection(street_col):
    """Intersection addresses such as 'KLING ST AT VINELAND AVE', 'LOMITA BLVD / MCCOY AVE' or 'FAIRFAX AVE @ WASHINGTON AVE'. A slash only counts with spaces around it."""
    return street_col.rlike(r"\b(AT|AND)\b|&|@|\s/\s")


# Fills a missing suffix from the address’s last word (single streets only)
def fill_suffix_from_address(df):
    """Fill an empty LA 'suffix' from the last word of a single-street address. Returns (number of cells filled, df)."""
    street = la_street_part(F.col("address"))
    last_word = F.regexp_extract(street, r"(\S+)$", 1)
    derived = F.lit(None).cast("string")
    for word, suffix in SUFFIX_FROM_ADDRESS.items():
        derived = F.when(last_word == word, F.lit(suffix)).otherwise(derived)
    fill = F.col("suffix").isNull() & F.col("address").isNotNull() & ~is_intersection(street) & derived.isNotNull()
    n = df.where(fill).count()
    return n, df.withColumn("suffix", F.when(fill, derived).otherwise(F.col("suffix")))


# Requests not in a park → “Not Applicable”
def is_park_location(location_type_col):
    """True when location_type describes a park. 'Parking Lot' is not a park."""
    lt = F.lower(location_type_col)
    return lt.rlike(r"\bpark\b|playground|garden") & ~lt.rlike("parking")


# Requests not in a park → “Not Applicable”
def label_park_facility_name(df):
    """A real park name for only 3 records; all others were 'Unspecified' or 'N/A', i.e. no park name recorded.
    - Request NOT located in a park (location_type is not a park type, or location_type is
      unknown) -> 'Not Applicable'. 
      For unknown location_type this is an assumption: those requests come almost entirely from DOT and DOB (streets, buildings), not parks.
    - Request located in a park -> stays NULL: a park name exists but was not recorded.
    Returns (df, number labelled 'Not Applicable', number left NULL)."""
    in_park = F.coalesce(is_park_location(F.col("location_type")), F.lit(False))
    na = F.col("park_facility_name").isNull() & ~in_park
    out = df.withColumn("park_facility_name",
                        F.when(na, F.lit(PARK_NOT_APPLICABLE)).otherwise(F.col("park_facility_name")))
    r = out.agg(F.sum((F.col("park_facility_name") == PARK_NOT_APPLICABLE).cast("int")).alias("na"),
                F.sum(F.col("park_facility_name").isNull().cast("int")).alias("null")).collect()[0]
    return out, r["na"], r["null"]


# Fills a missing city from its ZIP code
def zip_city_lookup(df):
    """Each ZIP belongs to one postal city; when a ZIP was recorded with several names (e.g. NEW YORK / MANHATTAN, or
    RIDGEWOOD / QUEENS) the most frequent one is used."""
    counts = (df.where(F.col("incident_zip").isNotNull() & F.col("city").isNotNull())
                .groupBy("incident_zip", "city").count())
    w = Window.partitionBy("incident_zip").orderBy(F.desc("count"), F.asc("city"))
    return (counts.withColumn("_r", F.row_number().over(w)).where("_r = 1")
                  .select(F.col("incident_zip").alias("_zip"), F.col("city").alias("_zip_city")))


# Fills a missing city from its ZIP code
def fill_city_from_zip(df):
    """Fill an empty NYC 'city' with the postal city of its ZIP code (zip_city_lookup).
    Rows without a ZIP or with a ZIP never recorded with a city, stay NULL.
    Returns (df, number of cells filled)."""
    joined = df.join(F.broadcast(zip_city_lookup(df)), F.col("incident_zip") == F.col("_zip"), "left")
    fill = F.col("city").isNull() & F.col("_zip_city").isNotNull()
    n = joined.where(fill).count()
    out = (joined.withColumn("city", F.when(fill, F.col("_zip_city")).otherwise(F.col("city")))
                 .select(df.columns))
    return out, n


# Complaint types that never have a sub-type → “Not Applicable”
def descriptor_groups(df):
    """For every (complaint_type, agency): 'Not Applicable' does not count as a real value."""
    real = F.col("descriptor").isNotNull() & (F.col("descriptor") != DESCRIPTOR_NOT_APPLICABLE)
    return (df.groupBy(F.col("complaint_type").alias("_ct"), F.col("agency").alias("_ag"))
              .agg(F.max(real.cast("int")).alias("_group_uses_de")))


# Complaint types that never have a sub-type → “Not Applicable”
def label_descriptor(df):
    """A missing descriptor becomes 'Not Applicable' when its (complaint_type, agency) group NEVER has a
    real descriptor: that kind of request has no sub-type. The agency is part of the key because one complaint type (Asbestos) has sub-types at one
    agency and none at another. Otherwise the missing value stays NULL.
    Returns (df, number labelled 'Not Applicable', number left NULL)."""
    groups = descriptor_groups(df)
    joined = df.join(F.broadcast(groups),
                     F.col("complaint_type").eqNullSafe(F.col("_ct")) &
                     F.col("agency").eqNullSafe(F.col("_ag")), "left")
    na = F.col("descriptor").isNull() & (F.col("_group_uses_de") == 0)
    joined = joined.withColumn("descriptor",
                               F.when(na, F.lit(DESCRIPTOR_NOT_APPLICABLE)).otherwise(F.col("descriptor")))
    out = joined.select(df.columns)
    r = out.agg(F.sum((F.col("descriptor") == DESCRIPTOR_NOT_APPLICABLE).cast("int")).alias("na"),
                F.sum(F.col("descriptor").isNull().cast("int")).alias("null")).collect()[0]
    return out, r["na"], r["null"]


# The same rule for the third-level detail
def descriptor_2_pairs(df):
    """For every (complaint_type, descriptor) pair: does ANY row have a real descriptor_2?
    'Not Applicable' does not count as a real value. NULL descriptors form their own pair."""
    real = F.col("descriptor_2").isNotNull() & (F.col("descriptor_2") != DESCRIPTOR2_NOT_APPLICABLE)
    return (df.groupBy(F.col("complaint_type").alias("_ct"), F.col("descriptor").alias("_de"))
              .agg(F.max(real.cast("int")).alias("_pair_uses_d2")))


# The same rule for the third-level detail
def label_descriptor_2(df):
    """A missing descriptor_2 (NULL, including former 'N/A' placeholders) becomes
    'Not Applicable' when its (complaint_type, descriptor) pair never has a real descriptor_2
    anywhere in the data. When the pair sometimes has one, the missing value is genuinely unknown and stays NULL.
    Returns (df, number labelled 'Not Applicable', number left NULL)."""
    pairs = descriptor_2_pairs(df)
    joined = df.join(F.broadcast(pairs),
                     F.col("complaint_type").eqNullSafe(F.col("_ct")) &
                     F.col("descriptor").eqNullSafe(F.col("_de")), "left")
    na = F.col("descriptor_2").isNull() & (F.col("_pair_uses_d2") == 0)
    joined = joined.withColumn("descriptor_2",
                               F.when(na, F.lit(DESCRIPTOR2_NOT_APPLICABLE)).otherwise(F.col("descriptor_2")))
    out = joined.select(df.columns)                      # original column order, helpers dropped
    r = out.agg(F.sum((F.col("descriptor_2") == DESCRIPTOR2_NOT_APPLICABLE).cast("int")).alias("na"),
                F.sum(F.col("descriptor_2").isNull().cast("int")).alias("null")).collect()[0]
    return out, r["na"], r["null"]


# Translates raw categories into the standard vocabulary.
def map_values(col, mapping):
    """Translate raw category values into the standardized vocabulary. NULL -> 'Unknown'; a value not in the mapping -> 'Other' (reported by validate())."""
    expr = F.when(F.col(col).isNull(), F.lit("Unknown"))
    for raw, std in mapping.items():
        expr = expr.when(F.col(col) == raw, F.lit(std))
    return expr.otherwise(F.lit("Other"))



# STEP 1 and 2: CLEAN EACH SOURCE.

def clean_nyc(raw):
    """Clean NYC 311. Keeps all 44 original columns (same names) and adds
    resolution_hours / resolution_valid."""
    header("STEP 1 - CLEAN NYC 311")
    df = trim_all(raw)

    # cleaning log
    z = F.col("incident_zip")
    log = df.agg(
        *[F.sum(F.lower(F.col(c)).isin(PLACEHOLDERS).cast("int")).alias(c) for c in NYC_PLACEHOLDER_COLS],
        F.sum((z.isNotNull() & ~(z.rlike(r"^\d{5}$") & z.rlike(NYC_ZIP_PREFIX))).cast("int")).alias("bad_zip"),
    ).collect()[0]
    for c in NYC_PLACEHOLDER_COLS:
        print(f"  placeholders -> NULL   {c:<24} {log[c]:>8,}")
    print(f"  invalid ZIP -> NULL    {'incident_zip':<24} {log['bad_zip']:>8,}")

    df = placeholders_to_null(df, NYC_PLACEHOLDER_COLS)
    df = df.withColumn("incident_zip", clean_zip("incident_zip", NYC_ZIP_PREFIX))
    df = to_timestamps(df, ["created_date", "closed_date", "due_date",
                            "resolution_action_updated_date"])
    df = cast_columns(df, {"unique_key": "long", "council_district": "int",
                           "x_coordinate_state_plane": "int", "y_coordinate_state_plane": "int",
                           "latitude": "double", "longitude": "double"})
    for c in NYC_CASE_COLS:
        df, n = unify_case_variants(df, c)
        print(f"  case variants merged   {c:<24} {n:>8,} groups")
    # Materialize the cleaned rows once (on disk) before the two labelling steps. Each step groups and joins the whole dataset; 
    
    df = df.persist(StorageLevel.DISK_ONLY)
    df.count()
    df, n_city = fill_city_from_zip(df)
    print(f"  city filled from ZIP   {'city':<24} {n_city:>8,}")
    df, n_pna, n_pnull = label_park_facility_name(df)
    print(f"  park_facility_name     -> '{PARK_NOT_APPLICABLE}' {n_pna:>8,}   kept NULL {n_pnull:>6,}")
    df, n_dna, n_dnull = label_descriptor(df)
    print(f"  descriptor missing     -> '{DESCRIPTOR_NOT_APPLICABLE}' {n_dna:>8,}   kept NULL {n_dnull:>6,}")
    df, n_na, n_null = label_descriptor_2(df)
    print(f"  descriptor_2 missing   -> '{DESCRIPTOR2_NOT_APPLICABLE}' {n_na:>8,}   kept NULL {n_null:>6,}")
    df = add_resolution(df, "created_date", "closed_date", "status")
    return df


def clean_la(raw):
    """Clean LA MyLA311. Keeps all 34 original columns and adds resolution_hours / resolution_valid."""
    header("STEP 1 - CLEAN LA MyLA311")
    df = trim_all(raw)

    # ZIP: keep valid codes; recover a missing/invalid ZIP from the address. The house number is never used as a ZIP. 
    # ZIPs that cannot be recovered stay NULL (an invalid ZIP), so the column only ever holds 5-digit codes.
    z = F.col("zipcode")
    addr_zip = zip_from_address(F.col("address"))
    valid = z.rlike(r"^\d{5}$") & z.rlike(LA_ZIP_PREFIX)
    log = df.agg(
        F.sum((z.isNotNull() & ~valid).cast("int")).alias("bad_zip"),
        F.sum(((z.isNull() | ~valid) & addr_zip.isNotNull()).cast("int")).alias("recovered"),
    ).collect()[0]
    print(f"  invalid ZIP -> NULL    {'zipcode':<24} {log['bad_zip']:>8,}")
    print(f"  ZIP recovered from address                      {log['recovered']:>8,}")
    df = df.withColumn("zipcode", F.coalesce(clean_zip("zipcode", LA_ZIP_PREFIX), addr_zip))

    df = to_timestamps(df, ["createddate", "updateddate", "servicedate", "closeddate"])
    df = cast_columns(df, {"latitude": "double", "longitude": "double", "cd": "int",
                           "nc": "int", "tbmpage": "int", "tbmrow": "int"})
    
    # cdmember: fill an empty council-member name from the council district number.
    # Put NULL because the district, and therefore the member, is unknown.
    member = F.lit(None).cast("string")
    for cd, name in CD_MEMBER.items():
        member = F.when(F.col("cd") == cd, F.lit(name)).otherwise(member)
    fill_member = F.col("cdmember").isNull() & member.isNotNull()
    n_member = df.where(fill_member).count()
    df = df.withColumn("cdmember", F.when(fill_member, member).otherwise(F.col("cdmember")))
    print(f"  cdmember filled from cd  {'cdmember':<20} {n_member:>8,}")

    # Y/N flags -> boolean. addressverified also uses 'M', so it stays text.
    for c in ["anonymous", "approximateaddress"]:
        df = df.withColumn(c, F.when(F.col(c) == "Y", True).when(F.col(c) == "N", False))
    
    # mobileos: NULL is structural, not missing. Every NULL comes from a non-app source, where a mobile OS does not apply. 
    # These NULLs are labelled 'Not Applicable'. A NULL on a 'Mobile App' request would be unknown, so it stays NULL.
    na_os = F.col("mobileos").isNull() & (F.col("requestsource") != "Mobile App")
    n_na = df.where(na_os).count()
    df = df.withColumn("mobileos", F.when(na_os, F.lit(MOBILEOS_NOT_APPLICABLE))
                                    .otherwise(F.col("mobileos")))
    print(f"  NULL -> '{MOBILEOS_NOT_APPLICABLE}'  {'mobileos':<20} {n_na:>8,}")

    # suffix: fill a missing suffix from the address, but only when the address is a single street whose LAST word is ST/STREET (-> 'ST') or AVE/AVENUE (-> 'AVE').
    # Not filled: intersections have two streets and no single suffix;
    n_suffix, df = fill_suffix_from_address(df)
    print(f"  suffix filled from address {'suffix':<17} {n_suffix:>8,}")

    # address: every empty address belongs to a 'Feedback' request, and those rows have no coordinates, ZIP, street or district either, so a location does not apply. They are labelled 'Not Applicable'. 
    # An empty address on any other request type would be missing, so it stays NULL. This runs after the ZIP recovery above, which reads the original address text.
    na_addr = F.col("address").isNull() & (F.col("requesttype") == "Feedback")
    n_addr = df.where(na_addr).count()
    df = df.withColumn("address", F.when(na_addr, F.lit(ADDRESS_NOT_APPLICABLE))
                                   .otherwise(F.col("address")))
    print(f"  NULL -> '{ADDRESS_NOT_APPLICABLE}'  {'address':<20} {n_addr:>8,}")

    df = add_resolution(df, "createddate", "closeddate", "status")
    print("  types cast, Y/N flags -> boolean, resolution attributes added")
    return df



# STEP 3: Store the cleaned data in two formats.
# save() writes Parquet, which keeps the column types, and write_single_csv() also writes a CSV.
def write_single_csv(df, name):
    """Write one CSV file."""
    tmp = os.path.join(PROCESSED, "csv", f"_tmp_{name}")
    (df.coalesce(1).write.mode("overwrite")
       .option("header", True).option("escape", '"')
       .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
       .csv(tmp))
    part = glob.glob(os.path.join(tmp, "part-*.csv"))[0]
    final = os.path.join(PROCESSED, "csv", f"{name}.csv")
    if os.path.exists(final):
        os.remove(final)
    shutil.move(part, final)
    shutil.rmtree(tmp, ignore_errors=True)
    return final


def save(df, name):
    """Persist a dataset as Parquet (keeps the column types)."""
    path = os.path.join(PROCESSED, f"{name}.parquet")
    df.write.mode("overwrite").parquet(path)
    saved = spark.read.parquet(path)
    print(f"  saved processed/{name}.parquet")
    if not SKIP_CSV:
        write_single_csv(saved, name)
        print(f"  saved processed/csv/{name}.csv")
    return saved



# STEP 4-5: COMMON SCHEMA, STANDARDIZATION, MERGE

def align(df):
    """Order and cast the columns exactly as UNIFIED_SCHEMA, so both cities line up."""
    return df.select([F.col(c).cast(t).alias(c) for c, t in UNIFIED_SCHEMA])


# rename NYC city’s columns to the shared names
def to_unified_nyc(nyc):
    """Map cleaned NYC columns onto the unified schema (EDA Part 4.1 mapping)."""
    return align(nyc.select(
        F.concat(F.lit("NYC-"), F.col("unique_key").cast("string")).alias("request_id"),
        F.lit("NYC").alias("source_city"),
        F.col("unique_key").cast("string").alias("source_id"),
        F.col("created_date").alias("created_at"),
        F.col("closed_date").alias("closed_at"),
        F.col("resolution_action_updated_date").alias("updated_at"),
        F.col("status"),
        map_values("status", STATUS_MAP["NYC"]).alias("status_std"),
        F.col("open_data_channel_type").alias("channel"),
        map_values("open_data_channel_type", CHANNEL_MAP["NYC"]).alias("channel_std"),
        F.col("agency"),
        F.col("agency_name"),
        F.col("complaint_type").alias("request_type"),
        # 'Not Applicable' becomes NULL here, consistent with LA.
        F.when(F.col("descriptor") == DESCRIPTOR_NOT_APPLICABLE, None)
         .otherwise(F.col("descriptor")).alias("request_detail"),
        F.col("incident_zip").alias("postal_code"),
        F.col("incident_address").alias("address"),
        F.col("borough").alias("district"),
        F.col("council_district"),
        F.col("police_precinct"),
        F.col("latitude"), F.col("longitude"),
        F.col("resolution_hours"), F.col("resolution_valid"),
    ))


# rename LA city’s columns to the shared names
def to_unified_la(la):
    """Map cleaned LA columns onto the unified schema (EDA Part 4.1 mapping).
    The address drops its trailing ', <zip>' so it has the same format as NYC's."""
    return align(la.select(
        F.concat(F.lit("LA-"), F.col("srnumber")).alias("request_id"),
        F.lit("LA").alias("source_city"),
        F.col("srnumber").alias("source_id"),
        F.col("createddate").alias("created_at"),
        F.col("closeddate").alias("closed_at"),
        F.col("updateddate").alias("updated_at"),
        F.col("status"),
        map_values("status", STATUS_MAP["LA"]).alias("status_std"),
        F.col("requestsource").alias("channel"),
        map_values("requestsource", CHANNEL_MAP["LA"]).alias("channel_std"),
        F.col("owner").alias("agency"),
        F.lit(None).alias("agency_name"),
        F.col("requesttype").alias("request_type"),
        F.lit(None).alias("request_detail"),
        F.col("zipcode").alias("postal_code"),
        # 'Not Applicable' becomes NULL again here, so the unified address column means the same thing for both cities.
        F.when(F.col("address") == ADDRESS_NOT_APPLICABLE, None)
         .otherwise(F.regexp_replace(F.col("address"), r",\s*\d{5}\s*$", "")).alias("address"),
        F.col("apc").alias("district"),
        F.col("cd").alias("council_district"),
        F.col("policeprecinct").alias("police_precinct"),
        F.col("latitude"), F.col("longitude"),
        F.col("resolution_hours"), F.col("resolution_valid"),
    ))


# Stacks the two cities by column name (unionByName).
def merge(nyc_u, la_u):
    """Union the two aligned datasets by column name into one unified dataset."""
    header("STEP 5 - MERGE")
    unified = nyc_u.unionByName(la_u)
    print(f"  unified columns: {len(unified.columns)}")
    return unified



# STEP 6: VALIDATE UNIFIED SCHEMA.

def validate(raw_counts, nyc, la, uni):
    """Check that nothing was lost or duplicated and that derived values are valid."""
    header("STEP 6 - VALIDATION")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    n_nyc, n_la, n_uni = nyc.count(), la.count(), uni.count()
    by_city = {r["source_city"]: r["count"] for r in uni.groupBy("source_city").count().collect()}
    check("cleaned NYC keeps every row", n_nyc == raw_counts["nyc"], f"{n_nyc:,} / {raw_counts['nyc']:,}")
    check("cleaned LA keeps every row", n_la == raw_counts["la"], f"{n_la:,} / {raw_counts['la']:,}")
    check("unified = NYC + LA", n_uni == n_nyc + n_la and by_city.get("NYC") == n_nyc
          and by_city.get("LA") == n_la, f"{n_uni:,} = {by_city.get('NYC', 0):,} + {by_city.get('LA', 0):,}")
    check("cleaned NYC keeps all columns", len(nyc.columns) == 44 + 2, f"{len(nyc.columns)} (44 + 2 derived)")
    check("cleaned LA keeps all columns", len(la.columns) == 34 + 2, f"{len(la.columns)} (34 + 2 derived)")
    in_park = F.coalesce(is_park_location(F.col("location_type")), F.lit(False))
    pk = nyc.agg(
        F.sum((F.col("park_facility_name").isNull() & ~in_park).cast("int")).alias("bad_null"),
        F.sum(((F.col("park_facility_name") == PARK_NOT_APPLICABLE) & in_park).cast("int")).alias("bad_na"),
        F.sum((F.col("park_facility_name") == PARK_NOT_APPLICABLE).cast("int")).alias("na"),
        F.sum(F.col("park_facility_name").isNull().cast("int")).alias("null")).collect()[0]
    check("NYC park_facility_name: 'Not Applicable' outside parks, NULL only in parks",
          pk["bad_null"] == 0 and pk["bad_na"] == 0,
          f"Not Applicable={pk['na']:,}, NULL={pk['null']:,}")
    cz = (nyc.join(F.broadcast(zip_city_lookup(nyc)), F.col("incident_zip") == F.col("_zip"), "left")
             .agg(F.sum((F.col("city").isNull() & F.col("_zip_city").isNotNull()).cast("int")).alias("left"),
                  F.sum(F.col("city").isNull().cast("int")).alias("null")).collect()[0])
    check("NYC city: every row with a known ZIP has a city", cz["left"] == 0,
          f"{cz['left']} left; city still NULL={cz['null']:,} (no ZIP)")
    de = (nyc.join(F.broadcast(descriptor_groups(nyc)),
                   F.col("complaint_type").eqNullSafe(F.col("_ct")) &
                   F.col("agency").eqNullSafe(F.col("_ag")), "left")
             .agg(F.sum((F.col("descriptor").isNull() & (F.col("_group_uses_de") == 0)).cast("int")).alias("bad_null"),
                  F.sum(((F.col("descriptor") == DESCRIPTOR_NOT_APPLICABLE) & (F.col("_group_uses_de") == 1))
                        .cast("int")).alias("bad_na"),
                  F.sum((F.col("descriptor") == DESCRIPTOR_NOT_APPLICABLE).cast("int")).alias("na"),
                  F.sum(F.col("descriptor").isNull().cast("int")).alias("null")).collect()[0])
    check("NYC descriptor: 'Not Applicable' only where the complaint type never has a sub-type",
          de["bad_null"] == 0 and de["bad_na"] == 0,
          f"Not Applicable={de['na']:,}, NULL={de['null']:,}")
    d2 = (nyc.join(F.broadcast(descriptor_2_pairs(nyc)),
                   F.col("complaint_type").eqNullSafe(F.col("_ct")) &
                   F.col("descriptor").eqNullSafe(F.col("_de")), "left")
             .agg(F.sum((F.col("descriptor_2").isNull() & (F.col("_pair_uses_d2") == 0)).cast("int")).alias("bad_null"),
                  F.sum(((F.col("descriptor_2") == DESCRIPTOR2_NOT_APPLICABLE) & (F.col("_pair_uses_d2") == 1))
                        .cast("int")).alias("bad_na"),
                  F.sum((F.col("descriptor_2") == DESCRIPTOR2_NOT_APPLICABLE).cast("int")).alias("na"),
                  F.sum(F.col("descriptor_2").isNull().cast("int")).alias("null")).collect()[0])
    check("NYC descriptor_2: 'Not Applicable' only where the request type never has one",
          d2["bad_null"] == 0 and d2["bad_na"] == 0,
          f"Not Applicable={d2['na']:,}, NULL={d2['null']:,}")
    os_counts = {r["mobileos"]: r["count"] for r in la.groupBy("mobileos").count().collect()}
    check("LA mobileos has no NULL (non-app rows = 'Not Applicable')",
          os_counts.get(None, 0) == 0,
          ", ".join(f"{k}={v:,}" for k, v in sorted(os_counts.items(), key=lambda x: -x[1])))
    a = la.agg(F.sum(F.col("address").isNull().cast("int")).alias("null"),
               F.sum((F.col("address") == ADDRESS_NOT_APPLICABLE).cast("int")).alias("na")).collect()[0]
    check("LA address has no NULL (Feedback rows = 'Not Applicable')", a["null"] == 0,
          f"NULL={a['null']:,}, Not Applicable={a['na']:,}")
    zr = la.agg(F.sum(F.col("zipcode").isNull().cast("int")).alias("null"),
                F.sum((F.col("zipcode").isNull() & zip_from_address(F.col("address")).isNotNull())
                      .cast("int")).alias("left")).collect()[0]
    check("LA zipcode: no ZIP left unrecovered in the address", zr["left"] == 0,
          f"{zr['left']} left; zipcode still NULL={zr['null']:,} (no ZIP in address / no address)")
    cm = la.agg(
        F.sum((F.col("cdmember").isNull() & F.col("cd").isNotNull()).cast("int")).alias("left"),
        F.sum(F.col("cdmember").isNull().cast("int")).alias("null")).collect()[0]
    check("LA cdmember: every row with a council district has a member", cm["left"] == 0,
          f"{cm['left']} left; cdmember still NULL={cm['null']:,} (no council district)")
    street = la_street_part(F.col("address"))
    last_word = F.regexp_extract(street, r"(\S+)$", 1)
    s = la.agg(
        F.sum(F.col("suffix").isNull().cast("int")).alias("null"),
        F.sum((F.col("suffix").isNull() & ~is_intersection(street)
               & last_word.isin(list(SUFFIX_FROM_ADDRESS))).cast("int")).alias("left")).collect()[0]
    check("LA suffix: no single-street ST/AVE address left without a suffix", s["left"] == 0,
          f"{s['left']} left; suffix still NULL={s['null']:,} (intersections / no suffix)")

    distinct_ids = uni.select("request_id").distinct().count()
    null_ids = uni.where(F.col("request_id").isNull()).count()
    check("request_id unique and not NULL", distinct_ids == n_uni and null_ids == 0,
          f"{distinct_ids:,} distinct, {null_ids} NULL")
    check("unified schema matches the design",
          [(f.name, f.dataType.simpleString()) for f in uni.schema.fields] == UNIFIED_SCHEMA)

    r = uni.agg(
        F.sum(F.col("created_at").isNull().cast("int")).alias("no_created"),
        F.sum((F.col("resolution_hours") < 0).cast("int")).alias("neg_hours"),
        F.sum((F.col("resolution_valid") & F.col("resolution_hours").isNull()).cast("int")).alias("valid_no_hours"),
        F.sum((F.col("resolution_valid") & (F.col("status") != "Closed")).cast("int")).alias("valid_not_closed"),
        F.sum((F.col("postal_code").isNotNull() & ~F.col("postal_code").rlike(r"^\d{5}$")).cast("int")).alias("bad_zip"),
        F.sum((~F.col("status_std").isin(STATUS_STD_VALUES)).cast("int")).alias("bad_status"),
        F.sum((~F.col("channel_std").isin(CHANNEL_STD_VALUES)).cast("int")).alias("bad_channel"),
    ).collect()[0]
    check("created_at present in every row", r["no_created"] == 0, f"{r['no_created']} missing")
    check("no negative resolution_hours", r["neg_hours"] == 0, f"{r['neg_hours']} negative")
    check("resolution_valid only for Closed rows with hours",
          r["valid_no_hours"] == 0 and r["valid_not_closed"] == 0)
    check("postal_code is 5 digits or NULL", r["bad_zip"] == 0, f"{r['bad_zip']} invalid")
    check("status_std / channel_std use the standard vocabularies",
          r["bad_status"] == 0 and r["bad_channel"] == 0)

    # categories that fell through to 'Other' because the mapping did not know them
    unmapped = (uni.where((F.col("channel_std") == "Other") &
                          ~F.col("channel").isin(["OTHER", "Fax", "Letter", "Walk-in", "Radio"]))
                   .select("source_city", "channel").distinct().collect())
    check("every raw channel value is mapped", len(unmapped) == 0,
          ", ".join(f"{u[0]}:{u[1]}" for u in unmapped))

    print("\n  Standardized status per city (raw -> standardized):")
    (uni.groupBy("source_city", "status", "status_std").count()
        .orderBy("source_city", F.desc("count")).show(30, truncate=False))
    print("  Standardized channel per city (raw -> standardized):")
    (uni.groupBy("source_city", "channel", "channel_std").count()
        .orderBy("source_city", F.desc("count")).show(30, truncate=False))
    print("  Resolution attributes per city:")
    (uni.groupBy("source_city")
        .agg(F.count("*").alias("rows"),
             F.sum(F.col("resolution_valid").cast("int")).alias("valid_resolutions"),
             F.round(F.expr("percentile_approx(CASE WHEN resolution_valid THEN resolution_hours END, 0.5)"), 2)
              .alias("median_hours"),
             F.round(F.max(F.when(F.col("resolution_valid"), F.col("resolution_hours"))), 1).alias("max_hours"))
        .orderBy("source_city").show(truncate=False))
    print("  Unified schema:")
    uni.printSchema()
    print("  Sample unified rows:")
    uni.where(F.col("source_city") == "NYC").show(2, vertical=True, truncate=40)
    uni.where(F.col("source_city") == "LA").show(2, vertical=True, truncate=40)

    print(f"\n  {sum(results)} of {len(results)} checks passed")
    return all(results)



# MAIN
# Load and cache both raw files → clean and save NYC → clean and save LA → unify, merge and save → validate → stop Spark.
os.makedirs(PROCESSED, exist_ok=True)
raw_nyc = load(NYC_PATH).persist(StorageLevel.MEMORY_AND_DISK)
raw_la = load(LA_PATH).persist(StorageLevel.MEMORY_AND_DISK)
raw_counts = {"nyc": raw_nyc.count(), "la": raw_la.count()}

nyc_clean = save(clean_nyc(raw_nyc), "nyc_clean")
la_clean = save(clean_la(raw_la), "la_clean")
raw_nyc.unpersist()
raw_la.unpersist()

header("STEP 4 - COMMON SCHEMA + STANDARDIZATION")
print(f"  {len(UNIFIED_SCHEMA)} unified columns: " + ", ".join(c for c, _ in UNIFIED_SCHEMA))
unified = save(merge(to_unified_nyc(nyc_clean), to_unified_la(la_clean)), "unified")

ok = validate(raw_counts, nyc_clean, la_clean, unified)
spark.stop()
sys.exit(0 if ok else 1)
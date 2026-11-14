

-- schema.sql - CS236 Phase 2, section 2.2: PostgreSQL tables for the three processed datasets.
-- Executed by phase2/load_db.py before loading (tables are recreated on every run).

-- Thi Thi Khine - tkhine002
-- Scott Gonzalez Barrios - sgonz266


-- Design choices
-- nyc_requests
-- Primary key: unique_key (BIGINT).
-- NOT NULL: created_date, agency, complaint_type, status, open_data_channel_type and resolution_valid
-- CHECKs: the ZIP is 5 digits, and resolution_hours >= 0

-- la_requests
-- Primary key: srnumber, a VARCHAR because it contains a hyphen.
-- NOT NULL: createddate, owner, requesttype, status, requestsource and resolution_valid.
-- CHECKs: the ZIP is 5 digits, addressverified IN ('Y','N','M'), and resolution_hours >= 0
-- anonymous and approximateaddress are BOOLEAN; cd, nc, tbmpage and tbmrow are INTEGER.

-- unified_requests
-- Keys: request_id is the primary key, plus UNIQUE (source_city, source_id), which links each row back to its source table.
-- CHECKs: source_city IN ('NYC','LA');status_std and channel_std must be one of their 7 standard values; the ZIP is 5 digits;resolution_hours >= 0.
-- agency_name and request_detail are nullable, because they’re NULL for every LA row.


-- Removes old tables, so every run starts clean

DROP TABLE IF EXISTS unified_requests;
DROP TABLE IF EXISTS nyc_requests;
DROP TABLE IF EXISTS la_requests;


-- Cleaned NYC: 46 columns (44 original + 2 derived), primary key unique_key

CREATE TABLE nyc_requests (
    unique_key                     BIGINT           PRIMARY KEY,
    created_date                   TIMESTAMP        NOT NULL,
    closed_date                    TIMESTAMP,
    agency                         TEXT             NOT NULL,
    agency_name                    TEXT,
    complaint_type                 TEXT             NOT NULL,
    descriptor                     TEXT,
    descriptor_2                   TEXT,
    location_type                  TEXT,
    incident_zip                   VARCHAR(5)       CHECK (incident_zip ~ '^[0-9]{5}$'),
    incident_address               TEXT,
    street_name                    TEXT,
    cross_street_1                 TEXT,
    cross_street_2                 TEXT,
    intersection_street_1          TEXT,
    intersection_street_2          TEXT,
    address_type                   TEXT,
    city                           TEXT,
    landmark                       TEXT,
    facility_type                  TEXT,
    status                         TEXT             NOT NULL,
    due_date                       TIMESTAMP,
    resolution_description         TEXT,
    resolution_action_updated_date TIMESTAMP,
    community_board                TEXT,
    council_district               INTEGER,
    police_precinct                TEXT,
    bbl                            TEXT,
    borough                        TEXT,
    x_coordinate_state_plane       INTEGER,
    y_coordinate_state_plane       INTEGER,
    open_data_channel_type         TEXT             NOT NULL,
    park_facility_name             TEXT,
    park_borough                   TEXT,
    vehicle_type                   TEXT,
    taxi_company_borough           TEXT,
    taxi_pick_up_location          TEXT,
    bridge_highway_name            TEXT,
    bridge_highway_direction       TEXT,
    road_ramp                      TEXT,
    bridge_highway_segment         TEXT,
    latitude                       DOUBLE PRECISION,
    longitude                      DOUBLE PRECISION,
    location                       TEXT,
    resolution_hours               DOUBLE PRECISION CHECK (resolution_hours >= 0),
    resolution_valid               BOOLEAN          NOT NULL
);


-- Cleaned LA: 36 columns (34 + 2), primary key srnumber

CREATE TABLE la_requests (
    srnumber                  VARCHAR(20)      PRIMARY KEY,
    createddate               TIMESTAMP        NOT NULL,
    updateddate               TIMESTAMP,
    actiontaken               TEXT,
    owner                     TEXT             NOT NULL,
    requesttype               TEXT             NOT NULL,
    status                    TEXT             NOT NULL,
    requestsource             TEXT             NOT NULL,
    createdbyuserorganization TEXT,
    mobileos                  TEXT,
    anonymous                 BOOLEAN,
    assignto                  TEXT,
    servicedate               TIMESTAMP,
    closeddate                TIMESTAMP,
    addressverified           VARCHAR(1)       CHECK (addressverified IN ('Y', 'N', 'M')),
    approximateaddress        BOOLEAN,
    address                   TEXT,
    housenumber               TEXT,
    direction                 TEXT,
    streetname                TEXT,
    suffix                    TEXT,
    zipcode                   VARCHAR(5)       CHECK (zipcode ~ '^[0-9]{5}$'),
    latitude                  DOUBLE PRECISION,
    longitude                 DOUBLE PRECISION,
    location                  TEXT,
    tbmpage                   INTEGER,
    tbmcolumn                 TEXT,
    tbmrow                    INTEGER,
    apc                       TEXT,
    cd                        INTEGER,
    cdmember                  TEXT,
    nc                        INTEGER,
    ncname                    TEXT,
    policeprecinct            TEXT,
    resolution_hours          DOUBLE PRECISION CHECK (resolution_hours >= 0),
    resolution_valid          BOOLEAN          NOT NULL
);


-- Unified: 23 columns, primary key request_id + UNIQUE (source_city, source_id)

CREATE TABLE unified_requests (
    request_id       VARCHAR(30)      PRIMARY KEY,                  -- 'NYC-<unique_key>' / 'LA-<srnumber>'
    source_city      VARCHAR(3)       NOT NULL CHECK (source_city IN ('NYC', 'LA')),
    source_id        VARCHAR(20)      NOT NULL,                     -- id in nyc_requests / la_requests
    created_at       TIMESTAMP        NOT NULL,
    closed_at        TIMESTAMP,
    updated_at       TIMESTAMP,
    status           TEXT             NOT NULL,
    status_std       VARCHAR(20)      NOT NULL CHECK (status_std IN
                         ('Open', 'In Progress', 'Pending', 'Closed', 'Cancelled', 'Referred', 'Unknown')),
    channel          TEXT             NOT NULL,
    channel_std      VARCHAR(20)      NOT NULL CHECK (channel_std IN
                         ('Phone', 'Online', 'Mobile App', 'Email', 'City Staff', 'Other', 'Unknown')),
    agency           TEXT             NOT NULL,
    agency_name      TEXT,                                          -- NULL for LA
    request_type     TEXT             NOT NULL,
    request_detail   TEXT,                                          -- NULL for LA
    postal_code      VARCHAR(5)       CHECK (postal_code ~ '^[0-9]{5}$'),
    address          TEXT,
    district         TEXT,
    council_district INTEGER,
    police_precinct  TEXT,
    latitude         DOUBLE PRECISION,
    longitude        DOUBLE PRECISION,
    resolution_hours DOUBLE PRECISION CHECK (resolution_hours >= 0),
    resolution_valid BOOLEAN          NOT NULL,
    CONSTRAINT unified_source_unique UNIQUE (source_city, source_id)
);

-- Indexes for the Phase 2 queries (filter by city, time, postal code, type).
CREATE INDEX unified_city_created_idx ON unified_requests (source_city, created_at);
CREATE INDEX unified_city_postal_idx  ON unified_requests (source_city, postal_code);
CREATE INDEX unified_city_type_idx    ON unified_requests (source_city, request_type);

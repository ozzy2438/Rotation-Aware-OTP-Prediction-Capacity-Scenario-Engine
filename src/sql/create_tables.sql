-- Spirit Airlines FLL Hub: DuckDB Schema
-- Tables are pre-populated by ETL pipeline from processed DataFrames.
-- This script creates views and indexes on top of the base tables.

-- ============================================================
-- VIEWS
-- ============================================================

-- Route Performance: aggregated OTP, delay stats per route per month
CREATE OR REPLACE VIEW route_performance AS
SELECT
    Route,
    Origin,
    Dest,
    Year,
    Month,
    COUNT(*)                                                    AS total_flights,
    SUM(Cancelled)                                              AS cancelled_flights,
    ROUND(SUM(Cancelled)::DOUBLE / COUNT(*), 4)                 AS cancel_rate,
    SUM(CASE WHEN Cancelled = 0 THEN 1 ELSE 0 END)              AS operated_flights,
    SUM(CASE WHEN ArrDel15 = 1 AND Cancelled = 0 THEN 1 ELSE 0 END) AS delayed_flights,
    ROUND(
        SUM(CASE WHEN ArrDel15 = 0 AND Cancelled = 0 THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(SUM(CASE WHEN Cancelled = 0 THEN 1 ELSE 0 END), 0),
        4
    )                                                           AS otp_rate,
    ROUND(AVG(CASE WHEN Cancelled = 0 THEN ArrDelay END), 2)    AS avg_arr_delay_min,
    ROUND(AVG(CASE WHEN Cancelled = 0 THEN DepDelay END), 2)    AS avg_dep_delay_min,
    ROUND(AVG(CASE WHEN Cancelled = 0 THEN CarrierDelay END), 2) AS avg_carrier_delay,
    ROUND(AVG(CASE WHEN Cancelled = 0 THEN WeatherDelay END), 2) AS avg_weather_delay,
    ROUND(AVG(CASE WHEN Cancelled = 0 THEN NASDelay END), 2)    AS avg_nas_delay,
    ROUND(AVG(CASE WHEN Cancelled = 0 THEN LateAircraftDelay END), 2) AS avg_late_ac_delay,
    Distance
FROM flights
GROUP BY Route, Origin, Dest, Year, Month, Distance;


-- Monthly OTP Summary: network-wide monthly view
CREATE OR REPLACE VIEW monthly_otp_summary AS
SELECT
    Year,
    Month,
    PRINTF('%d-%02d', Year, Month)                              AS YearMonth,
    COUNT(*)                                                    AS total_flights,
    SUM(Cancelled)                                              AS cancelled_flights,
    ROUND(SUM(Cancelled)::DOUBLE / COUNT(*), 4)                 AS cancel_rate,
    ROUND(
        SUM(CASE WHEN ArrDel15 = 0 AND Cancelled = 0 THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(SUM(CASE WHEN Cancelled = 0 THEN 1 ELSE 0 END), 0),
        4
    )                                                           AS network_otp,
    ROUND(AVG(CASE WHEN Cancelled = 0 THEN ArrDelay END), 2)   AS avg_arr_delay,
    ROUND(AVG(CASE WHEN Cancelled = 0 THEN DepDelay END), 2)   AS avg_dep_delay,
    COUNT(DISTINCT Route)                                       AS routes_operated
FROM flights
GROUP BY Year, Month
ORDER BY Year, Month;


-- Delay Cause Decomposition: fraction of delay attributable to each cause
CREATE OR REPLACE VIEW delay_cause_decomposition AS
SELECT
    Route,
    Year,
    Month,
    ROUND(AVG(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN CarrierDelay      ELSE NULL END), 2) AS avg_carrier_delay,
    ROUND(AVG(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN WeatherDelay      ELSE NULL END), 2) AS avg_weather_delay,
    ROUND(AVG(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN NASDelay          ELSE NULL END), 2) AS avg_nas_delay,
    ROUND(AVG(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN SecurityDelay     ELSE NULL END), 2) AS avg_security_delay,
    ROUND(AVG(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN LateAircraftDelay ELSE NULL END), 2) AS avg_late_ac_delay,
    SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN 1 ELSE 0 END)           AS delayed_count,
    ROUND(
        SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN CarrierDelay      ELSE 0 END)
        / NULLIF(
            SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN
                COALESCE(CarrierDelay,0) + COALESCE(WeatherDelay,0) +
                COALESCE(NASDelay,0) + COALESCE(SecurityDelay,0) +
                COALESCE(LateAircraftDelay,0) ELSE 0 END), 0),
        4
    )                                                                          AS carrier_delay_pct,
    ROUND(
        SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN WeatherDelay      ELSE 0 END)
        / NULLIF(
            SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN
                COALESCE(CarrierDelay,0) + COALESCE(WeatherDelay,0) +
                COALESCE(NASDelay,0) + COALESCE(SecurityDelay,0) +
                COALESCE(LateAircraftDelay,0) ELSE 0 END), 0),
        4
    )                                                                          AS weather_delay_pct,
    ROUND(
        SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN NASDelay          ELSE 0 END)
        / NULLIF(
            SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN
                COALESCE(CarrierDelay,0) + COALESCE(WeatherDelay,0) +
                COALESCE(NASDelay,0) + COALESCE(SecurityDelay,0) +
                COALESCE(LateAircraftDelay,0) ELSE 0 END), 0),
        4
    )                                                                          AS nas_delay_pct,
    ROUND(
        SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN LateAircraftDelay  ELSE 0 END)
        / NULLIF(
            SUM(CASE WHEN Cancelled = 0 AND ArrDel15 = 1 THEN
                COALESCE(CarrierDelay,0) + COALESCE(WeatherDelay,0) +
                COALESCE(NASDelay,0) + COALESCE(SecurityDelay,0) +
                COALESCE(LateAircraftDelay,0) ELSE 0 END), 0),
        4
    )                                                                          AS late_ac_delay_pct
FROM flights
GROUP BY Route, Year, Month;


-- Capacity Utilisation: join flights and capacity for LF insight
CREATE OR REPLACE VIEW capacity_utilization AS
SELECT
    c.Route,
    c.Year,
    c.Month,
    c.YearMonth,
    c.DepScheduled,
    c.DepPerformed,
    c.Seats,
    c.Passengers,
    c.LoadFactor,
    r.otp_rate,
    r.avg_arr_delay_min,
    r.cancel_rate,
    r.delayed_flights,
    r.operated_flights
FROM capacity c
LEFT JOIN route_performance r
    ON c.Route = r.Route
    AND c.Year  = r.Year
    AND c.Month = r.Month;


-- Weather Impact: how weather conditions correlate with delays at FLL
CREATE OR REPLACE VIEW weather_impact AS
SELECT
    f.Route,
    f.Year,
    f.Month,
    ROUND(AVG(f.weather_severity), 2)                                          AS avg_weather_severity,
    ROUND(AVG(CASE WHEN f.Cancelled = 0 THEN f.ArrDelay END), 2)               AS avg_arr_delay,
    ROUND(
        SUM(CASE WHEN f.ArrDel15 = 0 AND f.Cancelled = 0 THEN 1 ELSE 0 END)::DOUBLE
        / NULLIF(SUM(CASE WHEN f.Cancelled = 0 THEN 1 ELSE 0 END), 0),
        4
    )                                                                           AS otp_rate,
    SUM(f.thunderstorm_flag)                                                    AS thunderstorm_count,
    SUM(f.precipitation_flag)                                                   AS precipitation_count,
    SUM(f.low_visibility_flag)                                                  AS low_vis_count,
    SUM(f.wind_gust_flag)                                                       AS wind_gust_count,
    COUNT(*)                                                                    AS total_flights
FROM flights f
GROUP BY f.Route, f.Year, f.Month;

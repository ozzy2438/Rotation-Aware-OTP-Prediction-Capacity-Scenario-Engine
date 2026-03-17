-- Weather Impact Analysis: correlation between weather conditions and OTP/delays.

-- Monthly weather severity vs OTP by route
SELECT
    wi.Route,
    wi.Year,
    wi.Month,
    PRINTF('%d-%02d', wi.Year, wi.Month)                        AS YearMonth,
    wi.avg_weather_severity,
    wi.thunderstorm_count,
    wi.precipitation_count,
    wi.low_vis_count,
    wi.wind_gust_count,
    ROUND(wi.otp_rate * 100, 1)                                 AS otp_pct,
    wi.avg_arr_delay                                            AS avg_arr_delay_min,
    wi.total_flights,
    -- Categorise weather severity
    CASE
        WHEN wi.avg_weather_severity >= 4.0 THEN 'Severe'
        WHEN wi.avg_weather_severity >= 2.0 THEN 'Moderate'
        WHEN wi.avg_weather_severity >= 0.5 THEN 'Light'
        ELSE 'Clear'
    END                                                         AS weather_category
FROM weather_impact wi
ORDER BY wi.Year, wi.Month, wi.Route;


-- Delay cause breakdown for high-weather-severity months at FLL (all routes)
SELECT
    Year,
    Month,
    PRINTF('%d-%02d', Year, Month)                              AS YearMonth,
    SUM(avg_weather_severity * total_flights)
        / NULLIF(SUM(total_flights), 0)                         AS weighted_severity,
    ROUND(AVG(otp_rate) * 100, 1)                               AS avg_otp_pct,
    SUM(thunderstorm_count)                                     AS total_thunderstorms,
    SUM(precipitation_count)                                    AS total_precipitation,
    SUM(total_flights)                                          AS network_flights
FROM weather_impact
GROUP BY Year, Month
ORDER BY weighted_severity DESC
LIMIT 20;

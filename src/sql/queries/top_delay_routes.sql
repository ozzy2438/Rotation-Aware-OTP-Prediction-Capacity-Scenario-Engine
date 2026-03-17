-- Top Delay Routes: Rank FLL hub routes by average arrival delay
-- and on-time performance across the full analysis period.

SELECT
    Route,
    Dest,
    SUM(operated_flights)                                       AS total_operated,
    SUM(delayed_flights)                                        AS total_delayed,
    ROUND(
        SUM(delayed_flights)::DOUBLE / NULLIF(SUM(operated_flights), 0) * 100, 1
    )                                                           AS delay_rate_pct,
    ROUND(
        SUM(operated_flights - delayed_flights)::DOUBLE
        / NULLIF(SUM(operated_flights), 0) * 100, 1
    )                                                           AS otp_pct,
    ROUND(AVG(avg_arr_delay_min), 1)                            AS avg_arr_delay_min,
    ROUND(AVG(avg_carrier_delay), 1)                            AS avg_carrier_delay_min,
    ROUND(AVG(avg_weather_delay), 1)                            AS avg_weather_delay_min,
    ROUND(AVG(avg_nas_delay), 1)                                AS avg_nas_delay_min,
    ROUND(AVG(avg_late_ac_delay), 1)                            AS avg_late_ac_delay_min,
    ROUND(AVG(cancel_rate) * 100, 2)                            AS avg_cancel_rate_pct
FROM route_performance
GROUP BY Route, Dest
ORDER BY avg_arr_delay_min DESC NULLS LAST;

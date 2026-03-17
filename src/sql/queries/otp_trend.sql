-- OTP Trend: Monthly on-time performance trend for all routes
-- and network-wide aggregate.

-- Network-wide monthly OTP
SELECT
    YearMonth,
    Year,
    Month,
    network_otp                                                 AS network_otp_rate,
    ROUND(network_otp * 100, 1)                                 AS otp_pct,
    avg_arr_delay                                               AS avg_arr_delay_min,
    avg_dep_delay                                               AS avg_dep_delay_min,
    total_flights,
    cancelled_flights,
    ROUND(cancel_rate * 100, 2)                                 AS cancel_rate_pct,
    routes_operated
FROM monthly_otp_summary
ORDER BY Year, Month;

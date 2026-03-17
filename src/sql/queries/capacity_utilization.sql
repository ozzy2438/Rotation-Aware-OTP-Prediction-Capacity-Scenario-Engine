-- Capacity Utilization: Load factor trends and efficiency metrics.

-- Route-level load factor by month with OTP overlay
SELECT
    cu.Route,
    cu.YearMonth,
    cu.Year,
    cu.Month,
    cu.DepScheduled,
    cu.DepPerformed,
    ROUND((1 - cu.DepPerformed::DOUBLE / NULLIF(cu.DepScheduled, 0)) * 100, 2)  AS completion_factor_loss_pct,
    cu.Seats,
    cu.Passengers,
    ROUND(cu.LoadFactor * 100, 1)                               AS load_factor_pct,
    ROUND(cu.otp_rate * 100, 1)                                 AS otp_pct,
    cu.avg_arr_delay_min,
    cu.cancel_rate,
    -- Efficiency score: composite of LF and OTP
    ROUND((cu.LoadFactor * 0.6 + cu.otp_rate * 0.4) * 100, 1)  AS efficiency_score
FROM capacity_utilization cu
WHERE cu.LoadFactor IS NOT NULL
ORDER BY cu.Route, cu.YearMonth;


-- Summer vs winter load factor comparison per route
SELECT
    Route,
    CASE
        WHEN Month IN (6, 7, 8) THEN 'Summer (Jun-Aug)'
        WHEN Month IN (12, 1, 2) THEN 'Winter (Dec-Feb)'
        WHEN Month IN (3, 4, 5) THEN 'Spring (Mar-May)'
        ELSE 'Fall (Sep-Nov)'
    END                                                         AS season,
    ROUND(AVG(LoadFactor) * 100, 1)                             AS avg_lf_pct,
    ROUND(MIN(LoadFactor) * 100, 1)                             AS min_lf_pct,
    ROUND(MAX(LoadFactor) * 100, 1)                             AS max_lf_pct,
    ROUND(AVG(otp_rate) * 100, 1)                               AS avg_otp_pct,
    SUM(Passengers)                                             AS total_passengers,
    SUM(Seats)                                                  AS total_seats
FROM capacity_utilization
WHERE LoadFactor IS NOT NULL AND otp_rate IS NOT NULL
GROUP BY Route, season
ORDER BY Route, avg_lf_pct DESC;

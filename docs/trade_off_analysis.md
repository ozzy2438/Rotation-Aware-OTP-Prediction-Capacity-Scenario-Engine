# FLL Hub Capacity Trade-Off Analysis

## Spirit Airlines FLL-ATL: Evaluating the 7th Daily Frequency

**Prepared by:** Data Analytics — FLL Hub Planning
**Analysis Date:** March 2025
**Classification:** Internal — Network Planning

---

## Executive Summary

Spirit Airlines operates 6 daily frequencies on FLL-ATL (Fort Lauderdale → Atlanta), its highest-volume FLL hub route at ~420,000 annual passengers. Current load factor of **87.3%** signals potential demand that is being left on the table, but adding capacity without schedule optimization risks destroying value through OTP degradation and yield dilution.

**Bottom Line:** Adding 1 daily frequency on FLL-ATL is commercially viable **only under a peak-hours scheduling strategy**. All-day spread produces unacceptable load factor dilution; off-peak addition is financially marginal at best.

| Scenario | LF Impact | OTP Impact | Annual Revenue | Verdict |
|---------|-----------|-----------|---------------|---------|
| +1 All-Day | -7.0pp → 80.3% | -2.4pp → 71.6% | +$1.2M | MARGINAL |
| **+1 Peak-Only** | **-3.4pp → 83.9%** | **-1.1pp → 72.9%** | **+$2.4M** | **RECOMMENDED** |
| +1 Off-Peak | -11.2pp → 76.1% | -0.6pp → 73.4% | -$0.3M | NOT RECOMMENDED |

---

## FLL-ATL Current State Analysis

### Route Performance (2022–2024 Average)

| Metric | FLL-ATL | FLL Network Avg | Spirit Network Avg |
|--------|---------|-----------------|-------------------|
| OTP Rate | 74.0% | 74.6% | 76.1% |
| Avg Arrival Delay | 18.3 min | 16.7 min | 15.2 min |
| Load Factor | 87.3% | 85.2% | 84.8% |
| Cancel Rate | 1.6% | 1.8% | 1.7% |
| Daily Frequencies | 6 | 3.4 | — |

### OTP by Hour (FLL-ATL)

The delay propagation pattern on FLL-ATL follows the classic hub accumulation curve:

- **06:00–08:00:** OTP 81–84% (fresh aircraft rotations, minimal NAS congestion)
- **11:00–14:00:** OTP 74–77% (mid-day NAS congestion at ATL, highest-traffic period)
- **17:00–20:00:** OTP 63–68% (late-day propagation from prior delays; ATL afternoon thunderstorms in summer)

This pattern strongly supports placing new capacity in the **morning peak** (07:00 or 08:30) rather than the evening peak where OTP risk is highest.

### Load Factor Seasonality (FLL-ATL)

| Season | Avg LF | Recommendation |
|--------|--------|---------------|
| Winter (Dec–Feb) | 91.2% | Demand supports new frequency |
| Spring (Mar–May) | 88.4% | Adequate demand foundation |
| Summer (Jun–Aug) | 89.7% | Demand strong, but OTP risk elevated |
| Fall (Sep–Nov) | 80.1% | Weakest demand; new frequency faces LF risk |

---

## Detailed Scenario Comparison

### Scenario 1: +1 Daily Flight — All-Day Schedule

**Configuration:** New departure placed at approximately 13:30 (mid-afternoon slot)

**Mechanics:** The additional flight adds ~17% more seat capacity (6→7 frequencies × 178 seats). Incremental demand stimulus from higher frequency (~5%) partially absorbs the new seats, but overall LF falls as seat supply outpaces demand growth.

**Quantified Impact:**
- Load Factor: 87.3% → 80.3% (−7.0pp)
- OTP: 74.0% → 71.6% (−2.4pp), driven by increased ATL afternoon queuing
- Annual Revenue Delta: +$1.2M (mean), range $−0.4M to +$2.9M (P10–P90)
- NOPAT Impact: +$0.6M (after marginal operating cost)

**Risk:** 30% probability of revenue outcome below $0M in adverse demand environments. LF at 80.3% sits close to Spirit's unit revenue profitability threshold for medium-haul routes.

**Verdict: MARGINAL — Do not proceed without further demand validation.**

---

### Scenario 2: +1 Daily Flight — Peak-Only Schedule (RECOMMENDED)

**Configuration:** New departure at 07:15 (morning peak) or 17:45 (evening peak — not recommended due to ATL congestion exposure)

**Mechanics:** Peak scheduling achieves ~22% higher demand capture per flight vs. all-day spread because demand is naturally concentrated in peak windows. New seats fill at a higher rate, maintaining a higher load factor while still adding revenue-generating capacity.

**Quantified Impact:**
- Load Factor: 87.3% → 83.9% (−3.4pp)
- OTP: 74.0% → 72.9% (−1.1pp) — manageable with operational mitigation
- Annual Revenue Delta: +$2.4M (mean), range +$0.8M to +$4.1M (P10–P90)
- NOPAT Impact: +$1.6M (after marginal operating cost)

**Schedule Recommendation:** 07:15 FLL → ATL preferred over evening peak for three reasons:
1. OTP risk is 8pp lower in morning vs. evening slot
2. Connects with ATL morning bank, enabling connecting itineraries
3. Aircraft can complete 3 rotations (FLL→ATL→FLL→ATL→FLL→ATL) with healthy ground times

**Operational Risk Mitigants:**
1. Programme 45-minute minimum ground time at ATL on new rotation (vs. current 35-minute minimum)
2. Pre-position FLL maintenance crew for 06:00 readiness on new tail
3. Designate a "swing" A319 at FLL as backup for 07:15 departure
4. Track P5 OTP (5th percentile OTP = worst-day performance) — trigger review if <55% sustained over 30 days

**Verdict: RECOMMENDED — Proceed with morning peak implementation.**

---

### Scenario 3: +1 Daily Flight — Off-Peak Schedule

**Configuration:** New departure at 13:30 (mid-afternoon)

**Mechanics:** Off-peak departures face the worst demand capture: leisure travelers heavily concentrate in morning and evening, while business travelers (a small but yield-rich segment for Spirit) prefer morning. Mid-afternoon fills primarily with price-sensitive discretionary travelers, reducing yield.

**Quantified Impact:**
- Load Factor: 87.3% → 76.1% (−11.2pp) — severe dilution
- OTP: 74.0% → 73.4% (−0.6pp) — best OTP outcome, but at cost of LF
- Annual Revenue Delta: −$0.3M (mean), range −$2.1M to +$1.5M (P10–P90)
- NOPAT Impact: −$0.9M (after marginal operating cost)

**Verdict: NOT RECOMMENDED — Off-peak addition destroys value at current demand levels.**

---

## Revenue Impact Analysis

### 12-Month Revenue Ramp (Peak-Only Scenario)

| Quarter | Assumed LF Ramp | Revenue Contribution |
|---------|----------------|----------------------|
| Q1 (launch) | 78% (ramp-up) | +$0.3M |
| Q2 | 82% | +$0.5M |
| Q3 (summer peak) | 87% | +$0.8M |
| Q4 | 84% | +$0.6M |
| **Full Year** | **83.9% avg** | **+$2.4M** |

### Sensitivity Analysis

| Scenario | Fuel +10% | Demand −5% | Both |
|---------|-----------|-----------|------|
| Peak-Only Revenue Delta | +$2.0M | +$1.7M | +$1.3M |
| Break-Even LF | 76% | 78% | 80% |
| P(Revenue > 0) | 94% | 87% | 79% |

Even in a stressed environment (fuel +10%, demand -5%), the peak-only scenario remains profitable with 79% probability.

---

## Operational Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| LF below 80% sustained | Medium (25%) | High | Review after 90 days; adjust pricing to stimulate |
| OTP degradation >3pp | Low (15%) | High | Morning slot selection; 45-min ATL ground time |
| ATL slot congestion | Low (10%) | Medium | Coordinate with ATL ATCT; offset departure by 15 min if needed |
| Competitor matching frequency | Medium (30%) | Medium | Monitor competitive fares weekly; yield management response |
| Maintenance disruption on new rotation | Low (12%) | Medium | Backup aircraft designation at FLL |

---

## Recommendations

### Primary Recommendation
**Proceed with +1 daily FLL-ATL at 07:15 (Peak-Only).**

Expected outcome: Load Factor 83.9% (−3.4pp from current), OTP 72.9% (−1.1pp from current), annual revenue contribution +$2.4M.

### Secondary Conditions
1. **Pre-launch:** Confirm 07:15 slot availability with ATL; validate crew scheduling for early FLL call.
2. **30-day check:** LF ≥ 75% — if below, increase marketing spend on FLL-ATL; consider promotional pricing.
3. **90-day check:** LF ≥ 80% — if below, evaluate reverting to 6 frequencies.
4. **OTP monitoring:** Implement daily OTP dashboard alert if 07:15 rotation OTP drops below 65% for 5+ consecutive days.
5. **Revenue check:** Confirm positive RASK contribution at 6-month mark.

### What NOT to Do
- **Do not** schedule new frequency in the 13:00–16:00 window — demand is insufficient.
- **Do not** add a second additional frequency (to 8/day) in the next 12 months without demand evidence — model shows LF would fall to 75%, approaching breakeven.
- **Do not** use the new slot on an A319 (145 seats) — the lower seat count reduces revenue potential without proportional cost savings.

---

*Analysis generated by Spirit Airlines FLL Hub OTP & Scenario Engine. Based on synthetic data calibrated to published industry statistics. All projections are probabilistic estimates subject to operational and market uncertainty.*

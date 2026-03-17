"""
LLM conversational analytics layer for Spirit Airlines OTP Engine.

Converts natural language questions to DuckDB SQL queries using OpenAI GPT-4o-mini.
Falls back to pattern-matching when no API key is available.

Usage:
    python src/dashboard/llm_query.py
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "spirit_otp.duckdb"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Schema description for the LLM system prompt
# ---------------------------------------------------------------------------
DB_SCHEMA = """
You are a data analytics assistant for Spirit Airlines FLL Hub operations.
You have access to a DuckDB database with the following tables and views:

TABLE: flights
  FlightDate DATE, Reporting_Airline TEXT, Tail_Number TEXT,
  Origin TEXT, Dest TEXT, CRSDepTime INT, DepTime INT,
  DepDelay FLOAT, ArrDelay FLOAT, ArrDel15 INT (1=delayed≥15min),
  Cancelled INT (1=cancelled), CancellationCode TEXT,
  CarrierDelay FLOAT, WeatherDelay FLOAT, NASDelay FLOAT,
  SecurityDelay FLOAT, LateAircraftDelay FLOAT,
  Distance INT, AirTime FLOAT, AircraftType TEXT, Route TEXT,
  Year INT, Month INT, DayOfWeek INT (0=Mon), Quarter INT, DepHour INT,
  weather_severity FLOAT (0-10), thunderstorm_flag INT,
  precipitation_flag INT, low_visibility_flag INT, wind_gust_flag INT,
  tmpf FLOAT (temperature °F), vsby FLOAT (visibility miles)

TABLE: capacity
  Year INT, Month INT, YearMonth TEXT, Origin TEXT, Dest TEXT,
  Route TEXT, Reporting_Airline TEXT, AircraftType TEXT,
  DepScheduled INT, DepPerformed INT, Seats INT,
  Passengers INT, LoadFactor FLOAT

TABLE: weather
  valid DATETIME, station TEXT, tmpf FLOAT, dwpf FLOAT,
  drct FLOAT, sknt FLOAT, p01i FLOAT, vsby FLOAT, gust FLOAT,
  skyc1 TEXT, wxcodes TEXT, thunderstorm_flag INT,
  precipitation_flag INT, weather_severity FLOAT

VIEW: route_performance  -- route + month level OTP, delay stats
VIEW: monthly_otp_summary -- network-wide monthly OTP
VIEW: delay_cause_decomposition -- delay cause fractions per route/month
VIEW: capacity_utilization -- joins capacity + route_performance
VIEW: weather_impact -- weather conditions vs OTP

Routes in the data: FLL-ATL, FLL-LAS, FLL-LAX, FLL-ORD, FLL-DFW,
  FLL-MCO, FLL-JFK, FLL-BOS, FLL-DTW, FLL-MIA
Date range: 2022-01-01 to 2024-12-31
Airline: Spirit Airlines (NK)

Rules:
1. Return ONLY valid DuckDB SQL — no markdown, no explanation, no ```
2. Use ROUND() for all calculated metrics
3. Alias columns with readable names using AS
4. Limit results to 50 rows unless the user asks for more
5. For OTP rate queries, use: SUM(CASE WHEN ArrDel15=0 AND Cancelled=0 THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN Cancelled=0 THEN 1 ELSE 0 END), 0)
"""

# ---------------------------------------------------------------------------
# Pattern-matching fallback queries
# ---------------------------------------------------------------------------
FALLBACK_PATTERNS: list[tuple[str, str, str]] = [
    (
        r"(worst|best) otp|on.?time.*(route|performance)",
        """
        SELECT Route, ROUND(AVG(otp_rate)*100,1) AS otp_pct,
               ROUND(AVG(avg_arr_delay_min),1) AS avg_delay_min,
               SUM(operated_flights) AS total_flights
        FROM route_performance
        GROUP BY Route
        ORDER BY otp_pct ASC
        LIMIT 10
        """,
        "Routes ranked by OTP rate (worst first).",
    ),
    (
        r"delay cause|delay breakdown|what cause",
        """
        SELECT Route,
               ROUND(AVG(avg_carrier_delay),1) AS carrier_delay_avg,
               ROUND(AVG(avg_weather_delay),1) AS weather_delay_avg,
               ROUND(AVG(avg_nas_delay),1) AS nas_delay_avg,
               ROUND(AVG(avg_late_ac_delay),1) AS late_ac_delay_avg
        FROM delay_cause_decomposition
        GROUP BY Route
        ORDER BY carrier_delay_avg DESC
        LIMIT 10
        """,
        "Average delay minutes by cause for each route.",
    ),
    (
        r"load factor|capacity utilization|lf",
        """
        SELECT Route, YearMonth,
               ROUND(LoadFactor*100,1) AS load_factor_pct,
               ROUND(otp_rate*100,1) AS otp_pct,
               Passengers, Seats
        FROM capacity_utilization
        WHERE LoadFactor IS NOT NULL
        ORDER BY YearMonth DESC, load_factor_pct DESC
        LIMIT 20
        """,
        "Load factor and OTP by route and month.",
    ),
    (
        r"summer|winter|seasonal|season",
        """
        SELECT Route,
               CASE WHEN Month IN (6,7,8) THEN 'Summer'
                    WHEN Month IN (12,1,2) THEN 'Winter'
                    WHEN Month IN (3,4,5) THEN 'Spring'
                    ELSE 'Fall' END AS season,
               ROUND(AVG(LoadFactor)*100,1) AS avg_lf_pct,
               ROUND(AVG(otp_rate)*100,1) AS avg_otp_pct
        FROM capacity_utilization
        WHERE LoadFactor IS NOT NULL
        GROUP BY Route, season
        ORDER BY Route, season
        """,
        "Seasonal load factor and OTP comparison by route.",
    ),
    (
        r"weather|thunder|storm|delay.*weather|weather.*delay",
        """
        SELECT wi.Route,
               ROUND(AVG(wi.avg_weather_severity),2) AS avg_severity,
               SUM(wi.thunderstorm_count) AS thunderstorms,
               ROUND(AVG(wi.otp_rate)*100,1) AS avg_otp_pct,
               ROUND(AVG(wi.avg_arr_delay),1) AS avg_delay_min
        FROM weather_impact wi
        GROUP BY wi.Route
        ORDER BY avg_severity DESC
        """,
        "Weather severity impact on OTP and delays by route.",
    ),
    (
        r"monthly|trend|over time|month",
        """
        SELECT YearMonth, ROUND(network_otp*100,1) AS otp_pct,
               ROUND(avg_arr_delay,1) AS avg_delay_min,
               total_flights, cancelled_flights
        FROM monthly_otp_summary
        ORDER BY Year, Month
        """,
        "Monthly network-wide OTP trend.",
    ),
    (
        r"december|holiday|peak season",
        """
        SELECT Route,
               ROUND(AVG(otp_rate)*100,1) AS avg_otp_pct,
               ROUND(AVG(avg_arr_delay_min),1) AS avg_delay_min,
               SUM(operated_flights) AS total_flights
        FROM route_performance
        WHERE Month = 12
        GROUP BY Route
        ORDER BY avg_otp_pct ASC
        """,
        "Route OTP in December (holiday peak season).",
    ),
    (
        r"fll.atl|atlanta|fll to atl",
        """
        SELECT Year, Month,
               ROUND(otp_rate*100,1) AS otp_pct,
               ROUND(avg_arr_delay_min,1) AS avg_delay_min,
               operated_flights, delayed_flights
        FROM route_performance
        WHERE Route = 'FLL-ATL'
        ORDER BY Year, Month
        """,
        "FLL-ATL route performance over time.",
    ),
]


# ---------------------------------------------------------------------------
# LLM Query Engine
# ---------------------------------------------------------------------------

class LLMQueryEngine:
    """Natural language to SQL query engine for Spirit Airlines DuckDB.

    Uses OpenAI GPT-4o-mini for SQL generation with pattern-matching fallback.

    Attributes:
        db_path: Path to DuckDB database file.
        use_llm: Whether the OpenAI API key is available.
        conversation_history: List of prior messages in the session.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """Initialise the query engine.

        Args:
            db_path: Path to the DuckDB file. Defaults to project default.
        """
        self.db_path = db_path or DB_PATH
        self.use_llm = bool(OPENAI_API_KEY)
        self.conversation_history: list[dict[str, str]] = []

        if self.use_llm:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI LLM backend initialised")
            except ImportError:
                logger.warning("openai package not installed; using fallback")
                self.use_llm = False
        else:
            logger.info("No OPENAI_API_KEY found; using pattern-matching fallback")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict[str, Any]:
        """Process a natural language question and return structured response.

        Args:
            question: Plain-English analytics question.

        Returns:
            Dict with keys:
              - sql: The SQL query executed.
              - data: DataFrame with results.
              - interpretation: Natural language summary.
              - error: Error message (empty string if successful).
              - source: 'llm' or 'fallback'.
        """
        logger.info("Query received: %s", question[:80])

        sql = ""
        source = "fallback"
        error = ""

        if self.use_llm:
            sql, source, error = self._generate_sql_llm(question)

        if not sql or error:
            sql, fallback_interpretation = self._generate_sql_fallback(question)
            source = "fallback"
            error = ""
        else:
            fallback_interpretation = ""

        # Execute SQL
        data, exec_error = self._execute_sql(sql)

        if exec_error:
            # Try fallback
            sql, fallback_interpretation = self._generate_sql_fallback(question)
            data, exec_error = self._execute_sql(sql)
            source = "fallback"

        interpretation = (
            self._generate_interpretation_llm(question, data, sql)
            if self.use_llm and not exec_error
            else self._generate_interpretation_fallback(question, data, fallback_interpretation)
        )

        return {
            "sql": sql,
            "data": data,
            "interpretation": interpretation,
            "error": exec_error,
            "source": source,
            "row_count": len(data) if data is not None else 0,
        }

    def reset_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    # ------------------------------------------------------------------
    # LLM methods
    # ------------------------------------------------------------------

    def _generate_sql_llm(self, question: str) -> tuple[str, str, str]:
        """Use OpenAI to generate SQL from the question.

        Args:
            question: User question.

        Returns:
            Tuple of (sql, source, error).
        """
        try:
            self.conversation_history.append({"role": "user", "content": question})

            messages = [{"role": "system", "content": DB_SCHEMA}] + self.conversation_history

            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=600,
                temperature=0,
            )
            sql = response.choices[0].message.content.strip()
            # Strip any markdown fences that slipped through
            sql = re.sub(r"```sql\s*|```\s*", "", sql).strip()

            self.conversation_history.append({"role": "assistant", "content": sql})
            return sql, "llm", ""
        except Exception as exc:
            logger.warning("LLM SQL generation failed: %s", exc)
            return "", "llm", str(exc)

    def _generate_interpretation_llm(
        self,
        question: str,
        data: Optional[pd.DataFrame],
        sql: str,
    ) -> str:
        """Use OpenAI to interpret query results in natural language.

        Args:
            question: Original user question.
            data: Query result DataFrame.
            sql: The SQL that was executed.

        Returns:
            Natural language interpretation string.
        """
        try:
            if data is None or data.empty:
                return "No data was returned for your query."

            sample = data.head(5).to_string(index=False)
            prompt = (
                f"The user asked: '{question}'\n"
                f"SQL executed:\n{sql}\n\n"
                f"Results (first 5 rows):\n{sample}\n\n"
                f"Total rows: {len(data)}\n\n"
                "Provide a concise 2-4 sentence business interpretation of these results "
                "focused on actionable insights for Spirit Airlines operations."
            )
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise airline operations analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("LLM interpretation failed: %s", exc)
            return self._generate_interpretation_fallback(question, data, "")

    # ------------------------------------------------------------------
    # Fallback methods
    # ------------------------------------------------------------------

    def _generate_sql_fallback(self, question: str) -> tuple[str, str]:
        """Match question against known patterns and return a pre-written SQL.

        Args:
            question: User question.

        Returns:
            Tuple of (sql, interpretation_template).
        """
        q_lower = question.lower()
        for pattern, sql, interpretation in FALLBACK_PATTERNS:
            if re.search(pattern, q_lower):
                logger.info("Pattern match: %s", pattern)
                return sql.strip(), interpretation

        # Default: route performance summary
        default_sql = """
        SELECT Route,
               ROUND(AVG(otp_rate)*100,1) AS otp_pct,
               ROUND(AVG(avg_arr_delay_min),1) AS avg_delay_min,
               SUM(operated_flights) AS total_flights
        FROM route_performance
        GROUP BY Route
        ORDER BY otp_pct DESC
        LIMIT 10
        """
        return default_sql.strip(), "Route OTP and average delay summary."

    def _generate_interpretation_fallback(
        self,
        question: str,
        data: Optional[pd.DataFrame],
        template: str,
    ) -> str:
        """Generate a basic interpretation without the LLM.

        Args:
            question: Original question.
            data: Query result DataFrame.
            template: Pre-written template string.

        Returns:
            Interpretation string.
        """
        if data is None or data.empty:
            return "No data found matching your query. Try rephrasing or check the date range."

        n = len(data)
        summary = f"{template} Returned {n} record(s). "

        # Generic stats
        numeric_cols = data.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            summary += (
                f"The range of {col} is "
                f"{data[col].min():.1f} – {data[col].max():.1f}."
            )
        return summary

    # ------------------------------------------------------------------
    # SQL execution
    # ------------------------------------------------------------------

    def _execute_sql(self, sql: str) -> tuple[Optional[pd.DataFrame], str]:
        """Execute SQL against the DuckDB database.

        Args:
            sql: SQL query string.

        Returns:
            Tuple of (DataFrame or None, error message).
        """
        try:
            if not self.db_path.exists():
                return None, f"Database not found at {self.db_path}. Run ETL pipeline first."

            con = duckdb.connect(str(self.db_path), read_only=True)
            df = con.execute(sql).fetchdf()
            con.close()
            logger.info("SQL executed successfully: %d rows returned", len(df))
            return df, ""
        except Exception as exc:
            logger.error("SQL execution error: %s\nSQL: %s", exc, sql[:200])
            return None, str(exc)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def ask(question: str, db_path: Optional[Path] = None) -> dict[str, Any]:
    """One-shot convenience function to query the analytics engine.

    Args:
        question: Natural language question.
        db_path: Optional path to DuckDB file.

    Returns:
        Query result dict (see LLMQueryEngine.query).
    """
    engine = LLMQueryEngine(db_path=db_path)
    return engine.query(question)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = LLMQueryEngine()

    test_questions = [
        "Which route has the worst OTP?",
        "What are the most common delay causes at FLL?",
        "Compare load factors for summer vs winter",
        "Show me the OTP trend over time",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        result = engine.query(q)
        print(f"Source: {result['source']}")
        print(f"SQL: {result['sql'][:100]}...")
        print(f"Rows: {result['row_count']}")
        print(f"Answer: {result['interpretation']}")
        if result["data"] is not None and not result["data"].empty:
            print(result["data"].head(3).to_string(index=False))

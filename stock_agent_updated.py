import os
import re
import json
import operator
from typing import TypedDict, Annotated, Optional

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from langsmith import traceable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

import os

key = os.getenv("GROQ_API_KEY", "")

print(
    f"GROQ_DIAGNOSTIC: present={bool(key)}, length={len(key)}"
)

import os
import logging

logging.basicConfig(level=logging.INFO)

key = os.getenv("GROQ_API_KEY", "")

logging.info(
    "GROQ_API_KEY present=%s, length=%s",
    bool(key),
    len(key)
)

import os
import logging

logging.info(
    "GROQ_API_KEY present=%s, length=%s",
    bool(os.getenv("GROQ_API_KEY")),
    len(os.getenv("GROQ_API_KEY", ""))
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    temperature=0,
    api_key=GROQ_API_KEY,
)

ALLOWED_TASKS = {"market", "prediction", "risk", "research"}

# Known company names are mapped explicitly. Unknown bare symbols are NOT
# automatically converted to .NS because that could analyse the wrong company.
STOCK_ALIASES = {
    "TCS": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "WIPRO": "WIPRO.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "HCL TECH": "HCLTECH.NS",
    "HCLTECH": "HCLTECH.NS",
    "AXIS BANK": "AXISBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "LARSEN AND TOUBRO": "LT.NS",
    "L&T": "LT.NS",
    "LT": "LT.NS",
    "MARUTI": "MARUTI.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANIENT": "ADANIENT.NS",
    "TITAN": "TITAN.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "POWER GRID": "POWERGRID.NS",
}


def normalize_ticker(ticker: str) -> str:
    """Validate an exchange-qualified ticker; never guess its exchange."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker is missing. Please provide a company name or ticker.")

    symbol = ticker.strip().upper().replace("$", "")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,24}", symbol):
        raise ValueError(f"Invalid ticker format: {ticker!r}")

    if "." not in symbol:
        raise ValueError(
            f"Ticker '{symbol}' has no exchange suffix. Use an explicit symbol such as RELIANCE.NS or RELIANCE.BO."
        )
    return symbol


def extract_ticker(query: str) -> Optional[str]:
    """Extract a known company alias or an exchange-qualified ticker."""
    query_upper = query.upper()

    for alias in sorted(STOCK_ALIASES, key=len, reverse=True):
        pattern = r"(?<![A-Z0-9])" + re.escape(alias) + r"(?![A-Z0-9])"
        if re.search(pattern, query_upper):
            return STOCK_ALIASES[alias]

    match = re.search(
        r"(?<![A-Z0-9])([A-Z][A-Z0-9]{0,14}\.(?:NS|BO|NASDAQ|NYSE|L|TO))(?![A-Z0-9])",
        query_upper,
    )
    if match:
        return normalize_ticker(match.group(1))

    return None


# ============================================================
# 2. SHARED LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict, total=False):
    query: str
    ticker: str
    tasks: list[str]
    results: Annotated[list[dict], operator.add]
    final_answer: str
    needs_clarification: bool
    clarification_question: str


# ============================================================
# 3. MARKET DATA AGENT
# ============================================================

@traceable(name="Market Agent")
def market_agent(ticker: str) -> dict:
    symbol = normalize_ticker(ticker)
    stock = yf.Ticker(symbol)
    history = stock.history(period="5d", interval="1d")

    if history.empty:
        raise ValueError(f"No market data found for {symbol}. Check the ticker and exchange.")

    latest = history.iloc[-1]
    try:
        latest_quote = stock.fast_info.last_price
        if latest_quote is None or pd.isna(latest_quote):
            latest_quote = float(latest["Close"])
    except Exception:
        latest_quote = float(latest["Close"])

    return {
        "agent": "Market Agent",
        "ticker": symbol,
        "latest_quote": round(float(latest_quote), 2),
        "latest_daily_close": round(float(latest["Close"]), 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
        "latest_daily_date": str(history.index[-1].date()),
        "currency": "INR" if symbol.endswith((".NS", ".BO")) else "Check exchange currency",
        "source": "Yahoo Finance",
        "data_limitation": "Quote may be delayed; verify with an official market-data provider.",
    }


# ============================================================
# 4. PREDICTION AGENT
# ============================================================

@traceable(name="Prediction Agent")
def prediction_agent(ticker: str) -> dict:
    symbol = normalize_ticker(ticker)
    history = yf.Ticker(symbol).history(period="2y", interval="1d")

    if history.empty or len(history) < 100:
        raise ValueError(f"Insufficient historical data for {symbol}.")

    df = history[["Close"]].copy()
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["Close"].shift(lag - 1)

    df["rolling_mean_5"] = df["Close"].rolling(5).mean()
    df["rolling_mean_10"] = df["Close"].rolling(10).mean()
    df["target"] = df["Close"].shift(-1)

    features = [
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
        "rolling_mean_5", "rolling_mean_10",
    ]
    df = df.dropna()
    if len(df) < 50:
        raise ValueError("Not enough clean rows to train the prediction model.")

    X, y = df[features], df["target"]
    split = int(len(df) * 0.8)
    if split <= 0 or split >= len(df):
        raise ValueError("Unable to create a valid chronological train/test split.")

    model = LinearRegression()
    model.fit(X.iloc[:split], y.iloc[:split])
    test_predictions = model.predict(X.iloc[split:])
    mae = mean_absolute_error(y.iloc[split:], test_predictions)

    close = history["Close"].dropna()
    last_5, last_10 = close.tail(5), close.tail(10)
    next_input = pd.DataFrame([{
        "lag_1": last_5.iloc[-1],
        "lag_2": last_5.iloc[-2],
        "lag_3": last_5.iloc[-3],
        "lag_4": last_5.iloc[-4],
        "lag_5": last_5.iloc[-5],
        "rolling_mean_5": last_5.mean(),
        "rolling_mean_10": last_10.mean(),
    }], columns=features)

    predicted_price = model.predict(next_input)[0]
    return {
        "agent": "Prediction Agent",
        "ticker": symbol,
        "last_close": round(float(close.iloc[-1]), 2),
        "predicted_next_trading_day_close": round(float(predicted_price), 2),
        "historical_test_mae": round(float(mae), 2),
        "model": "Linear Regression",
        "training_period": "Approximately 2 years",
        "prediction_warning": "Experimental estimate, not a reliable investment forecast or guaranteed price.",
    }


# ============================================================
# 5. RISK ANALYSIS AGENT
# ============================================================

@traceable(name="Risk Analysis Agent")
def risk_agent(ticker: str) -> dict:
    symbol = normalize_ticker(ticker)
    history = yf.Ticker(symbol).history(period="1y", interval="1d")

    if history.empty or len(history) < 30:
        raise ValueError(f"Insufficient historical data for risk analysis of {symbol}.")

    close = history["Close"].dropna()
    returns = close.pct_change().dropna()
    if returns.empty:
        raise ValueError("Unable to calculate daily returns.")

    volatility = returns.std() * (252 ** 0.5) * 100
    max_drawdown = (close / close.cummax() - 1).min() * 100
    var_95 = max(0, -returns.quantile(0.05) * 100)

    return {
        "agent": "Risk Analysis Agent",
        "ticker": symbol,
        "annualized_volatility_percent": round(float(volatility), 2),
        "maximum_drawdown_percent": round(float(max_drawdown), 2),
        "historical_daily_var_95_percent": round(float(var_95), 2),
        "data_period": "Approximately 1 year",
        "method": "Historical daily returns",
        "warning": "Historical risk does not guarantee future risk.",
    }


# ============================================================
# 6. COMPANY RESEARCH AGENT
# ============================================================

@traceable(name="Research Agent")
def research_agent(ticker: str) -> dict:
    symbol = normalize_ticker(ticker)
    info = yf.Ticker(symbol).info

    fields = [
        "longName", "sector", "industry", "marketCap", "trailingPE",
        "forwardPE", "priceToBook", "dividendYield", "returnOnEquity",
        "profitMargins", "totalRevenue",
    ]
    fundamentals = {field: info.get(field) for field in fields if info.get(field) is not None}

    if not fundamentals:
        raise ValueError(f"Company fundamentals unavailable for {symbol}.")

    return {
        "agent": "Research Agent",
        "ticker": symbol,
        "fundamentals": fundamentals,
        "source": "Yahoo Finance",
        "note": "Fundamental data availability and freshness vary by exchange.",
    }


# ============================================================
# 7. DETERMINISTIC SUPERVISOR (NO STRUCTURED-OUTPUT DEPENDENCY)
# ============================================================

def select_tasks(query: str) -> list[str]:
    q = query.lower()
    all_terms = [
        "comprehensive", "complete analysis", "full analysis", "all agents",
        "good investment", "should i buy", "investment analysis", "analyse everything",
    ]
    if any(term in q for term in all_terms):
        return ["market", "prediction", "risk", "research"]

    tasks = []
    if any(term in q for term in ["price", "market", "quote", "ohlc", "volume", "trading"]):
        tasks.append("market")
    if any(term in q for term in ["predict", "prediction", "forecast", "tomorrow", "next day"]):
        tasks.append("prediction")
    if any(term in q for term in ["risk", "volatility", "drawdown", "var", "value at risk"]):
        tasks.append("risk")
    if any(term in q for term in ["research", "fundamental", "pe ratio", "market cap", "revenue", "company analysis"]):
        tasks.append("research")

    # General stock questions should run the complete analysis rather than silently
    # returning an incomplete market-only answer.
    return list(dict.fromkeys(tasks)) if tasks else ["market", "prediction", "risk", "research"]


def supervisor_node(state: AgentState) -> dict:
    query = state.get("query", "").strip()
    if not query:
        return {
            "ticker": "",
            "tasks": [],
            "needs_clarification": True,
            "clarification_question": "Please enter a stock question and specify the company or ticker.",
        }

    ticker = extract_ticker(query)
    tasks = select_tasks(query)

    if not ticker:
        return {
            "ticker": "",
            "tasks": tasks,
            "needs_clarification": True,
            "clarification_question": (
                "Which stock should I analyse? Please provide a company name such as TCS, "
                "Reliance, Infosys or Wipro, or an exchange-qualified ticker such as RELIANCE.NS."
            ),
        }

    print("\nSupervisor selected tasks:", tasks)
    print("Selected stock:", ticker)
    return {
        "ticker": ticker,
        "tasks": [task for task in tasks if task in ALLOWED_TASKS],
        "needs_clarification": False,
    }


# ============================================================
# 8. PARALLEL DISPATCH + CLARIFICATION ROUTING
# ============================================================

def dispatch_tasks(state: AgentState):
    ticker = state.get("ticker")
    if not ticker:
        raise ValueError("Ticker missing: refusing to dispatch agents.")

    tasks = state.get("tasks", [])
    if not tasks:
        raise ValueError("No specialist tasks selected.")

    return [
        Send("specialist_worker", {"task": task, "ticker": ticker})
        for task in tasks
    ]


def clarification_node(state: AgentState) -> dict:
    return {
        "final_answer": state.get(
            "clarification_question",
            "Please specify the stock name or exchange-qualified ticker.",
        )
    }


def route_after_supervisor(state: AgentState):
    if state.get("needs_clarification"):
        return "clarification"
    return dispatch_tasks(state)


# ============================================================
# 9. SPECIALIST WORKER
# ============================================================

def specialist_worker(state: dict) -> dict:
    task = state["task"]
    ticker = state["ticker"]
    print(f"\nExecuting {task} agent for {ticker}...")

    try:
        if task == "market":
            result = market_agent(ticker)
        elif task == "prediction":
            result = prediction_agent(ticker)
        elif task == "risk":
            result = risk_agent(ticker)
        elif task == "research":
            result = research_agent(ticker)
        else:
            result = {"agent": task, "ticker": ticker, "error": "Unknown task"}
    except Exception as exc:
        result = {"agent": task, "ticker": ticker, "error": f"{type(exc).__name__}: {exc}"}

    return {"results": [result]}


# ============================================================
# 10. FINAL REPORT AGENT
# ============================================================

@traceable(name="Final Report Agent")
def final_report_node(state: AgentState) -> dict:
    results = state.get("results", [])
    prompt = f"""
You are an enterprise financial research report assistant.

Original user question:
{state.get('query', '')}

Confirmed stock ticker:
{state.get('ticker', '')}

Specialist agent results:
{json.dumps(results, default=str)}

Prepare a professional report with:
1. Stock identification.
2. Latest available market data.
3. ML prediction, if available.
4. Risk analysis, if available.
5. Company fundamentals, if available.
6. Failed agents, missing data, source limitations and data date.

Rules:
- Use only the supplied results. Never invent prices or financial metrics.
- Clearly distinguish live/indicative quote from daily closing price.
- Clearly label prediction as an experimental model estimate, not a reliable forecast.
- Explain failed specialist agents.
- Do not present this as personalized investment advice.
- Use INR for Indian NSE/BSE stocks.
"""

    try:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        response = llm.invoke(prompt)
        answer = response.content
        if isinstance(answer, list):
            answer = "\n".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in answer
            )
    except Exception as exc:
        answer = (
            f"Final report generation failed: {type(exc).__name__}: {exc}\n\n"
            "Raw specialist results:\n"
            + json.dumps(results, indent=2, default=str)
        )

    return {"final_answer": answer}


# ============================================================
# 11. BUILD LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("clarification", clarification_node)
builder.add_node("specialist_worker", specialist_worker)
builder.add_node("final_report", final_report_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_after_supervisor)
builder.add_edge("clarification", END)
builder.add_edge("specialist_worker", "final_report")
builder.add_edge("final_report", END)

graph = builder.compile()


# ============================================================
# 12. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    print("\n===== ENTERPRISE STOCK AI SYSTEM =====")
    print("Type 'exit' to stop.")

    while True:
        try:
            query = input("\nAsk your stock question: ").strip()
            if query.lower() in {"exit", "quit", "stop"}:
                print("Goodbye!")
                break
            if not query:
                print("Please enter a question.")
                continue

            result = graph.invoke(
                {"query": query, "results": []},
                config={
                    "run_name": "Enterprise Stock Analysis",
                    "tags": ["stock-agent", "langgraph", "multi-agent"],
                    "metadata": {
                        "application": "Stock AI",
                        "environment": os.getenv("APP_ENV", "development"),
                    },
                },
            )

            print("\n========== FINAL REPORT ==========\n")
            print(result.get("final_answer", "No report generated."))

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as exc:
            print(f"\nApplication error: {type(exc).__name__}: {exc}")

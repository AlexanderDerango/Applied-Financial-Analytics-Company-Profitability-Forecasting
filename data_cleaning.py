"""
FinLens — Financial ML Analyzer
================================
Usage:
    python analyze.py

Folder structure expected:
    data/
        AAPL/
            income_statement.csv
            balance_sheet.csv
            cashflow_statement.csv
        MSFT/
            income_statement.csv
            ...
        (up to 7 companies)

Output:
    financial_report.html   ← open this in any browser

Requirements:
    pip install pandas numpy scikit-learn scipy
"""

import os
import sys
import json
import math
import glob
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — Edit these paths to match your folder layout
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "data"          # Root folder containing one subfolder per company
OUTPUT     = "financial_report.html"

# Column name mappings — handles variations across different data providers
COL_MAP = {
    # Income Statement
    "revenue":       ["totalRevenue", "total_revenue", "revenue", "Revenue"],
    "gross_profit":  ["grossProfit", "gross_profit", "GrossProfit"],
    "op_income":     ["operatingIncome", "operating_income", "OperatingIncome"],
    "net_income":    ["netIncome", "net_income", "NetIncome", "netIncomeFromContinuingOperations"],
    "ebitda":        ["ebitda", "EBITDA", "Ebitda"],
    "rd":            ["researchAndDevelopment", "research_and_development", "R&D"],
    "cogs":          ["costOfRevenue", "cost_of_revenue", "costofGoodsAndServicesSold"],
    "ebit":          ["ebit", "EBIT"],
    "sga":           ["sellingGeneralAndAdministrative", "selling_general_administrative"],
    "income_tax":    ["incomeTaxExpense", "income_tax_expense"],
    "interest_exp":  ["interestExpense", "interest_expense", "interestAndDebtExpense"],
    # Balance Sheet
    "total_assets":  ["totalAssets", "total_assets", "TotalAssets"],
    "total_liab":    ["totalLiabilities", "total_liabilities", "TotalLiabilities"],
    "equity":        ["totalShareholderEquity", "stockholdersEquity", "total_shareholder_equity", "TotalEquity"],
    "lt_debt":       ["longTermDebtNoncurrent", "longTermDebt", "long_term_debt", "LongTermDebt"],
    "cash":          ["cashAndCashEquivalentsAtCarryingValue", "cashAndShortTermInvestments", "cash"],
    "inventory":     ["inventory", "Inventory"],
    "current_assets":["totalCurrentAssets", "current_assets"],
    "current_liab":  ["totalCurrentLiabilities", "current_liabilities"],
    "retained_earn": ["retainedEarnings", "retained_earnings"],
    "goodwill":      ["goodwill", "Goodwill"],
    # Cash Flow
    "op_cf":         ["operatingCashflow", "operating_cashflow", "netCashProvidedByOperatingActivities"],
    "capex":         ["capitalExpenditures", "capital_expenditures", "capex"],
    "cf_net_income": ["netIncome", "net_income", "profitLoss"],
    "dividends":     ["dividendPayoutCommonStock", "dividendPayout", "dividends_paid"],
    "buybacks":      ["paymentsForRepurchaseOfCommonStock", "paymentsForRepurchaseOfEquity"],
}

MAX_COMPANIES = 7

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def find_col(df, aliases):
    """Return first matching column name from aliases list, or None."""
    for a in aliases:
        if a in df.columns:
            return a
    return None

def get_col(df, key):
    """Get column values by logical key, returns Series of floats or NaN."""
    col = find_col(df, COL_MAP[key])
    if col is None:
        return pd.Series([np.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

def load_csv(path):
    """Load a CSV, return DataFrame sorted ascending by fiscal year."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Find date column
    date_col = next((c for c in df.columns if "fiscal" in c.lower() or "date" in c.lower()), None)
    if date_col:
        df["_year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
        df = df.sort_values("_year").reset_index(drop=True)
    return df

def find_files(company_dir):
    """Auto-detect income/balance/cashflow CSVs in a company folder."""
    files = {}
    for f in Path(company_dir).glob("*.csv"):
        name = f.name.lower()
        if any(k in name for k in ["income", "profit_loss", "pnl", "pl_"]):
            files["income"] = str(f)
        elif any(k in name for k in ["balance", "bs_", "balance_sheet"]):
            files["balance"] = str(f)
        elif any(k in name for k in ["cash", "cashflow", "cash_flow"]):
            files["cashflow"] = str(f)
    return files

def load_company(company_dir, name=None):
    """Load and merge all three statements for a company."""
    files = find_files(company_dir)
    missing = [k for k in ["income", "balance", "cashflow"] if k not in files]
    if missing:
        print(f"  ⚠  {name or company_dir}: missing {missing} — skipping")
        return None

    inc = load_csv(files["income"])
    bal = load_csv(files["balance"])
    cf  = load_csv(files["cashflow"])

    return {"income": inc, "balance": bal, "cashflow": cf, "name": name or Path(company_dir).name}


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL — LINEAR REGRESSION + CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

def linear_regression_with_ci(x, y, forecast_steps=3, alpha=0.10):
    """
    Fit OLS linear regression, return:
      - slope, intercept, r2
      - fitted values
      - forecast for next `forecast_steps` periods
      - 90% prediction intervals on forecast
      - t-stat and p-value for slope
    """
    # Drop NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    n = len(xc)

    if n < 3:
        return None  # not enough data

    X = xc.reshape(-1, 1)
    model = LinearRegression().fit(X, yc)
    slope     = float(model.coef_[0])
    intercept = float(model.intercept_)
    y_pred    = model.predict(X)
    r2        = r2_score(yc, y_pred)

    # Residual std error
    residuals = yc - y_pred
    s = np.sqrt(np.sum(residuals**2) / (n - 2)) if n > 2 else 0.0

    # t-stat for slope
    x_mean = xc.mean()
    sxx = np.sum((xc - x_mean)**2)
    se_slope = s / np.sqrt(sxx) if sxx > 0 else 1e-9
    t_stat = slope / se_slope if se_slope > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

    # Forecast
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    future_x = np.array([xc[-1] + i + 1 for i in range(forecast_steps)])
    forecast_y = slope * future_x + intercept

    # Prediction interval: s * sqrt(1 + 1/n + (x* - xbar)^2 / Sxx)
    pi_half = [
        t_crit * s * math.sqrt(1 + 1/n + (fx - x_mean)**2 / sxx) if sxx > 0 else 0
        for fx in future_x
    ]
    lower = forecast_y - pi_half
    upper = forecast_y + pi_half

    # CAGR
    valid_y = yc[yc > 0]
    cagr = float((valid_y[-1] / valid_y[0]) ** (1 / (len(valid_y) - 1)) - 1) \
        if len(valid_y) >= 2 else None

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "n": n,
        "t_stat": t_stat,
        "p_value": p_value,
        "se_slope": se_slope,
        "fitted": y_pred.tolist(),
        "fitted_x": xc.tolist(),
        "forecast_x": future_x.tolist(),
        "forecast_y": forecast_y.tolist(),
        "forecast_lower": lower.tolist(),
        "forecast_upper": upper.tolist(),
        "cagr": cagr,
        "trend": "upward" if slope > 0 else "downward",
        "significance": "significant" if p_value < 0.05 else "not significant",
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where((b != 0) & ~np.isnan(b) & ~np.isnan(a), a / b, np.nan)
    return result

def analyze_company(raw):
    inc = raw["income"]
    bal = raw["balance"]
    cf  = raw["cashflow"]
    name = raw["name"]

    # ── Extract core series ──────────────────────────────────────────────────
    years    = inc["_year"].values if "_year" in inc.columns else np.arange(len(inc))
    xs       = np.arange(len(years), dtype=float)

    revenue   = get_col(inc, "revenue").values
    gross     = get_col(inc, "gross_profit").values
    op_inc    = get_col(inc, "op_income").values
    net_inc   = get_col(inc, "net_income").values
    ebitda    = get_col(inc, "ebitda").values
    rd        = get_col(inc, "rd").values
    cogs      = get_col(inc, "cogs").values
    sga       = get_col(inc, "sga").values
    inc_tax   = get_col(inc, "income_tax").values
    int_exp   = get_col(inc, "interest_exp").values

    bal_years  = bal["_year"].values if "_year" in bal.columns else years
    bal_xs     = np.arange(len(bal_years), dtype=float)
    tot_assets = get_col(bal, "total_assets").values
    tot_liab   = get_col(bal, "total_liab").values
    equity     = get_col(bal, "equity").values
    lt_debt    = get_col(bal, "lt_debt").values
    cash       = get_col(bal, "cash").values
    cur_assets = get_col(bal, "current_assets").values
    cur_liab   = get_col(bal, "current_liab").values
    retained   = get_col(bal, "retained_earn").values

    cf_years = cf["_year"].values if "_year" in cf.columns else years
    cf_xs    = np.arange(len(cf_years), dtype=float)
    op_cf    = get_col(cf, "op_cf").values
    capex    = np.abs(get_col(cf, "capex").values)   # ensure positive
    dividends = get_col(cf, "dividends").values

    # ── Derived metrics ──────────────────────────────────────────────────────
    free_cf      = np.where(~np.isnan(op_cf) & ~np.isnan(capex), op_cf - capex, np.nan)
    gross_margin = safe_div(gross, revenue)
    op_margin    = safe_div(op_inc, revenue)
    net_margin   = safe_div(net_inc, revenue)
    fcf_margin   = safe_div(free_cf, revenue[:len(free_cf)])
    rd_pct       = safe_div(rd, revenue)
    roa          = safe_div(net_inc, tot_assets[:len(net_inc)])
    roe          = safe_div(net_inc, equity[:len(net_inc)])
    debt_equity  = safe_div(lt_debt, equity)
    debt_assets  = safe_div(tot_liab, tot_assets)
    current_ratio = safe_div(cur_assets, cur_liab)
    effective_tax = safe_div(inc_tax, np.where(inc_tax + net_inc != 0, inc_tax + net_inc, np.nan))
    interest_cov  = safe_div(op_inc, int_exp)

    # ── Run regressions ──────────────────────────────────────────────────────
    def reg(x, y, steps=3):
        return linear_regression_with_ci(x, y, forecast_steps=steps)

    regressions = {
        "revenue":       reg(xs, revenue),
        "net_income":    reg(xs, net_inc),
        "gross_margin":  reg(xs, gross_margin),
        "op_margin":     reg(xs, op_margin),
        "net_margin":    reg(xs, net_margin),
        "roa":           reg(bal_xs[:len(roa)], roa),
        "roe":           reg(bal_xs[:len(roe)], roe),
        "free_cf":       reg(cf_xs[:len(free_cf)], free_cf),
        "ebitda":        reg(xs, ebitda),
        "debt_equity":   reg(bal_xs, debt_equity),
        "current_ratio": reg(bal_xs[:len(current_ratio)], current_ratio),
    }

    # ── KPIs (latest values) ─────────────────────────────────────────────────
    def last(arr):
        valid = arr[~np.isnan(arr)]
        return float(valid[-1]) if len(valid) > 0 else None

    def yoy(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2: return None
        prev = float(valid[-2])
        curr = float(valid[-1])
        return (curr - prev) / abs(prev) if prev != 0 else None

    kpis = {
        "revenue":       last(revenue),
        "revenue_yoy":   yoy(revenue),
        "net_income":    last(net_inc),
        "net_income_yoy":yoy(net_inc),
        "gross_margin":  last(gross_margin),
        "op_margin":     last(op_margin),
        "net_margin":    last(net_margin),
        "roa":           last(roa),
        "roe":           last(roe),
        "debt_equity":   last(debt_equity),
        "current_ratio": last(current_ratio),
        "fcf":           last(free_cf),
        "fcf_yoy":       yoy(free_cf),
        "ebitda":        last(ebitda),
        "cash":          last(cash),
        "interest_cov":  last(interest_cov),
    }

    # ── Serialize all series to JSON-safe lists ──────────────────────────────
    def to_list(arr):
        return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                for v in arr]

    series = {
        "years":         [int(y) for y in years],
        "bal_years":     [int(y) for y in bal_years],
        "cf_years":      [int(y) for y in cf_years],
        "revenue":       to_list(revenue),
        "gross_profit":  to_list(gross),
        "op_income":     to_list(op_inc),
        "net_income":    to_list(net_inc),
        "ebitda":        to_list(ebitda),
        "rd":            to_list(rd),
        "cogs":          to_list(cogs),
        "sga":           to_list(sga),
        "gross_margin":  to_list(gross_margin),
        "op_margin":     to_list(op_margin),
        "net_margin":    to_list(net_margin),
        "fcf_margin":    to_list(fcf_margin),
        "rd_pct":        to_list(rd_pct),
        "total_assets":  to_list(tot_assets),
        "total_liab":    to_list(tot_liab),
        "equity":        to_list(equity),
        "lt_debt":       to_list(lt_debt),
        "cash":          to_list(cash),
        "retained":      to_list(retained),
        "op_cf":         to_list(op_cf),
        "capex":         to_list(capex),
        "free_cf":       to_list(free_cf),
        "dividends":     to_list(dividends),
        "roa":           to_list(roa),
        "roe":           to_list(roe),
        "debt_equity":   to_list(debt_equity),
        "debt_assets":   to_list(debt_assets),
        "current_ratio": to_list(current_ratio),
        "effective_tax": to_list(effective_tax),
        "interest_cov":  to_list(interest_cov),
    }

    # Clean regressions for JSON (remove numpy types)
    def clean_reg(r):
        if r is None: return None
        out = {}
        for k, v in r.items():
            if isinstance(v, (list, np.ndarray)):
                out[k] = [None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x) for x in v]
            elif isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = None
            else:
                out[k] = v
        return out

    return {
        "name": name,
        "kpis": kpis,
        "series": series,
        "regressions": {k: clean_reg(v) for k, v in regressions.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FinLens ML Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
:root{--bg:#070a0f;--surface:#0d1117;--surface2:#161b24;--border:#1e2530;--accent:#00e5ff;--accent2:#7c3aed;--accent3:#f59e0b;--green:#10b981;--red:#ef4444;--text:#e2e8f0;--muted:#64748b;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;min-height:100vh;}

/* ── HEADER ── */
header{padding:0 40px;display:flex;align-items:stretch;justify-content:space-between;border-bottom:1px solid var(--border);background:var(--surface);position:sticky;top:0;z-index:200;}
.logo{font-family:'Syne',sans-serif;font-weight:800;font-size:22px;color:var(--accent);display:flex;align-items:center;gap:10px;padding:18px 0;}
.logo span{color:var(--text);}
.logo-sub{font-size:12px;color:var(--muted);font-weight:400;}

/* ── PAGE TABS (main nav) ── */
.page-tabs{display:flex;align-items:stretch;gap:0;}
.page-tab{padding:0 28px;font-family:'Syne',sans-serif;font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);cursor:pointer;border-bottom:3px solid transparent;display:flex;align-items:center;gap:8px;transition:all .15s;white-space:nowrap;}
.page-tab:hover{color:var(--text);}
.page-tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.page-tab .tab-count{font-size:9px;background:var(--surface2);border:1px solid var(--border);padding:2px 6px;border-radius:2px;}
.page-tab.active .tab-count{background:rgba(0,229,255,.1);border-color:rgba(0,229,255,.3);color:var(--accent);}
.header-meta{display:flex;align-items:center;gap:10px;}
.tag{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);background:var(--surface2);padding:4px 10px;border:1px solid var(--border);}

/* ── PAGES ── */
.page{display:none;}
.page.active{display:block;}

/* ════════════════════════════════════════════
   PAGE 1 — PRELOADED COMPANIES
════════════════════════════════════════════ */
.company-nav{background:var(--surface);border-bottom:1px solid var(--border);padding:0 40px;display:flex;overflow-x:auto;}
.company-tab{padding:15px 22px;font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:all .15s;}
.company-tab:hover{color:var(--text);}
.company-tab.active{color:var(--accent);border-bottom-color:var(--accent);}

.dash-body{padding:32px 40px;}

/* KPI */
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:16px;margin-bottom:28px;}
.kpi{background:var(--surface);border:1px solid var(--border);padding:18px 16px;position:relative;overflow:hidden;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);}
.kpi.neg::before{background:var(--red);}
.kpi.warn::before{background:var(--accent3);}
.kpi-label{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;}
.kpi-value{font-family:'Syne',sans-serif;font-size:19px;font-weight:700;line-height:1;margin-bottom:6px;}
.kpi-change{font-size:11px;}
.up{color:var(--green);}.down{color:var(--red);}

/* CHARTS */
.chart-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-bottom:20px;}
.chart-grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:20px;}
.chart-card{background:var(--surface);border:1px solid var(--border);padding:22px;}
.chart-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;}
.chart-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;margin-bottom:3px;}
.chart-sub{font-size:10px;color:var(--muted);}
.chart-badge{font-size:9px;padding:3px 7px;background:rgba(0,229,255,.1);color:var(--accent);border:1px solid rgba(0,229,255,.2);letter-spacing:1px;white-space:nowrap;}
.chart-badge.purple{background:rgba(124,58,237,.1);color:#a78bfa;border-color:rgba(124,58,237,.2);}
.chart-badge.amber{background:rgba(245,158,11,.1);color:var(--accent3);border-color:rgba(245,158,11,.2);}
.chart-container{position:relative;height:210px;}
.chart-container.tall{height:270px;}

/* REGRESSION TABLE */
.reg-section{background:var(--surface);border:1px solid var(--border);margin-bottom:20px;padding:22px;}
.section-title{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;margin-bottom:4px;}
.section-sub{font-size:10px;color:var(--muted);margin-bottom:18px;letter-spacing:1px;}
table{width:100%;border-collapse:collapse;font-size:11px;}
thead th{text-align:left;padding:9px 11px;font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);font-weight:400;}
tbody tr{border-bottom:1px solid rgba(30,37,48,.5);transition:background .1s;}
tbody tr:hover{background:var(--surface2);}
tbody td{padding:10px 11px;}
td.pos{color:var(--green);}td.neg{color:var(--red);}td.neu{color:var(--accent3);}
.bar{height:4px;background:var(--accent);display:inline-block;vertical-align:middle;margin-right:6px;}
.bar.red{background:var(--red);}

/* FORECAST */
.forecast-panel{background:var(--surface2);border:1px solid var(--border);padding:22px;display:flex;gap:0;margin-bottom:28px;}
.forecast-item{flex:1;text-align:center;padding:0 16px;}
.forecast-item+.forecast-item{border-left:1px solid var(--border);}
.f-label{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;}
.f-value{font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:var(--accent);}
.f-ci{font-size:10px;color:var(--muted);margin-top:4px;}
.f-r2{font-size:10px;color:var(--muted);}
.section-divider{font-family:'Syne',sans-serif;font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--muted);padding:18px 0 10px;border-top:1px solid var(--border);margin-bottom:18px;margin-top:6px;}

/* ════════════════════════════════════════════
   PAGE 2 — UPLOAD YOUR OWN DATA
════════════════════════════════════════════ */
#page-upload{min-height:calc(100vh - 61px);}

.upload-hero{padding:52px 40px 36px;display:flex;align-items:flex-start;gap:60px;border-bottom:1px solid var(--border);}
.upload-hero-text h1{font-family:'Syne',sans-serif;font-size:36px;font-weight:800;line-height:1.1;margin-bottom:12px;}
.upload-hero-text h1 em{color:var(--accent);font-style:normal;display:block;}
.upload-hero-text p{font-size:13px;color:var(--muted);line-height:1.7;max-width:420px;}
.upload-hero-steps{display:flex;flex-direction:column;gap:12px;min-width:280px;}
.step{display:flex;align-items:flex-start;gap:14px;}
.step-num{font-family:'Syne',sans-serif;font-size:10px;font-weight:800;color:var(--accent);background:rgba(0,229,255,.1);border:1px solid rgba(0,229,255,.2);width:24px;height:24px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;}
.step-text{font-size:11px;color:var(--muted);line-height:1.5;}
.step-text strong{color:var(--text);display:block;margin-bottom:2px;}

.upload-workspace{padding:36px 40px;}
.workspace-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;}
.workspace-title{font-family:'Syne',sans-serif;font-size:14px;font-weight:800;letter-spacing:.5px;}
.workspace-actions{display:flex;gap:10px;}

.btn{padding:10px 20px;font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;cursor:pointer;transition:all .15s;border:none;}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--muted);}
.btn-outline:hover{border-color:var(--accent);color:var(--accent);}
.btn-primary{background:var(--accent);color:var(--bg);}
.btn-primary:hover{background:#fff;}
.btn-primary:disabled{background:var(--muted);cursor:not-allowed;}
.btn-danger{background:transparent;border:1px solid rgba(239,68,68,.3);color:var(--red);font-size:10px;padding:6px 12px;}
.btn-danger:hover{background:rgba(239,68,68,.1);}

/* Company cards grid */
.company-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:20px;margin-bottom:28px;}
.company-card{background:var(--surface);border:1px solid var(--border);padding:20px;position:relative;transition:border-color .15s;}
.company-card.ready{border-color:rgba(16,185,129,.3);}
.company-card-header{display:flex;align-items:center;gap:10px;margin-bottom:14px;}
.company-card-num{font-family:'Syne',sans-serif;font-size:10px;font-weight:800;color:var(--muted);background:var(--surface2);border:1px solid var(--border);width:22px;height:22px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.company-name-input{background:transparent;border:none;border-bottom:1px solid var(--border);color:var(--accent);font-family:'Syne',sans-serif;font-size:13px;font-weight:700;letter-spacing:1px;text-transform:uppercase;flex:1;padding:3px 0;outline:none;}
.company-name-input::placeholder{color:var(--muted);font-weight:400;letter-spacing:0;text-transform:none;}
.remove-card{position:absolute;top:12px;right:12px;background:transparent;border:none;color:var(--muted);cursor:pointer;font-size:14px;line-height:1;padding:2px 5px;transition:color .1s;}
.remove-card:hover{color:var(--red);}

.file-slots{display:flex;flex-direction:column;gap:8px;}
.file-slot{position:relative;border:1px dashed var(--border);padding:10px 12px;font-size:11px;color:var(--muted);cursor:pointer;transition:all .15s;display:flex;align-items:center;gap:10px;}
.file-slot:hover{border-color:var(--accent);color:var(--text);background:rgba(0,229,255,.02);}
.file-slot.loaded{border-color:var(--green);border-style:solid;color:var(--green);background:rgba(16,185,129,.03);}
.file-slot input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
.slot-icon{font-size:15px;flex-shrink:0;}
.slot-text{flex:1;overflow:hidden;}
.slot-name{font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.slot-hint{font-size:9px;color:var(--muted);margin-top:1px;letter-spacing:.5px;}
.slot-check{color:var(--green);font-size:12px;flex-shrink:0;}

.add-card-btn{border:1px dashed var(--border);background:transparent;padding:20px;display:flex;align-items:center;justify-content:center;gap:10px;color:var(--muted);cursor:pointer;font-family:'DM Mono',monospace;font-size:11px;transition:all .15s;min-height:180px;}
.add-card-btn:hover{border-color:var(--accent);color:var(--accent);}

/* Results area inside page 2 */
.upload-results{padding:0 40px 40px;display:none;}
.upload-results.visible{display:block;}
.upload-results-header{display:flex;align-items:center;justify-content:space-between;padding:20px 0;border-top:1px solid var(--border);margin-bottom:0;}
.upload-results-title{font-family:'Syne',sans-serif;font-size:13px;font-weight:800;letter-spacing:.5px;}
.upload-company-nav{display:flex;overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-bottom:none;padding:0 0 0 0;}
.upload-company-tab{padding:13px 20px;font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:all .15s;}
.upload-company-tab:hover{color:var(--text);}
.upload-company-tab.active{color:var(--accent);border-bottom-color:var(--accent);}
#uploadDashBody{background:var(--surface);border:1px solid var(--border);padding:28px;}

::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);}
</style>
</head>
<body>

<!-- ═══════════ HEADER ═══════════ -->
<header>
  <div class="logo">
    Fin<span>Lens</span>
    <span class="logo-sub">ML Analytics</span>
  </div>
  <div class="page-tabs">
    <div class="page-tab active" data-page="preloaded" onclick="switchPage('preloaded')">
      📊 Preloaded Companies
      <span class="tab-count" id="preloadCount">0</span>
    </div>
    <div class="page-tab" data-page="upload" onclick="switchPage('upload')">
      ＋ Add Your Data
      <span class="tab-count" id="uploadCount">0</span>
    </div>
  </div>
  <div class="header-meta">
    <span class="tag">Linear Regression + CI</span>
    <span class="tag">__GENERATED__</span>
  </div>
</header>

<!-- ═══════════ PAGE 1: PRELOADED ═══════════ -->
<div class="page active" id="page-preloaded">
  <div class="company-nav" id="nav"></div>
  <div class="dash-body" id="body"></div>
</div>

<!-- ═══════════ PAGE 2: UPLOAD ═══════════ -->
<div class="page" id="page-upload">

  <!-- Hero -->
  <div class="upload-hero">
    <div class="upload-hero-text">
      <h1>Analyze <em>Your Companies</em></h1>
      <p>Upload your own financial CSVs — income statement, balance sheet, and cash flow — for any company. The same ML regression models and dashboard will be generated instantly in the browser.</p>
    </div>
    <div class="upload-hero-steps">
      <div class="step">
        <div class="step-num">1</div>
        <div class="step-text"><strong>Add a company card</strong>Enter the ticker or name, then drop in the 3 CSV files.</div>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <div class="step-text"><strong>File naming</strong>Filenames must contain <code>income</code>, <code>balance</code>, or <code>cash</code> anywhere in the name.</div>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <div class="step-text"><strong>Hit Analyze</strong>Runs the full linear regression + confidence interval model right in your browser. No Python needed.</div>
      </div>
    </div>
  </div>

  <!-- Workspace -->
  <div class="upload-workspace">
    <div class="workspace-header">
      <div class="workspace-title">Company Workspace</div>
      <div class="workspace-actions">
        <button class="btn btn-outline" onclick="addCompanyCard()">+ Add Company</button>
        <button class="btn btn-primary" id="analyzeBtn" onclick="runUploadAnalysis()" disabled>▶ Run Analysis</button>
      </div>
    </div>
    <div class="company-cards" id="companyCards">
      <!-- Cards injected by JS -->
    </div>
  </div>

  <!-- Results (shown after analysis) -->
  <div class="upload-results" id="uploadResults">
    <div class="upload-results-header">
      <div class="upload-results-title">Analysis Results</div>
      <button class="btn btn-outline" onclick="resetUpload()">↺ Reset & Upload New</button>
    </div>
    <div class="upload-company-nav" id="uploadNav"></div>
    <div id="uploadDashBody"></div>
  </div>

</div><!-- end page-upload -->

<script>
// ════════════════════════════════════════════════════════
// DATA (injected by Python)
// ════════════════════════════════════════════════════════
const DATA = __DATA__;
const companies = Object.keys(DATA);
const charts = {};

// ── PAGE SWITCHING ───────────────────────────────────────
function switchPage(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.page-tab').forEach(t => t.classList.toggle('active', t.dataset.page === page));
  document.getElementById(`page-${page}`).classList.add('active');
}

// ── FORMAT UTILS ─────────────────────────────────────────
function fmt(n,unit='$'){
  if(n==null||isNaN(n))return '—';
  const a=Math.abs(n);
  let s;
  if(a>=1e12)s=(n/1e12).toFixed(2)+'T';
  else if(a>=1e9)s=(n/1e9).toFixed(2)+'B';
  else if(a>=1e6)s=(n/1e6).toFixed(1)+'M';
  else if(a>=1e3)s=(n/1e3).toFixed(1)+'K';
  else s=n.toFixed(2);
  return unit+s;
}
function pct(n,d=1){if(n==null||isNaN(n))return '—';return(n*100).toFixed(d)+'%';}
function x2(n){if(n==null||isNaN(n))return '—';return n.toFixed(2)+'x';}
function arw(v){return v==null?'—':(v>0?'▲ ':'▼ ')+pct(Math.abs(v))+' YoY';}
function arwCls(v){return v==null?'':(v>0?'up':'down');}
function trendQ(r2){
  if(r2==null)return['—',''];
  if(r2>=0.90)return['STRONG','pos'];
  if(r2>=0.70)return['MODERATE','neu'];
  if(r2>=0.50)return['WEAK','neu'];
  return['NOISY','neg'];
}

// ── CHART UTILS ───────────────────────────────────────────
const P=['#00e5ff','#7c3aed','#f59e0b','#10b981','#ef4444','#f97316','#ec4899'];
const PA=(c,a)=>c+Math.round(a*255).toString(16).padStart(2,'0');

function destroyChart(id){if(charts[id]){charts[id].destroy();delete charts[id];}}

function lineChart(id,labels,datasets,opts={}){
  destroyChart(id);
  const el=document.getElementById(id);if(!el)return;
  const ctx=el.getContext('2d');
  charts[id]=new Chart(ctx,{
    type:'line',data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:opts.legend??false,labels:{color:'#64748b',font:{family:'DM Mono',size:10}}}},
      scales:{
        x:{grid:{color:'#0f1923'},ticks:{color:'#64748b',font:{family:'DM Mono',size:10}}},
        y:{grid:{color:'#0f1923'},ticks:{color:'#64748b',font:{family:'DM Mono',size:10},callback:opts.yFmt??(v=>fmt(v))}}
      },
      elements:{point:{radius:3,hoverRadius:5}},
      ...(opts.extra||{})}
  });
}

function barChart(id,labels,datasets,opts={}){
  destroyChart(id);
  const el=document.getElementById(id);if(!el)return;
  const ctx=el.getContext('2d');
  charts[id]=new Chart(ctx,{
    type:'bar',data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:opts.legend??false,labels:{color:'#64748b',font:{family:'DM Mono',size:10}}}},
      scales:{
        x:{grid:{color:'#0f1923'},ticks:{color:'#64748b',font:{family:'DM Mono',size:10}}},
        y:{grid:{color:'#0f1923'},ticks:{color:'#64748b',font:{family:'DM Mono',size:10},callback:opts.yFmt??(v=>fmt(v))}}
      }}
  });
}

function extLabels(years,r,steps=3){
  if(!r)return years.map(String);
  const last=years[years.length-1];
  return[...years.map(String),...Array(steps).fill(0).map((_,i)=>(last+i+1)+'E')];
}
function extSeries(hist,r,steps=3){
  return[...hist,...Array(r?steps:0).fill(null)];
}
function forecastSeries(hist,r,steps=3){
  if(!r)return Array(hist.length+steps).fill(null);
  const pad=Array(hist.length-1).fill(null);
  const last=hist.filter(v=>v!=null).slice(-1)[0];
  return[...pad,last,...r.forecast_y.slice(0,steps)];
}
function ciUpper(hist,r,steps=3){
  if(!r)return Array(hist.length+steps).fill(null);
  return[...Array(hist.length).fill(null),...r.forecast_upper.slice(0,steps)];
}
function ciLower(hist,r,steps=3){
  if(!r)return Array(hist.length+steps).fill(null);
  return[...Array(hist.length).fill(null),...r.forecast_lower.slice(0,steps)];
}

// ── SHARED DASHBOARD HTML BUILDER ────────────────────────
function buildDashHTML(d, idPrefix) {
  const k=d.kpis, s=d.series, r=d.regressions;
  const years=s.years;
  const nextY=years[years.length-1]+1;
  return `
    <div class="kpi-row">
      <div class="kpi">
        <div class="kpi-label">Revenue (Latest)</div>
        <div class="kpi-value">${fmt(k.revenue)}</div>
        <div class="kpi-change ${arwCls(k.revenue_yoy)}">${arw(k.revenue_yoy)}</div>
      </div>
      <div class="kpi ${k.net_income<0?'neg':''}">
        <div class="kpi-label">Net Income</div>
        <div class="kpi-value">${fmt(k.net_income)}</div>
        <div class="kpi-change ${arwCls(k.net_income_yoy)}">${arw(k.net_income_yoy)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Gross Margin</div>
        <div class="kpi-value">${pct(k.gross_margin)}</div>
        <div class="kpi-change" style="color:var(--muted)">Op: ${pct(k.op_margin)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">ROE / ROA</div>
        <div class="kpi-value">${pct(k.roe)}</div>
        <div class="kpi-change" style="color:var(--muted)">ROA: ${pct(k.roa)}</div>
      </div>
      <div class="kpi ${k.debt_equity>2?'neg':k.debt_equity>1?'warn':''}">
        <div class="kpi-label">Debt / Equity</div>
        <div class="kpi-value">${x2(k.debt_equity)}</div>
        <div class="kpi-change" style="color:var(--muted)">Curr Ratio: ${x2(k.current_ratio)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Free Cash Flow</div>
        <div class="kpi-value">${fmt(k.fcf)}</div>
        <div class="kpi-change ${arwCls(k.fcf_yoy)}">${arw(k.fcf_yoy)}</div>
      </div>
    </div>

    <div class="forecast-panel">
      <div class="forecast-item">
        <div class="f-label">Revenue ${nextY}E</div>
        <div class="f-value">${r.revenue?fmt(r.revenue.forecast_y[0]):'—'}</div>
        <div class="f-ci">${r.revenue?'CI: '+fmt(r.revenue.forecast_lower[0])+' – '+fmt(r.revenue.forecast_upper[0]):'—'}</div>
        <div class="f-r2">R²=${r.revenue?r.revenue.r2.toFixed(4):'—'} · CAGR=${r.revenue&&r.revenue.cagr!=null?pct(r.revenue.cagr):'—'}</div>
      </div>
      <div class="forecast-item">
        <div class="f-label">Net Income ${nextY}E</div>
        <div class="f-value">${r.net_income?fmt(r.net_income.forecast_y[0]):'—'}</div>
        <div class="f-ci">${r.net_income?'CI: '+fmt(r.net_income.forecast_lower[0])+' – '+fmt(r.net_income.forecast_upper[0]):'—'}</div>
        <div class="f-r2">R²=${r.net_income?r.net_income.r2.toFixed(4):'—'} · p=${r.net_income?r.net_income.p_value.toFixed(4):'—'}</div>
      </div>
      <div class="forecast-item">
        <div class="f-label">Free Cash Flow ${nextY}E</div>
        <div class="f-value">${r.free_cf?fmt(r.free_cf.forecast_y[0]):'—'}</div>
        <div class="f-ci">${r.free_cf?'CI: '+fmt(r.free_cf.forecast_lower[0])+' – '+fmt(r.free_cf.forecast_upper[0]):'—'}</div>
        <div class="f-r2">R²=${r.free_cf?r.free_cf.r2.toFixed(4):'—'}</div>
      </div>
      <div class="forecast-item">
        <div class="f-label">EBITDA ${nextY}E</div>
        <div class="f-value">${r.ebitda?fmt(r.ebitda.forecast_y[0]):'—'}</div>
        <div class="f-ci">${r.ebitda?'CI: '+fmt(r.ebitda.forecast_lower[0])+' – '+fmt(r.ebitda.forecast_upper[0]):'—'}</div>
        <div class="f-r2">R²=${r.ebitda?r.ebitda.r2.toFixed(4):'—'}</div>
      </div>
    </div>

    <div class="section-divider">── Profitability & Growth</div>
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">Revenue & Net Income</div><div class="chart-sub">With regression + 90% confidence interval forecast</div></div><div class="chart-badge">LIN REG + CI</div></div>
        <div class="chart-container tall"><canvas id="${idPrefix}_rev"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">Profitability Margins</div><div class="chart-sub">Gross · Operating · Net — with trend lines</div></div><div class="chart-badge">MARGINS</div></div>
        <div class="chart-container tall"><canvas id="${idPrefix}_margins"></canvas></div>
      </div>
    </div>
    <div class="chart-grid-3">
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">EBITDA</div><div class="chart-sub">With forecast + CI</div></div><div class="chart-badge purple">EBITDA</div></div>
        <div class="chart-container"><canvas id="${idPrefix}_ebitda"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">ROE & ROA</div><div class="chart-sub">Return on Equity & Assets</div></div><div class="chart-badge purple">RETURNS</div></div>
        <div class="chart-container"><canvas id="${idPrefix}_returns"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">R&D Spend</div><div class="chart-sub">As % of Revenue</div></div><div class="chart-badge amber">R&D</div></div>
        <div class="chart-container"><canvas id="${idPrefix}_rd"></canvas></div>
      </div>
    </div>

    <div class="section-divider">── Cash Flow & Balance Sheet</div>
    <div class="chart-grid">
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">Free Cash Flow</div><div class="chart-sub">Operating CF · CapEx · FCF with forecast + CI</div></div><div class="chart-badge">FCF + CI</div></div>
        <div class="chart-container tall"><canvas id="${idPrefix}_fcf"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">Capital Structure</div><div class="chart-sub">Assets · Liabilities · Equity</div></div><div class="chart-badge">BALANCE SHEET</div></div>
        <div class="chart-container tall"><canvas id="${idPrefix}_capital"></canvas></div>
      </div>
    </div>
    <div class="chart-grid-3">
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">Debt / Equity</div><div class="chart-sub">Leverage trend with forecast</div></div></div>
        <div class="chart-container"><canvas id="${idPrefix}_leverage"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">Current Ratio</div><div class="chart-sub">Short-term liquidity</div></div></div>
        <div class="chart-container"><canvas id="${idPrefix}_liquidity"></canvas></div>
      </div>
      <div class="chart-card">
        <div class="chart-header"><div><div class="chart-title">Retained Earnings & Cash</div><div class="chart-sub">Value accumulation over time</div></div></div>
        <div class="chart-container"><canvas id="${idPrefix}_retained"></canvas></div>
      </div>
    </div>

    <div class="reg-section">
      <div class="section-title">ML Model — Linear Regression Statistics</div>
      <div class="section-sub">SKLEARN OLS · 90% PREDICTION INTERVALS · P-VALUE SIGNIFICANCE TESTING</div>
      <table>
        <thead>
          <tr>
            <th>Metric</th><th>Slope/yr</th><th>R²</th><th>t-stat</th><th>p-value</th>
            <th>Significance</th><th>Trend Quality</th><th>Direction</th>
            <th>${nextY}E Forecast</th><th>90% CI Lower</th><th>90% CI Upper</th><th>CAGR</th>
          </tr>
        </thead>
        <tbody id="${idPrefix}_regTable"></tbody>
      </table>
    </div>
  `;
}

function renderCharts(d, idPrefix) {
  const s=d.series, r=d.regressions;
  const years=s.years;
  const STEPS=3;
  const extY=extLabels(years,r.revenue,STEPS);
  const extBY=extLabels(s.bal_years||years,r.roa,STEPS);

  lineChart(`${idPrefix}_rev`,extY,[
    {label:'Revenue',data:extSeries(s.revenue,r.revenue,STEPS),borderColor:P[0],backgroundColor:PA(P[0],0.08),fill:true,tension:.3},
    {label:'Net Income',data:extSeries(s.net_income,r.net_income,STEPS),borderColor:P[2],tension:.3},
    {label:'Rev Forecast',data:forecastSeries(s.revenue,r.revenue,STEPS),borderColor:P[0],borderDash:[6,3],borderWidth:2,pointRadius:4,fill:false},
    {label:'NI Forecast',data:forecastSeries(s.net_income,r.net_income,STEPS),borderColor:P[2],borderDash:[6,3],borderWidth:2,pointRadius:4,fill:false},
    {label:'CI Upper',data:ciUpper(s.revenue,r.revenue,STEPS),borderColor:'transparent',backgroundColor:PA(P[0],0.12),fill:'-1',pointRadius:0,borderWidth:0},
    {label:'CI Lower',data:ciLower(s.revenue,r.revenue,STEPS),borderColor:'transparent',backgroundColor:PA(P[0],0.12),fill:false,pointRadius:0,borderWidth:0},
  ],{legend:true});

  lineChart(`${idPrefix}_margins`,extLabels(years,r.gross_margin,STEPS),[
    {label:'Gross',data:extSeries(s.gross_margin,r.gross_margin,STEPS),borderColor:P[0],tension:.3},
    {label:'Operating',data:extSeries(s.op_margin,r.op_margin,STEPS),borderColor:P[1],tension:.3},
    {label:'Net',data:extSeries(s.net_margin,r.net_margin,STEPS),borderColor:P[2],tension:.3},
    {label:'Gross Trend',data:forecastSeries(s.gross_margin,r.gross_margin,STEPS),borderColor:P[0],borderDash:[4,3],borderWidth:1.5,pointRadius:3,fill:false},
    {label:'Op Trend',data:forecastSeries(s.op_margin,r.op_margin,STEPS),borderColor:P[1],borderDash:[4,3],borderWidth:1.5,pointRadius:3,fill:false},
  ],{legend:true,yFmt:v=>(v*100).toFixed(1)+'%'});

  lineChart(`${idPrefix}_ebitda`,extLabels(years,r.ebitda,STEPS),[
    {label:'EBITDA',data:extSeries(s.ebitda,r.ebitda,STEPS),borderColor:P[1],backgroundColor:PA(P[1],0.08),fill:true,tension:.3},
    {label:'Forecast',data:forecastSeries(s.ebitda,r.ebitda,STEPS),borderColor:P[1],borderDash:[6,3],borderWidth:2,pointRadius:4,fill:false},
    {label:'CI Upper',data:ciUpper(s.ebitda,r.ebitda,STEPS),borderColor:'transparent',backgroundColor:PA(P[1],0.12),fill:'-1',pointRadius:0,borderWidth:0},
    {label:'CI Lower',data:ciLower(s.ebitda,r.ebitda,STEPS),borderColor:'transparent',backgroundColor:PA(P[1],0.12),fill:false,pointRadius:0,borderWidth:0},
  ],{legend:true});

  lineChart(`${idPrefix}_returns`,extBY,[
    {label:'ROE',data:extSeries(s.roe,r.roe,STEPS),borderColor:P[0],tension:.3},
    {label:'ROA',data:extSeries(s.roa,r.roa,STEPS),borderColor:P[3],tension:.3},
    {label:'ROE Trend',data:forecastSeries(s.roe,r.roe,STEPS),borderColor:P[0],borderDash:[4,3],borderWidth:1.5,pointRadius:3,fill:false},
  ],{legend:true,yFmt:v=>(v*100).toFixed(1)+'%'});

  lineChart(`${idPrefix}_rd`,years.map(String),[
    {label:'R&D %',data:s.rd_pct,borderColor:P[2],backgroundColor:PA(P[2],0.08),fill:true,tension:.3},
  ],{yFmt:v=>(v*100).toFixed(1)+'%'});

  barChart(`${idPrefix}_fcf`,extLabels(s.cf_years||years,r.free_cf,STEPS),[
    {label:'Op CF',data:[...s.op_cf,...Array(STEPS).fill(null)],backgroundColor:PA(P[0],0.3),borderColor:P[0],borderWidth:1},
    {label:'CapEx',data:[...s.capex.map(v=>v!=null?-v:null),...Array(STEPS).fill(null)],backgroundColor:PA(P[4],0.3),borderColor:P[4],borderWidth:1},
    {label:'Free CF',data:[...s.free_cf,...Array(STEPS).fill(null)],backgroundColor:PA(P[3],0.3),borderColor:P[3],borderWidth:1},
  ],{legend:true});

  barChart(`${idPrefix}_capital`,(s.bal_years||years).map(String),[
    {label:'Assets',data:s.total_assets,backgroundColor:PA(P[0],0.2),borderColor:P[0],borderWidth:1},
    {label:'Liabilities',data:s.total_liab,backgroundColor:PA(P[4],0.2),borderColor:P[4],borderWidth:1},
    {label:'Equity',data:s.equity,backgroundColor:PA(P[3],0.2),borderColor:P[3],borderWidth:1},
  ],{legend:true});

  lineChart(`${idPrefix}_leverage`,extBY,[
    {label:'D/E',data:extSeries(s.debt_equity,r.debt_equity,STEPS),borderColor:P[2],tension:.3},
    {label:'Forecast',data:forecastSeries(s.debt_equity,r.debt_equity,STEPS),borderColor:P[2],borderDash:[6,3],borderWidth:2,pointRadius:4,fill:false},
    {label:'CI Upper',data:ciUpper(s.debt_equity,r.debt_equity,STEPS),borderColor:'transparent',backgroundColor:PA(P[2],0.12),fill:'-1',pointRadius:0,borderWidth:0},
    {label:'CI Lower',data:ciLower(s.debt_equity,r.debt_equity,STEPS),borderColor:'transparent',backgroundColor:PA(P[2],0.12),fill:false,pointRadius:0,borderWidth:0},
  ],{yFmt:v=>v.toFixed(2)+'x'});

  lineChart(`${idPrefix}_liquidity`,extBY,[
    {label:'Current Ratio',data:extSeries(s.current_ratio,r.current_ratio,STEPS),borderColor:P[3],tension:.3},
    {label:'Forecast',data:forecastSeries(s.current_ratio,r.current_ratio,STEPS),borderColor:P[3],borderDash:[6,3],borderWidth:2,pointRadius:4,fill:false},
  ],{yFmt:v=>v.toFixed(2)+'x'});

  lineChart(`${idPrefix}_retained`,(s.bal_years||years).map(String),[
    {label:'Retained Earnings',data:s.retained,borderColor:P[5],tension:.3,fill:false},
    {label:'Cash',data:s.cash,borderColor:P[3],tension:.3,fill:false},
  ],{legend:true});

  // Regression table
  const tbody=document.getElementById(`${idPrefix}_regTable`);
  if(!tbody)return;
  const rows=[
    {name:'Revenue',reg:r.revenue,fmtFn:v=>fmt(v)},
    {name:'Net Income',reg:r.net_income,fmtFn:v=>fmt(v)},
    {name:'EBITDA',reg:r.ebitda,fmtFn:v=>fmt(v)},
    {name:'Free Cash Flow',reg:r.free_cf,fmtFn:v=>fmt(v)},
    {name:'Gross Margin',reg:r.gross_margin,fmtFn:v=>pct(v)},
    {name:'Op Margin',reg:r.op_margin,fmtFn:v=>pct(v)},
    {name:'Net Margin',reg:r.net_margin,fmtFn:v=>pct(v)},
    {name:'ROE',reg:r.roe,fmtFn:v=>pct(v)},
    {name:'ROA',reg:r.roa,fmtFn:v=>pct(v)},
    {name:'Debt / Equity',reg:r.debt_equity,fmtFn:v=>x2(v)},
    {name:'Current Ratio',reg:r.current_ratio,fmtFn:v=>x2(v)},
  ];
  tbody.innerHTML=rows.map(row=>{
    const reg=row.reg;
    if(!reg)return`<tr><td>${row.name}</td><td colspan="11" style="color:var(--muted)">Insufficient data</td></tr>`;
    const[q,qc]=trendQ(reg.r2);
    const dc=reg.slope>0?'pos':'neg';
    const pc=reg.p_value<0.05?'pos':'neg';
    const bw=Math.round(reg.r2*80);
    return`<tr>
      <td>${row.name}</td>
      <td class="${dc}">${row.fmtFn(reg.slope)}/yr</td>
      <td>${reg.r2.toFixed(4)}</td>
      <td>${reg.t_stat.toFixed(3)}</td>
      <td class="${pc}">${reg.p_value.toFixed(4)}</td>
      <td class="${pc}">${reg.p_value<0.05?'✓ SIG':'✗ NOT SIG'}</td>
      <td class="${qc}"><span class="bar ${qc==='neg'?'red':''}" style="width:${bw}px"></span>${q}</td>
      <td class="${dc}">${reg.slope>0?'↗ UP':'↘ DOWN'}</td>
      <td>${row.fmtFn(reg.forecast_y[0])}</td>
      <td style="color:var(--muted)">${row.fmtFn(reg.forecast_lower[0])}</td>
      <td style="color:var(--muted)">${row.fmtFn(reg.forecast_upper[0])}</td>
      <td class="${reg.cagr!=null&&reg.cagr>0?'pos':'neg'}">${reg.cagr!=null?pct(reg.cagr):'—'}</td>
    </tr>`;
  }).join('');
}

// ════════════════════════════════════════════════════════
// PAGE 1 — PRELOADED COMPANIES
// ════════════════════════════════════════════════════════
let activePL = null;

function buildNav(){
  const nav=document.getElementById('nav');
  nav.innerHTML=companies.map(id=>`
    <div class="company-tab" data-id="${id}" onclick="renderCompany('${id}')">${DATA[id].name}</div>
  `).join('');
  document.getElementById('preloadCount').textContent=companies.length;
}

function renderCompany(id){
  activePL=id;
  document.querySelectorAll('.company-tab').forEach(t=>t.classList.toggle('active',t.dataset.id===id));
  const d=DATA[id];
  const body=document.getElementById('body');
  body.innerHTML=buildDashHTML(d,'pl');
  setTimeout(()=>renderCharts(d,'pl'),60);
}

// ════════════════════════════════════════════════════════
// PAGE 2 — UPLOAD YOUR OWN DATA
// ════════════════════════════════════════════════════════
const uploadStore={};  // { cardId: { income, balance, cashflow, name } }
let cardCount=0;
let activeUpload=null;

const SLOT_TYPES=[
  {key:'income',  icon:'📊',label:'Income Statement', hint:'contains "income"'},
  {key:'balance', icon:'🏦',label:'Balance Sheet',    hint:'contains "balance"'},
  {key:'cashflow',icon:'💵',label:'Cash Flow',        hint:'contains "cash"'},
];

function addCompanyCard(){
  cardCount++;
  const id=`card${cardCount}`;
  uploadStore[id]={};

  const grid=document.getElementById('companyCards');

  const card=document.createElement('div');
  card.className='company-card';
  card.id=`cc_${id}`;
  card.innerHTML=`
    <button class="remove-card" onclick="removeCard('${id}')" title="Remove">✕</button>
    <div class="company-card-header">
      <div class="company-card-num">${cardCount}</div>
      <input class="company-name-input" id="cname_${id}" type="text"
        placeholder="Ticker / Company name" oninput="checkAnalyzeBtn()">
    </div>
    <div class="file-slots">
      ${SLOT_TYPES.map(t=>`
        <div class="file-slot" id="fslot_${id}_${t.key}">
          <input type="file" accept=".csv"
            data-card="${id}" data-type="${t.key}"
            onchange="handleCardFile(this)">
          <span class="slot-icon">${t.icon}</span>
          <div class="slot-text">
            <div class="slot-name" id="fname_${id}_${t.key}">${t.label}</div>
            <div class="slot-hint">${t.hint}</div>
          </div>
          <span class="slot-check" id="fcheck_${id}_${t.key}" style="display:none">✓</span>
        </div>`).join('')}
    </div>
  `;
  grid.appendChild(card);
  checkAnalyzeBtn();
}

function removeCard(id){
  const el=document.getElementById(`cc_${id}`);
  if(el)el.remove();
  delete uploadStore[id];
  checkAnalyzeBtn();
}

function handleCardFile(input){
  const {card,type}=input.dataset;
  const file=input.files[0];
  if(!file)return;

  const reader=new FileReader();
  reader.onload=e=>{
    uploadStore[card][type]=parseBrowserCSV(e.target.result);

    document.getElementById(`fslot_${card}_${type}`).classList.add('loaded');
    const short=file.name.length>26?file.name.substring(0,24)+'…':file.name;
    document.getElementById(`fname_${card}_${type}`).textContent=short;
    document.getElementById(`fcheck_${card}_${type}`).style.display='';

    const nameEl=document.getElementById(`cname_${card}`);
    if(!nameEl.value){
      nameEl.value=file.name
        .replace(/[_-]?(income|balance|cash).*/i,'')
        .replace(/[_-]/g,' ').toUpperCase().trim().substring(0,10);
    }

    // Mark card as ready if all 3 loaded
    const s=uploadStore[card];
    if(s.income&&s.balance&&s.cashflow){
      document.getElementById(`cc_${card}`).classList.add('ready');
    }
    checkAnalyzeBtn();
  };
  reader.readAsText(file);
}

function parseBrowserCSV(text){
  const lines=text.trim().split('\n');
  const headers=lines[0].split(',').map(h=>h.trim().replace(/"/g,''));
  return lines.slice(1).map(line=>{
    const vals=line.split(',');
    const obj={};
    headers.forEach((h,i)=>{
      const v=(vals[i]||'').trim().replace(/"/g,'');
      obj[h]=(!v||v==='nan'||v==='None'||v==='null')?null:isNaN(v)?v:parseFloat(v);
    });
    return obj;
  }).filter(r=>r[headers[0]]);
}

function checkAnalyzeBtn(){
  const anyReady=Object.values(uploadStore).some(s=>s.income&&s.balance&&s.cashflow);
  document.getElementById('analyzeBtn').disabled=!anyReady;
}

function analyzeInBrowser(files,name){
  const sortByYear=rows=>[...rows].sort((a,b)=>{
    const ya=parseInt((a.fiscalDateEnding||'').substring(0,4));
    const yb=parseInt((b.fiscalDateEnding||'').substring(0,4));
    return ya-yb;
  });

  const inc=sortByYear(files.income);
  const bal=sortByYear(files.balance);
  const cf =sortByYear(files.cashflow);

  const COL={
    revenue:    ['totalRevenue','revenue'],
    gross:      ['grossProfit','gross_profit'],
    op_income:  ['operatingIncome','operating_income'],
    net_income: ['netIncome','net_income','netIncomeFromContinuingOperations'],
    ebitda:     ['ebitda','EBITDA'],
    rd:         ['researchAndDevelopment','research_and_development'],
    tot_assets: ['totalAssets','total_assets'],
    tot_liab:   ['totalLiabilities','total_liabilities'],
    equity:     ['totalShareholderEquity','stockholdersEquity'],
    lt_debt:    ['longTermDebtNoncurrent','longTermDebt'],
    cash:       ['cashAndCashEquivalentsAtCarryingValue','cashAndShortTermInvestments'],
    cur_assets: ['totalCurrentAssets'],
    cur_liab:   ['totalCurrentLiabilities'],
    retained:   ['retainedEarnings'],
    op_cf:      ['operatingCashflow','netCashProvidedByOperatingActivities'],
    capex:      ['capitalExpenditures'],
    dividends:  ['dividendPayoutCommonStock','dividendPayout'],
  };

  function getCol(rows,key){
    const aliases=COL[key]||[key];
    const col=aliases.find(a=>rows[0]&&a in rows[0]);
    return rows.map(r=>col?(r[col]==null?null:parseFloat(r[col])):null);
  }

  function sdiv(a,b){return(b&&b!==0&&!isNaN(b)&&!isNaN(a)&&a!=null)?a/b:null;}
  function toList(arr){return arr.map(v=>(v==null||isNaN(v))?null:v);}

  const years=inc.map(r=>parseInt((r.fiscalDateEnding||'').substring(0,4))).filter(Boolean);
  const xs=years.map((_,i)=>i);

  const revenue   =getCol(inc,'revenue');
  const gross     =getCol(inc,'gross');
  const op_inc    =getCol(inc,'op_income');
  const net_inc   =getCol(inc,'net_income');
  const ebitda    =getCol(inc,'ebitda');
  const rd        =getCol(inc,'rd');
  const tot_assets=getCol(bal,'tot_assets');
  const tot_liab  =getCol(bal,'tot_liab');
  const equity    =getCol(bal,'equity');
  const lt_debt   =getCol(bal,'lt_debt');
  const cash      =getCol(bal,'cash');
  const cur_assets=getCol(bal,'cur_assets');
  const cur_liab  =getCol(bal,'cur_liab');
  const retained  =getCol(bal,'retained');
  const op_cf     =getCol(cf,'op_cf');
  const capex     =getCol(cf,'capex').map(v=>v!=null?Math.abs(v):null);
  const dividends =getCol(cf,'dividends');

  const free_cf      =op_cf.map((o,i)=>(o!=null&&capex[i]!=null)?o-capex[i]:null);
  const gross_margin =revenue.map((r,i)=>sdiv(gross[i],r));
  const op_margin    =revenue.map((r,i)=>sdiv(op_inc[i],r));
  const net_margin   =revenue.map((r,i)=>sdiv(net_inc[i],r));
  const roa          =net_inc.map((n,i)=>sdiv(n,tot_assets[i]));
  const roe          =net_inc.map((n,i)=>sdiv(n,equity[i]));
  const debt_equity  =lt_debt.map((d,i)=>sdiv(d,equity[i]));
  const cur_ratio    =cur_assets.map((c,i)=>sdiv(c,cur_liab[i]));
  const rd_pct       =rd.map((r,i)=>sdiv(r,revenue[i]));
  const fcf_margin   =free_cf.map((f,i)=>sdiv(f,revenue[i]));

  function jsReg(xsArr,ysArr,steps=3){
    const pairs=xsArr.map((x,i)=>[x,ysArr[i]]).filter(([a,b])=>a!=null&&b!=null&&!isNaN(a)&&!isNaN(b));
    if(pairs.length<3)return null;
    const xv=pairs.map(p=>p[0]),yv=pairs.map(p=>p[1]),n=pairs.length;
    const sx=xv.reduce((a,b)=>a+b,0),sy=yv.reduce((a,b)=>a+b,0);
    const sxy=xv.reduce((a,b,i)=>a+b*yv[i],0),sxx=xv.reduce((a,b)=>a+b*b,0);
    const denom=(n*sxx-sx*sx)||1e-9;
    const slope=(n*sxy-sx*sy)/denom;
    const intercept=(sy-slope*sx)/n;
    const xm=sx/n,ym=sy/n;
    const ssTot=yv.reduce((a,b)=>a+(b-ym)**2,0);
    const fitted=xv.map(x=>slope*x+intercept);
    const residuals=yv.map((y,i)=>y-fitted[i]);
    const ssRes=residuals.reduce((a,b)=>a+b*b,0);
    const r2=ssTot===0?1:1-ssRes/ssTot;
    const s=Math.sqrt(ssRes/Math.max(n-2,1));
    const Sxx=xv.reduce((a,b)=>a+(b-xm)**2,0);
    const se_slope=Sxx>0?s/Math.sqrt(Sxx):1e-9;
    const t_stat=slope/se_slope;
    const p_value=Math.min(1,2*Math.exp(-0.717*Math.abs(t_stat)-0.416*t_stat*t_stat));
    const t_crit=1.833;
    const future_x=Array.from({length:steps},(_,i)=>xv[xv.length-1]+i+1);
    const forecast_y=future_x.map(x=>slope*x+intercept);
    const pi_half=future_x.map(x=>t_crit*s*Math.sqrt(1+1/n+(x-xm)**2/Math.max(Sxx,1e-9)));
    const validPos=yv.filter(v=>v>0);
    const cagr=validPos.length>=2?Math.pow(validPos[validPos.length-1]/validPos[0],1/(validPos.length-1))-1:null;
    return{slope,intercept,r2,t_stat,p_value,se_slope,fitted,fitted_x:xv,
      forecast_x:future_x,forecast_y,
      forecast_lower:forecast_y.map((y,i)=>y-pi_half[i]),
      forecast_upper:forecast_y.map((y,i)=>y+pi_half[i]),
      cagr,trend:slope>0?'upward':'downward',
      significance:p_value<0.05?'significant':'not significant'};
  }

  function reg(vals){return jsReg(xs.slice(0,vals.length),vals);}
  const last=arr=>{const v=arr.filter(x=>x!=null);return v.length?v[v.length-1]:null;};
  const yoy=arr=>{const v=arr.filter(x=>x!=null);if(v.length<2)return null;const l=v[v.length-1],p=v[v.length-2];return p?((l-p)/Math.abs(p)):null;};

  return{
    name,
    kpis:{
      revenue:last(revenue),revenue_yoy:yoy(revenue),
      net_income:last(net_inc),net_income_yoy:yoy(net_inc),
      gross_margin:last(gross_margin),op_margin:last(op_margin),net_margin:last(net_margin),
      roa:last(roa),roe:last(roe),
      debt_equity:last(debt_equity),current_ratio:last(cur_ratio),
      fcf:last(free_cf),fcf_yoy:yoy(free_cf),
      ebitda:last(ebitda),cash:last(cash),interest_cov:null,
    },
    series:{
      years,bal_years:years,cf_years:years,
      revenue:toList(revenue),gross_profit:toList(gross),
      op_income:toList(op_inc),net_income:toList(net_inc),
      ebitda:toList(ebitda),rd:toList(rd),cogs:[],sga:[],
      gross_margin:toList(gross_margin),op_margin:toList(op_margin),
      net_margin:toList(net_margin),fcf_margin:toList(fcf_margin),rd_pct:toList(rd_pct),
      total_assets:toList(tot_assets),total_liab:toList(tot_liab),
      equity:toList(equity),lt_debt:toList(lt_debt),cash:toList(cash),
      retained:toList(retained),op_cf:toList(op_cf),capex:toList(capex),
      free_cf:toList(free_cf),dividends:toList(dividends),
      roa:toList(roa),roe:toList(roe),
      debt_equity:toList(debt_equity),debt_assets:[],
      current_ratio:toList(cur_ratio),effective_tax:[],interest_cov:[],
    },
    regressions:{
      revenue:reg(revenue),net_income:reg(net_inc),
      gross_margin:reg(gross_margin),op_margin:reg(op_margin),net_margin:reg(net_margin),
      roa:reg(roa),roe:reg(roe),free_cf:reg(free_cf),
      ebitda:reg(ebitda),debt_equity:reg(debt_equity),current_ratio:reg(cur_ratio),
    },
  };
}

// Upload results store
const uploadResults={};

function runUploadAnalysis(){
  let analyzed=0;
  Object.entries(uploadStore).forEach(([cardId,files])=>{
    if(!files.income||!files.balance||!files.cashflow)return;
    const name=document.getElementById(`cname_${cardId}`)?.value?.trim()||cardId;
    uploadResults[cardId]=analyzeInBrowser(files,name);
    analyzed++;
  });

  if(!analyzed)return;

  // Show results section
  const resultsEl=document.getElementById('uploadResults');
  resultsEl.classList.add('visible');
  document.getElementById('uploadCount').textContent=analyzed;

  // Build nav tabs
  const nav=document.getElementById('uploadNav');
  nav.innerHTML=Object.entries(uploadResults).map(([id,d])=>`
    <div class="upload-company-tab" data-uid="${id}" onclick="showUploadCompany('${id}')">${d.name}</div>
  `).join('');

  // Show first
  const firstId=Object.keys(uploadResults)[0];
  showUploadCompany(firstId);

  // Scroll to results
  resultsEl.scrollIntoView({behavior:'smooth',block:'start'});
}

function showUploadCompany(id){
  activeUpload=id;
  document.querySelectorAll('.upload-company-tab').forEach(t=>t.classList.toggle('active',t.dataset.uid===id));
  const d=uploadResults[id];
  const body=document.getElementById('uploadDashBody');
  body.innerHTML=buildDashHTML(d,'up');
  setTimeout(()=>renderCharts(d,'up'),60);
}

function resetUpload(){
  document.getElementById('uploadResults').classList.remove('visible');
  document.getElementById('uploadDashBody').innerHTML='';
  document.getElementById('uploadNav').innerHTML='';
  document.getElementById('uploadCount').textContent='0';
  Object.keys(uploadResults).forEach(k=>delete uploadResults[k]);
}

// ── BOOT ────────────────────────────────────────────────
// Start with one empty company card on page 2
addCompanyCard();
// Build page 1
buildNav();
if(companies.length>0) renderCompany(companies[0]);
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def discover_companies(data_dir):
    """Find all company subdirectories with at least one CSV."""
    base = Path(data_dir)
    if not base.exists():
        return []
    return [d for d in sorted(base.iterdir()) if d.is_dir()]

def main():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  FinLens — Financial ML Analyzer")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    dirs = discover_companies(DATA_DIR)
    if not dirs:
        print(f"❌  No company folders found in '{DATA_DIR}/'")
        print(f"\nExpected structure:")
        print(f"  {DATA_DIR}/")
        print(f"    AAPL/")
        print(f"      income_statement.csv")
        print(f"      balance_sheet.csv")
        print(f"      cashflow_statement.csv")
        print(f"    MSFT/")
        print(f"      ...")
        sys.exit(1)

    all_results = {}
    loaded = 0

    for company_dir in dirs[:MAX_COMPANIES]:
        name = company_dir.name
        print(f"  📂  Loading {name}...")
        raw = load_company(str(company_dir), name)
        if raw is None:
            continue
        print(f"      → Analyzing {name} ({len(raw['income'])} years of data)...")
        result = analyze_company(raw)
        all_results[f"company{loaded+1}"] = result
        loaded += 1
        rev_r2 = result['regressions']['revenue']['r2'] if result['regressions']['revenue'] else 0
        print(f"      ✓  Done  |  Revenue R²={rev_r2:.4f}")

    if not all_results:
        print("\n❌  No companies successfully loaded.")
        sys.exit(1)

    print(f"\n  📊  Generating report for {loaded} company/companies...")

    # Inject data into HTML
    from datetime import datetime
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_json = json.dumps(all_results, allow_nan=False)
    html = HTML_TEMPLATE.replace("__DATA__", data_json)
    html = html.replace("__GENERATED__", f"Generated {generated}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✅  Report saved → {OUTPUT}")
    print(f"\n  Open {OUTPUT} in any browser to view the dashboard.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


if __name__ == "__main__":
    main()

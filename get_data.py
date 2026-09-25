import requests
import pandas as pd
from google.colab import files

ALPHA_KEY = "(API)"  # <-- Replace with your Alpha Vantage API key
SYMBOL = "NVDA"

def get_financials(function_name, symbol, key, filename):
    url = f"https://www.alphavantage.co/query?function={function_name}&symbol={symbol}&apikey={key}"
    data = requests.get(url).json()
    if 'annualReports' in data:
        df = pd.DataFrame(data['annualReports'])
        df.columns = [c.strip() for c in df.columns]
        df.to_csv(filename, index=False)
        print(f"{filename} saved!")
    else:
        print(f"Error fetching {function_name}: {data}")
    return

get_financials("BALANCE_SHEET", SYMBOL, ALPHA_KEY, "nvda_balance_sheet.csv")

get_financials("INCOME_STATEMENT", SYMBOL, ALPHA_KEY, "nvda_income_statement.csv")

get_financials("CASH_FLOW", SYMBOL, ALPHA_KEY, "nvda_cashflow_statement.csv")

files.download(f"{SYMBOL}_balance_sheet.csv")
files.download(f"{SYMBOL}_income_statement.csv")
files.download(f"{SYMBOL}_cashflow_statement.csv")

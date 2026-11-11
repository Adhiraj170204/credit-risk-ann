"""Download the Credit Scoring dataset (4,446 loan applicants, Spanish bank) into data/."""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/gastonstat/CreditScoring/master/CleanCreditScoring.csv"
TARGET = Path(__file__).resolve().parents[1] / "data" / "CreditScoring.csv"

if __name__ == "__main__":
    TARGET.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, TARGET)
    print(f"Saved {TARGET}")

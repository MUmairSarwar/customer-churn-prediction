from pathlib import Path
from urllib.request import urlretrieve


URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, DESTINATION)
    print(f"Downloaded dataset to {DESTINATION}")


if __name__ == "__main__":
    main()

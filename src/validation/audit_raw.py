import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "train.csv"
REPORT_DIR = ROOT / "reports"
REPORT_FILE = REPORT_DIR / "raw_audit.json"

def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    order_dates = pd.to_datetime(
        df["Order Date"], format="%d/%m/%Y", errors="coerce"
    )
    ship_dates = pd.to_datetime(
        df["Ship Date"], format="%d/%m/%Y", errors="coerce"
    )

    postal_numeric = pd.to_numeric(df["Postal Code"], errors="coerce")
    postal_valid = (
        postal_numeric.notna()
        & postal_numeric.between(0, 99999)
        & (postal_numeric % 1 == 0)
    )

    duplicate_count = int(
        df.duplicated(
            subset=[col for col in df.columns if col != "Row ID"]
        ).sum()
    )

    product_name_counts = df.groupby("Product ID")["Product Name"].nunique()
    shipping_delay = (ship_dates - order_dates).dt.days

    q1 = df["Sales"].quantile(0.25)
    q3 = df["Sales"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    report = {
        "source_file": str(RAW_FILE),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows_excluding_row_id": duplicate_count,
        "postal_codes_not_five_digits": int((~postal_valid).sum()),
        "missing_postal_codes": int(df["Postal Code"].isna().sum()),
        "product_ids_with_multiple_names": int((product_name_counts > 1).sum()),
        "invalid_order_dates": int(order_dates.isna().sum()),
        "invalid_ship_dates": int(ship_dates.isna().sum()),
        "negative_shipping_delays": int((shipping_delay < 0).sum()),
        "sales_total": round(float(df["Sales"].sum()), 4),
        "iqr_outlier_rows": int((df["Sales"] > upper_bound).sum()),
        "iqr_upper_bound": round(float(upper_bound), 4)
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Audited: {RAW_FILE}")
    print(f"Rows x columns: {report['rows']} x {report['columns']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Duplicate rows excluding Row ID: {report['duplicate_rows_excluding_row_id']}")
    print(f"Postal codes not five digits: {report['postal_codes_not_five_digits']}")
    print(f"Missing postal codes: {report['missing_postal_codes']}")
    print(f"Product IDs with multiple names: {report['product_ids_with_multiple_names']}")
    print(f"Invalid order dates: {report['invalid_order_dates']}")
    print(f"Invalid ship dates: {report['invalid_ship_dates']}")
    print(f"Negative shipping delays: {report['negative_shipping_delays']}")
    print(f"Sales total: {report['sales_total']:.4f}")
    print(f"IQR outlier rows: {report['iqr_outlier_rows']}")
    print(f"Report written to: {REPORT_FILE}")

if __name__ == "__main__":
    main()

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "train.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "train_clean.csv"
REPORT_FILE = ROOT / "reports" / "cleaning_report.json"


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    raw_rows = len(df)
    raw_sales_total = float(pd.to_numeric(df["Sales"], errors="raise").sum())

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    duplicate_columns = [col for col in df.columns if col != "Row ID"]
    duplicate_mask = df.duplicated(subset=duplicate_columns)
    removed_duplicates = int(duplicate_mask.sum())
    removed_duplicate_sales = float(df.loc[duplicate_mask, "Sales"].sum())

    df = df.drop_duplicates(subset=duplicate_columns, keep="first").copy()

    postal = pd.to_numeric(df["Postal Code"], errors="coerce")
    postal_invalid = (
        df["Postal Code"].notna()
        & (
            postal.isna()
            | (postal < 0)
            | (postal > 99999)
            | (postal % 1 != 0)
        )
    )

    if postal_invalid.any():
        raise ValueError(
            f"Found {int(postal_invalid.sum())} invalid non-missing postal codes."
        )

    missing_postal = int(postal.isna().sum())

    df["Postal Code"] = (
        postal.astype("Int64")
        .astype("string")
        .str.zfill(5)
        .fillna("Unknown")
    )

    for col in ["Order Date", "Ship Date"]:
        df[col] = pd.to_datetime(
            df[col],
            format="%d/%m/%Y",
            errors="coerce"
        )

        if df[col].isna().any():
            raise ValueError(f"Invalid or missing dates found in {col}.")

    df["Sales"] = pd.to_numeric(df["Sales"], errors="raise")

    if df["Sales"].isna().any() or (df["Sales"] <= 0).any():
        raise ValueError("Sales contains missing or non-positive values.")

    if df[["Product ID", "Product Name"]].isna().any().any():
        raise ValueError("Missing product identifiers or product names.")

    product_pairs = sorted(
        set(zip(df["Product ID"], df["Product Name"]))
    )

    product_mapping = {
        pair: index + 1
        for index, pair in enumerate(product_pairs)
    }

    df["Product Key"] = [
        product_mapping[(product_id, product_name)]
        for product_id, product_name
        in zip(df["Product ID"], df["Product Name"])
    ]

    q1 = df["Sales"].quantile(0.25)
    q3 = df["Sales"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    df["Sales Outlier"] = df["Sales"] > upper_bound
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Shipping Delay"] = (
        df["Ship Date"] - df["Order Date"]
    ).dt.days

    if (df["Shipping Delay"] < 0).any():
        raise ValueError("Negative shipping delays detected.")

    if df.isna().any().any():
        missing = df.isna().sum()
        raise ValueError(
            f"Unexpected missing values:\n{missing[missing > 0]}"
        )

    if df.duplicated(subset=duplicate_columns).any():
        raise ValueError("Duplicate records remain after cleaning.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    report = {
        "source_file": str(RAW_FILE),
        "output_file": str(OUTPUT_FILE),
        "raw_rows": raw_rows,
        "cleaned_rows": len(df),
        "columns": len(df.columns),
        "removed_duplicates": removed_duplicates,
        "removed_duplicate_sales": round(removed_duplicate_sales, 4),
        "missing_postal_codes_replaced": missing_postal,
        "remaining_missing_values": int(df.isna().sum().sum()),
        "remaining_duplicate_rows": int(
            df.duplicated(subset=duplicate_columns).sum()
        ),
        "product_id_name_combinations": len(product_pairs),
        "product_ids_with_multiple_names": int(
            (df.groupby("Product ID")["Product Name"].nunique() > 1).sum()
        ),
        "sales_outliers_flagged": int(df["Sales Outlier"].sum()),
        "sales_outliers_removed": 0,
        "iqr_upper_bound": round(float(upper_bound), 4),
        "raw_sales_total": round(raw_sales_total, 4),
        "cleaned_sales_total": round(float(df["Sales"].sum()), 4),
        "cleaning_validation": "PASSED"
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("DATA CLEANING COMPLETED")
    print("-" * 45)
    print(f"Raw rows: {report['raw_rows']}")
    print(f"Cleaned rows: {report['cleaned_rows']}")
    print(f"Columns: {report['columns']}")
    print(f"Removed duplicates: {report['removed_duplicates']}")
    print(f"Missing postal codes replaced: {report['missing_postal_codes_replaced']}")
    print(f"Remaining missing values: {report['remaining_missing_values']}")
    print(f"Product combinations: {report['product_id_name_combinations']}")
    print(f"Sales outliers flagged: {report['sales_outliers_flagged']}")
    print(f"Sales outliers removed: {report['sales_outliers_removed']}")
    print(f"Raw sales total: {report['raw_sales_total']:.4f}")
    print(f"Cleaned sales total: {report['cleaned_sales_total']:.4f}")
    print(f"Validation: {report['cleaning_validation']}")
    print(f"Clean dataset: {OUTPUT_FILE}")
    print(f"Cleaning report: {REPORT_FILE}")


if __name__ == "__main__":
    main()

from pathlib import Path
import getpass
import sys

import pandas as pd
import psycopg


ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = ROOT / "data" / "processed" / "train_clean.csv"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
}


def date_key(value):
    return int(value.strftime("%Y%m%d"))


def main():
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    df = pd.read_csv(CSV_FILE)

    required = [
        "Row ID", "Order ID", "Order Date", "Ship Date",
        "Customer ID", "Customer Name", "Segment",
        "Product ID", "Product Name", "Category", "Sub-Category",
        "Country", "Region", "State", "City", "Postal Code",
        "Ship Mode", "Sales", "Shipping Delay", "Sales Outlier",
    ]

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"], errors="raise"
    ).dt.normalize()

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"], errors="raise"
    ).dt.normalize()

    if df["Row ID"].duplicated().any():
        raise ValueError("Duplicate Row ID detected.")

    if df[required].isna().any().any():
        raise ValueError("Missing values detected in required columns.")

    if (df["Sales"] <= 0).any():
        raise ValueError("Non-positive sales detected.")

    if (df["Shipping Delay"] < 0).any():
        raise ValueError("Negative shipping delay detected.")

    password = getpass.getpass("PostgreSQL password: ")
    DB_CONFIG["password"] = password

    print(f"Source rows: {len(df)}")
    print("Connecting to PostgreSQL...")

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM sales_dw.fact_sales;
                """)
                existing_fact_rows = cur.fetchone()[0]

                if existing_fact_rows > 0:
                    raise RuntimeError(
                        "fact_sales is not empty. "
                        "Loading stopped to avoid duplicate data."
                    )

                cur.execute("""
                    SELECT COUNT(*)
                    FROM sales_dw.dim_customer;
                """)
                if cur.fetchone()[0] > 0:
                    raise RuntimeError(
                        "Dimensions are not empty. "
                        "Loading stopped to avoid duplicate data."
                    )

                # 1. Date dimension: include both order and ship dates.
                dates = sorted(set(df["Order Date"]) | set(df["Ship Date"]))

                for dt in dates:
                    pydate = dt.date()
                    key = date_key(pydate)
                    quarter = (pydate.month - 1) // 3 + 1

                    cur.execute("""
                        INSERT INTO sales_dw.dim_date (
                            date_key, full_date, year, quarter,
                            month, month_name, day, day_of_week,
                            day_name, is_weekend
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        );
                    """, (
                        key,
                        pydate,
                        pydate.year,
                        quarter,
                        pydate.month,
                        pydate.strftime("%B"),
                        pydate.day,
                        pydate.isoweekday(),
                        pydate.strftime("%A"),
                        pydate.isoweekday() >= 6,
                    ))

                # 2. Customer dimension.
                customers = (
                    df[["Customer ID", "Customer Name", "Segment"]]
                    .drop_duplicates()
                    .sort_values("Customer ID")
                )

                for row in customers.itertuples(index=False, name=None):
                    cur.execute("""
                        INSERT INTO sales_dw.dim_customer (
                            customer_id, customer_name, segment
                        )
                        VALUES (%s, %s, %s);
                    """, row)

                # 3. Product dimension: business key is Product ID + Name.
                products = (
                    df[[
                        "Product ID", "Product Name",
                        "Category", "Sub-Category"
                    ]]
                    .drop_duplicates()
                    .sort_values(["Product ID", "Product Name"])
                )

                for row in products.itertuples(index=False, name=None):
                    cur.execute("""
                        INSERT INTO sales_dw.dim_product (
                            product_id, product_name, category, sub_category
                        )
                        VALUES (%s, %s, %s, %s);
                    """, row)

                # 4. Geography dimension.
                geography_columns = [
                    "Country", "Region", "State", "City", "Postal Code"
                ]
                geographies = (
                    df[geography_columns]
                    .drop_duplicates()
                    .sort_values(geography_columns)
                )

                for row in geographies.itertuples(index=False, name=None):
                    cur.execute("""
                        INSERT INTO sales_dw.dim_geography (
                            country, region, state, city, postal_code
                        )
                        VALUES (%s, %s, %s, %s, %s);
                    """, tuple(str(value) for value in row))

                # 5. Shipping mode dimension.
                ship_modes = sorted(df["Ship Mode"].unique())

                for mode in ship_modes:
                    cur.execute("""
                        INSERT INTO sales_dw.dim_ship_mode (ship_mode)
                        VALUES (%s);
                    """, (mode,))

                # Build surrogate-key lookups from PostgreSQL.
                cur.execute("""
                    SELECT customer_id, customer_key
                    FROM sales_dw.dim_customer;
                """)
                customer_keys = dict(cur.fetchall())

                cur.execute("""
                    SELECT product_id, product_name, product_key
                    FROM sales_dw.dim_product;
                """)
                product_keys = {
                    (pid, pname): key
                    for pid, pname, key in cur.fetchall()
                }

                cur.execute("""
                    SELECT country, region, state, city, postal_code,
                           geography_key
                    FROM sales_dw.dim_geography;
                """)
                geography_keys = {
                    (country, region, state, city, postal): key
                    for country, region, state, city, postal, key
                    in cur.fetchall()
                }

                cur.execute("""
                    SELECT ship_mode, ship_mode_key
                    FROM sales_dw.dim_ship_mode;
                """)
                ship_mode_keys = dict(cur.fetchall())

                # 6. Fact table.
                fact_rows = []

                for record in df.to_dict(orient="records"):

                    order_date = pd.Timestamp(record["Order Date"]).date()
                    ship_date = pd.Timestamp(record["Ship Date"]).date()

                    postal_code = str(record["Postal Code"])

                    customer_key = customer_keys[record["Customer ID"]]
                    product_key = product_keys[
                        (record["Product ID"], record["Product Name"])
                    ]

                    geography_key = geography_keys[(
                        record["Country"],
                        record["Region"],
                        record["State"],
                        record["City"],
                        postal_code,
                    )]

                    ship_mode_key = ship_mode_keys[record["Ship Mode"]]

                    fact_rows.append((
                        int(record["Row ID"]),
                        record["Order ID"],
                        date_key(order_date),
                        date_key(ship_date),
                        customer_key,
                        product_key,
                        geography_key,
                        ship_mode_key,
                        float(record["Sales"]),
                        int(record["Shipping Delay"]),
                        bool(record["Sales Outlier"]),
                    ))

                cur.executemany("""
                    INSERT INTO sales_dw.fact_sales (
                        source_line_id, order_id,
                        order_date_key, ship_date_key,
                        customer_key, product_key,
                        geography_key, ship_mode_key,
                        sales, shipping_delay, sales_outlier
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    );
                """, fact_rows)

                # Verify counts and sales before transaction commits.
                expected_counts = {
                    "dim_date": len(dates),
                    "dim_customer": len(customers),
                    "dim_product": len(products),
                    "dim_geography": len(geographies),
                    "dim_ship_mode": len(ship_modes),
                    "fact_sales": len(df),
                }

                print("\n=== LOAD VALIDATION ===")
                for table, expected in expected_counts.items():
                    cur.execute(
                        f"SELECT COUNT(*) FROM sales_dw.{table};"
                    )
                    actual = cur.fetchone()[0]
                    print(f"{table}: {actual} rows (expected {expected})")

                    if actual != expected:
                        raise RuntimeError(
                            f"Row count mismatch for {table}: "
                            f"{actual} != {expected}"
                        )

                cur.execute("""
                    SELECT COUNT(*), SUM(sales),
                           COUNT(DISTINCT source_line_id)
                    FROM sales_dw.fact_sales;
                """)
                fact_count, total_sales, unique_source_lines = cur.fetchone()

                expected_sales = round(float(df["Sales"].sum()), 4)
                actual_sales = round(float(total_sales), 4)

                print("\n=== SALES RECONCILIATION ===")
                print(f"CSV sales total: {expected_sales:.4f}")
                print(f"DW sales total:  {actual_sales:.4f}")
                print(f"Unique source lines: {unique_source_lines}")

                if actual_sales != expected_sales:
                    raise RuntimeError("Sales totals do not reconcile.")

                if unique_source_lines != fact_count:
                    raise RuntimeError("Duplicate source line IDs detected.")

                print("\nETL completed successfully.")

    print("Transaction committed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nETL FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

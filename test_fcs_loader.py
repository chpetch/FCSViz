import numpy as np
import pandas as pd
import fcsparser

FILE = "Example files/FlowRepository_FR-FCM-ZZZ4_files/Miltenyi Biotec - MACSQuant Analyzer.fcs"
ASINH_COFACTOR = 150
SCATTER_CHANNELS = {"FSC-A", "SSC-A", "Time"}


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main():
    # --- 1. Parse ---
    section("1. PARSE")
    try:
        meta, data = fcsparser.parse(FILE, reformat_meta=True)
        print(f"  OK — file loaded successfully")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    # --- 2. Metadata summary ---
    section("2. METADATA SUMMARY")
    keys_of_interest = [
        "$CYT", "$DATE", "$TOT", "$PAR", "$SRC",
        "TUBE NAME", "EXPERIMENT NAME", "LASER NAMES",
    ]
    for k in keys_of_interest:
        if k in meta:
            print(f"  {k:<20} {meta[k]}")
    print(f"  {'DataFrame shape':<20} {data.shape[0]:,} events × {data.shape[1]} channels")

    # --- 3. Channel inventory ---
    section("3. CHANNEL INVENTORY")
    print(f"  {'Channel':<22} {'Min':>10} {'Max':>10} {'Mean':>10} {'% Negative':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for col in data.columns:
        pct_neg = (data[col] < 0).mean() * 100
        print(f"  {col:<22} {data[col].min():>10.1f} {data[col].max():>10.1f} {data[col].mean():>10.1f} {pct_neg:>11.1f}%")

    # --- 4. DataFrame preview (first 100 rows) ---
    section("4. DATAFRAME PREVIEW — first 100 rows")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(data.head(100).to_string())

    # --- 5. Data quality checks ---
    section("5. DATA QUALITY CHECKS")
    nan_counts = data.isna().sum()
    inf_counts = np.isinf(data.values).sum(axis=0)
    zero_channels = [col for col in data.columns if (data[col] == 0).all()]

    if nan_counts.sum() == 0:
        print("  NaN values   : none")
    else:
        print("  NaN values:")
        print(nan_counts[nan_counts > 0].to_string())

    if inf_counts.sum() == 0:
        print("  Inf values   : none")
    else:
        for col, n in zip(data.columns, inf_counts):
            if n > 0:
                print(f"  Inf in {col}: {n}")

    if zero_channels:
        print(f"  All-zero     : {zero_channels}")
    else:
        print("  All-zero ch  : none")

    # --- 6. Asinh transform preview ---
    section("6. ASINH TRANSFORM PREVIEW (cofactor=150, fluorescence channels only)")
    fluor_cols = [col for col in data.columns if col not in SCATTER_CHANNELS]
    transformed = np.arcsinh(data[fluor_cols] / ASINH_COFACTOR)
    print(f"  {'Channel':<22} {'Raw min':>10} {'Raw max':>10} {'Asinh min':>10} {'Asinh max':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for col in fluor_cols:
        raw_min, raw_max = data[col].min(), data[col].max()
        t_min, t_max = transformed[col].min(), transformed[col].max()
        print(f"  {col:<22} {raw_min:>10.1f} {raw_max:>10.1f} {t_min:>10.2f} {t_max:>10.2f}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import math
# A preprocessor for generating option pricing timeseries data from historical stock prices.
#It computes rolling volatility and uses it to create a grid of option parameters (S, K, r, sigma, T, type).

def annualize_vol(std_daily: float, trading_days: int = 252) -> float:
    if np.isnan(std_daily) or std_daily <= 0:
        return np.nan
    return float(std_daily * math.sqrt(trading_days))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default="options_timeseries.csv")
    parser.add_argument("--close-col", default="Close")
    parser.add_argument("--rf", type=float, default=0.05)
    parser.add_argument("--q", type=float, default=0.0)
    parser.add_argument("--strikes", default="0.8,0.9,1.0,1.1,1.2")
    parser.add_argument("--maturities", default="30,90,180")
    parser.add_argument("--roll", type=int, default=20)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if "Date" not in df.columns or args.close_col not in df.columns:
        raise ValueError(f"CSV must have 'Date' and '{args.close_col}' columns. Found: {list(df.columns)}")

    # Sort and compute daily log returns
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["CloseUsed"] = df[args.close_col].astype(float)
    df["log_ret"] = np.log(df["CloseUsed"] / df["CloseUsed"].shift(1))

    # Rolling volatility (sample std, ddof=1), then annualize
    df["roll_std_daily"] = df["log_ret"].rolling(args.roll, min_periods=args.roll).std(ddof=1)
    df["sigma_annual"] = df["roll_std_daily"].apply(annualize_vol)

    # Prepare grids
    strike_factors = [float(x) for x in args.strikes.split(",") if x.strip() != ""]
    maturities_days = [int(x) for x in args.maturities.split(",") if x.strip() != ""]

    rows = []
    for _, row in df.iterrows():
        S0 = float(row["CloseUsed"])
        sigma = float(row["sigma_annual"]) if not pd.isna(row["sigma_annual"]) else np.nan
        date_str = row["Date"].date().isoformat()

        # Skip days without enough data for rolling vol
        if np.isnan(sigma) or sigma <= 0.0 or S0 <= 0.0:
            continue

        for days in maturities_days:
            T = days / 365.0
            for kf in strike_factors:
                K = S0 * kf
                rows.append({"date": date_str, "S": round(S0, 6), "K": round(K, 6),
                             "r": args.rf, "sigma": sigma, "T": T, "type": "C", "q": args.q})
                rows.append({"date": date_str, "S": round(S0, 6), "K": round(K, 6),
                             "r": args.rf, "sigma": sigma, "T": T, "type": "P", "q": args.q})

    out_df = pd.DataFrame(rows, columns=["date","S","K","r","sigma","T","type","q"])
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(out_df)} rows to {out_path.resolve()}")
    if len(out_df) == 0:
        print("[WARN] Output is empty. Try smaller --roll (e.g., 5) or provide more history.")

if __name__ == "__main__":
    main()

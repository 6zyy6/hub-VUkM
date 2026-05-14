from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests


TOKEN = "zgaLG8unUPr"
OUTPUT_DIR = Path(__file__).resolve().parent / "stock_plots"


def fetch_kline(code: str, period: str, start_date: str, end_date: str, adjust_type: int = 0) -> pd.DataFrame:
    url = f"https://api.autostock.cn/v1/stock/kline/{period}"
    response = requests.get(
        url,
        params={
            "token": TOKEN,
            "code": code,
            "startDate": start_date,
            "endDate": end_date,
            "type": adjust_type,
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    if not rows:
        raise ValueError(f"未获取到 {code} 的 {period} K 线数据")

    frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ["open", "close", "high", "low", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["amplitude"] = ((frame["high"] - frame["low"]) / frame["open"].replace(0, pd.NA)).fillna(0.0)
    return frame.sort_values("date").reset_index(drop=True)


def generate_suggestion(day_df: pd.DataFrame, week_df: pd.DataFrame) -> tuple[str, str]:
    latest_close = day_df.iloc[-1]["close"]
    day_amplitude = day_df.iloc[-1]["amplitude"]
    week_amplitude = week_df.iloc[-1]["amplitude"]

    min_close = float(day_df["close"].min())
    max_close = float(day_df["close"].max())
    close_range = max(max_close - min_close, 1e-6)
    position_ratio = float((latest_close - min_close) / close_range)

    day_q75 = float(day_df["amplitude"].quantile(0.75))
    week_q75 = float(week_df["amplitude"].quantile(0.75))
    day_q25 = float(day_df["amplitude"].quantile(0.25))
    week_q25 = float(week_df["amplitude"].quantile(0.25))

    if position_ratio >= 0.75 and (day_amplitude >= day_q75 or week_amplitude >= week_q75):
        action = "适合分批卖出"
        reason = "当前价格接近区间高位，且近期日波动或周波动偏大，短期回撤风险上升。"
    elif position_ratio <= 0.30 and day_amplitude <= day_q75 and week_amplitude <= week_q25:
        action = "适合买入"
        reason = "当前价格靠近区间低位，周波动回落，短期情绪没有继续恶化，更适合分批低吸。"
    elif position_ratio <= 0.35 and day_amplitude <= day_q25:
        action = "适合买入"
        reason = "当前价格处于相对低位，且短期日波动收敛，说明抛压可能在减弱。"
    else:
        action = "适合继续观察"
        reason = "价格位置和波动水平都不处于明显极值，暂时缺少高胜率买卖信号。"

    return action, reason


def plot_volatility(code: str, day_df: pd.DataFrame, week_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(day_df["date"], day_df["amplitude"], label="Daily amplitude", linewidth=1.8, color="#1f77b4")
    ax.plot(week_df["date"], week_df["amplitude"], label="Weekly amplitude", linewidth=2.2, color="#d62728")
    ax.set_title(f"{code} daily and weekly volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()

    output_path = OUTPUT_DIR / f"{code}_volatility.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="股票周波动与日波动可视化")
    parser.add_argument("--code", required=True, help="股票代码，例如 sh600519")
    parser.add_argument("--start-date", default=(datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d"))
    parser.add_argument("--end-date", default=datetime.today().strftime("%Y-%m-%d"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    day_df = fetch_kline(args.code, "day", args.start_date, args.end_date)
    week_df = fetch_kline(args.code, "week", args.start_date, args.end_date)
    output_path = plot_volatility(args.code, day_df, week_df)
    action, reason = generate_suggestion(day_df, week_df)

    print(f"图片已生成: {output_path}")
    print(f"建议: {action}")
    print(f"原因: {reason}")


if __name__ == "__main__":
    main()
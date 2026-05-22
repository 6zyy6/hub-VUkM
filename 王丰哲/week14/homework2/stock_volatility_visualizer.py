#!/usr/bin/env python3
"""Fetch daily/weekly K-lines, draw volatility in one SVG, and print timing advice."""

from __future__ import annotations

import argparse
import html
import json
import os
import statistics
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://api.autostock.cn/v1/stock/kline/{period}"
DEFAULT_TOKEN = "zgaLG8unUPr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw daily and weekly stock volatility in one SVG chart."
    )
    parser.add_argument("--code", required=True, help="Stock code, for example 000001")
    parser.add_argument("--start-date", help="Start date, YYYY-MM-DD")
    parser.add_argument("--end-date", help="End date, YYYY-MM-DD")
    parser.add_argument(
        "--adjust-type",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0=unadjusted, 1=forward adjusted, 2=back adjusted",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=120,
        help="Max recent daily K-line records to draw",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=52,
        help="Max recent weekly K-line records to draw",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG path. Defaults to <code>_volatility.svg",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("AUTOSTOCK_TOKEN", DEFAULT_TOKEN),
        help="AutoStock token. Defaults to AUTOSTOCK_TOKEN or the course token.",
    )
    return parser.parse_args()


def fetch_kline(
    code: str,
    period: str,
    start_date: str | None,
    end_date: str | None,
    adjust_type: int,
    token: str,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"token": token, "code": code, "type": adjust_type}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    response = requests.get(BASE_URL.format(period=period), params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()

    if payload.get("code") != 200:
        raise RuntimeError(f"AutoStock API error for {period}: {payload}")

    rows = payload.get("data") or []
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            date = datetime.strptime(str(row[0]), "%Y-%m-%d").date()
            open_price = float(row[1])
            close = float(row[2])
            high = float(row[3])
            low = float(row[4])
            volume = float(row[5])
        except (TypeError, ValueError):
            continue

        volatility = ((high - low) / close * 100) if close else 0.0
        change = ((close - open_price) / open_price * 100) if open_price else 0.0
        records.append(
            {
                "date": date,
                "date_text": date.isoformat(),
                "open": open_price,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "volatility_pct": volatility,
                "change_pct": change,
            }
        )

    if not records:
        raise RuntimeError(f"No valid {period} K-line records returned for {code}")
    return records


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def select_advice(
    code: str, daily: list[dict[str, Any]], weekly: list[dict[str, Any]]
) -> dict[str, Any]:
    daily_vols = [item["volatility_pct"] for item in daily]
    weekly_vols = [item["volatility_pct"] for item in weekly]
    daily_q25 = percentile(daily_vols, 0.25)
    daily_q75 = percentile(daily_vols, 0.75)
    weekly_q75 = percentile(weekly_vols, 0.75)
    weekly_avg = statistics.mean(weekly_vols)

    buy_pool = [
        item
        for item in daily
        if item["volatility_pct"] <= daily_q25 and item["change_pct"] >= -1.5
    ] or daily
    sell_pool = [
        item
        for item in daily
        if item["volatility_pct"] >= daily_q75 and item["change_pct"] <= 1.5
    ] or daily

    buy = min(buy_pool, key=lambda item: (item["close"], item["volatility_pct"]))
    sell = max(sell_pool, key=lambda item: (item["close"], item["volatility_pct"]))

    latest_daily = daily[-1]
    latest_weekly = weekly[-1]
    if (
        latest_daily["volatility_pct"] <= daily_q25
        and latest_weekly["volatility_pct"] <= weekly_avg
    ):
        current_view = "日线和周线波动同步偏低，可作为低吸观察区，建议分批而不是一次性买入。"
    elif (
        latest_daily["volatility_pct"] >= daily_q75
        or latest_weekly["volatility_pct"] >= weekly_q75
    ):
        current_view = "最新波动已经放大，追高风险较高，更适合减仓、止盈或等待回落确认。"
    else:
        current_view = "当前波动处于中性区间，建议等待日线波动收敛或周线方向进一步确认。"

    return {
        "code": code,
        "buy_time": buy["date_text"],
        "buy_reason": (
            f"日波动 {buy['volatility_pct']:.2f}%，处于近期低波动区；"
            f"收盘价 {buy['close']:.2f}，适合观察或分批买入。"
        ),
        "sell_time": sell["date_text"],
        "sell_reason": (
            f"日波动 {sell['volatility_pct']:.2f}%，处于近期高波动区；"
            f"收盘价 {sell['close']:.2f}，适合止盈、减仓或等待回落。"
        ),
        "latest_daily_volatility_pct": round(latest_daily["volatility_pct"], 4),
        "latest_weekly_volatility_pct": round(latest_weekly["volatility_pct"], 4),
        "daily_low_threshold_pct": round(daily_q25, 4),
        "daily_high_threshold_pct": round(daily_q75, 4),
        "weekly_high_threshold_pct": round(weekly_q75, 4),
        "current_view": current_view,
        "risk_note": "该结果只基于历史K线波动，不构成投资建议。",
    }


def points_for(
    records: list[dict[str, Any]],
    min_date: date,
    max_date: date,
    max_vol: float,
    box: dict[str, int],
) -> str:
    span_days = max((max_date - min_date).days, 1)
    max_vol = max(max_vol, 1.0)
    points = []
    for item in records:
        x = box["left"] + ((item["date"] - min_date).days / span_days) * box["width"]
        y = box["top"] + box["height"] - (
            item["volatility_pct"] / max_vol
        ) * box["height"]
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def text(x: float, y: float, content: str, size: int = 14, fill: str = "#111827") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}">{html.escape(content)}</text>'
    )


def render_svg(
    code: str,
    daily: list[dict[str, Any]],
    weekly: list[dict[str, Any]],
    advice: dict[str, Any],
) -> str:
    width, height = 1100, 650
    box = {"left": 76, "top": 82, "width": 940, "height": 390}
    all_records = daily + weekly
    min_date = min(item["date"] for item in all_records)
    max_date = max(item["date"] for item in all_records)
    max_vol = max(item["volatility_pct"] for item in all_records) * 1.12
    max_vol = max(max_vol, 1.0)

    daily_points = points_for(daily, min_date, max_date, max_vol, box)
    weekly_points = points_for(weekly, min_date, max_date, max_vol, box)

    grid = []
    for index in range(6):
        ratio = index / 5
        y = box["top"] + box["height"] * ratio
        value = max_vol * (1 - ratio)
        grid.append(
            f'<line x1="{box["left"]}" y1="{y:.1f}" '
            f'x2="{box["left"] + box["width"]}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>'
        )
        grid.append(text(18, y + 5, f"{value:.1f}%", 12, "#6b7280"))

    date_labels = [
        (box["left"], min_date.isoformat()),
        (box["left"] + box["width"] / 2 - 44, daily[len(daily) // 2]["date_text"]),
        (box["left"] + box["width"] - 86, max_date.isoformat()),
    ]
    for x, label in date_labels:
        grid.append(text(x, box["top"] + box["height"] + 34, label, 12, "#6b7280"))

    buy_line = f"买入观察：{advice['buy_time']}，{advice['buy_reason']}"
    sell_line = f"卖出/减仓：{advice['sell_time']}，{advice['sell_reason']}"
    view_line = f"当前判断：{advice['current_view']}"

    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            text(76, 40, f"{code} 日波动与周波动对比", 26, "#111827"),
            text(76, 66, "波动率 = (最高价 - 最低价) / 收盘价 * 100%", 13, "#6b7280"),
            *grid,
            f'<rect x="{box["left"]}" y="{box["top"]}" width="{box["width"]}" height="{box["height"]}" fill="none" stroke="#9ca3af" stroke-width="1"/>',
            f'<polyline fill="none" stroke="#2563eb" stroke-width="2.1" points="{daily_points}"/>',
            f'<polyline fill="none" stroke="#dc2626" stroke-width="3" stroke-dasharray="8 5" points="{weekly_points}"/>',
            '<circle cx="812" cy="48" r="5" fill="#2563eb"/>',
            text(824, 53, "日波动", 13, "#374151"),
            '<circle cx="900" cy="48" r="5" fill="#dc2626"/>',
            text(912, 53, "周波动", 13, "#374151"),
            '<rect x="76" y="510" width="940" height="104" rx="8" fill="#f9fafb" stroke="#d1d5db"/>',
            text(96, 540, buy_line, 14, "#14532d"),
            text(96, 570, sell_line, 14, "#7f1d1d"),
            text(96, 600, view_line, 14, "#374151"),
            "</svg>",
        ]
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output or f"{args.code}_volatility.svg")

    daily = fetch_kline(
        args.code, "day", args.start_date, args.end_date, args.adjust_type, args.token
    )[-args.days :]
    weekly = fetch_kline(
        args.code, "week", args.start_date, args.end_date, args.adjust_type, args.token
    )[-args.weeks :]

    advice = select_advice(args.code, daily, weekly)
    svg = render_svg(args.code, daily, weekly, advice)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")

    result = {"chart": str(output), "advice": advice}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

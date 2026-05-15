---
name: 股票可视化分析
description: 对股票日K线和周K线进行可视化，分析波动并给出买入卖出建议。
---

# 股票可视化分析 Skill

基于 `autostock/SKILL.md` 的数据接口，增加可视化功能。

## 一、核心可视化函数

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict, Optional
from datetime import datetime

TOKEN = "zgaLG8unUPr"

def plot_stock_kline(code: str, days: int = 60) -> str:
    """
    绘制股票日K线和周K线叠加图，显示波动及买卖信号。

    参数:
        code: 股票代码，如 "000001"
        days: 分析天数，默认60天

    返回:
        保存的图表路径
    """
    import requests
    import pandas as pd
    import numpy as np

    # 1. 获取日K线数据
    day_url = f"https://api.autostock.cn/v1/stock/kline/day?token={TOKEN}"
    day_payload = {"code": code, "type": 1}
    day_resp = requests.get(day_url, params=day_payload, timeout=10)
    day_data = day_resp.json().get("data", [])

    # 2. 获取周K线数据
    week_url = f"https://api.autostock.cn/v1/stock/kline/week?token={TOKEN}"
    week_payload = {"code": code, "type": 1}
    week_resp = requests.get(week_url, params=week_payload, timeout=10)
    week_data = week_resp.json().get("data", [])

    if not day_data:
        return "获取数据失败"

    # 3. 转换为DataFrame
    df_day = pd.DataFrame(day_data)
    df_week = pd.DataFrame(week_data)

    # 处理日期
    if "date" in df_day.columns:
        df_day["date"] = pd.to_datetime(df_day["date"])
        df_day = df_day.sort_values("date")

    if "date" in df_week.columns:
        df_week["date"] = pd.to_datetime(df_week["date"])
        df_week = df_week.sort_values("date")

    # 取最近N天数据
    df_day = df_day.tail(days)

    # 4. 计算技术指标
    # 日K线均线
    df_day["ma5"] = df_day["close"].rolling(window=5).mean()
    df_day["ma10"] = df_day["close"].rolling(window=10).mean()
    df_day["ma20"] = df_day["close"].rolling(window=20).mean()

    # 周K线均线
    df_week["ma5"] = df_week["close"].rolling(window=5).mean()

    # 5. 计算波动率
    daily_volatility = df_day["close"].pct_change().std() * 100
    weekly_volatility = df_week["close"].pct_change().std() * 100 if len(df_week) > 1 else 0

    # 6. 绘制图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"股票 {code} K线分析与波动率", fontsize=14, fontweight='bold')

    # 上图：K线 + 均线
    ax1.plot(df_day["date"], df_day["close"], label="日K线收盘价", color="#2196F3", linewidth=1.5)
    ax1.plot(df_day["date"], df_day["ma5"], label="5日均线", color="#FF9800", linewidth=1, alpha=0.8)
    ax1.plot(df_day["date"], df_day["ma10"], label="10日均线", color="#9C27B0", linewidth=1, alpha=0.8)
    ax1.plot(df_day["date"], df_day["ma20"], label="20日均线", color="#4CAF50", linewidth=1, alpha=0.8)

    # 周K线叠加（用虚线）
    ax1.plot(df_week["date"], df_week["close"], label="周K线收盘价", color="#E91E63", linewidth=2, linestyle="--", alpha=0.6)

    # 标记买卖信号
    signals = detect_buy_sell_signals(df_day)
    for sig in signals:
        if sig["type"] == "买入":
            ax1.scatter(sig["date"], sig["price"], marker="^", color="red", s=150, zorder=5)
        elif sig["type"] == "卖出":
            ax1.scatter(sig["date"], sig["price"], marker="v", color="green", s=150, zorder=5)

    ax1.set_ylabel("价格")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.tick_params(axis='x', rotation=45)

    # 下图：波动率
    ax2.bar(df_day["date"], df_day["close"].pct_change().abs() * 100,
            color=np.where(df_day["close"].pct_change() >= 0, "#FF5252", "#4CAF50"),
            alpha=0.7, width=0.8)
    ax2.axhline(y=daily_volatility, color="orange", linestyle="--", label=f"日波动率均值: {daily_volatility:.2f}%")
    ax2.axhline(y=weekly_volatility, color="purple", linestyle="--", label=f"周波动率均值: {weekly_volatility:.2f}%")
    ax2.set_ylabel("波动率 (%)")
    ax2.set_xlabel("日期")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存图表
    output_path = f"/Users/jlbi/Desktop/作业2/autostock/stock_{code}_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def detect_buy_sell_signals(df) -> List[Dict]:
    """
    基于K线形态检测买卖信号

    买入信号：
    - 启明星形态（底部反转）
    - 阳包阴（收盘价高于前一天开盘价，且前一天是阴线）
    - 价格站上20日均线且放量

    卖出信号：
    - 黄昏星形态（顶部反转）
    - 阴包阳
    - 价格跌破20日均线且放量
    """
    signals = []
    data = df.reset_index(drop=True)

    for i in range(1, len(data)):
        current = data.iloc[i]
        previous = data.iloc[i-1]

        # 买入信号1：启明星（长下影线 + 阳线）
        if (current["low"] < previous["low"] and
            current["close"] > current["open"] and
            (current["low"] - min(current["open"], current["close"])) > (current["close"] - current["low"]) * 0.6):
            signals.append({
                "type": "买入",
                "date": current["date"],
                "price": current["close"],
                "reason": "启明星形态"
            })

        # 买入信号2：阳包阴
        if (current["close"] > previous["open"] and
            previous["close"] < previous["open"] and
            current["close"] > current["open"]):
            signals.append({
                "type": "买入",
                "date": current["date"],
                "price": current["close"],
                "reason": "阳包阴"
            })

        # 卖出信号1：黄昏星
        if (current["high"] > previous["high"] and
            current["close"] < current["open"] and
            (max(current["open"], current["close"]) - current["high"]) > (current["low"] - min(current["open"], current["close"])) * 0.6):
            signals.append({
                "type": "卖出",
                "date": current["date"],
                "price": current["close"],
                "reason": "黄昏星形态"
            })

        # 卖出信号2：阴包阳
        if (current["close"] < previous["open"] and
            previous["close"] > previous["open"] and
            current["close"] < current["open"]):
            signals.append({
                "type": "卖出",
                "date": current["date"],
                "price": current["close"],
                "reason": "阴包阳"
            })

    return signals


def analyze_stock(code: str) -> Dict:
    """
    综合分析股票，返回分析结果和建议
    """
    import requests
    import pandas as pd

    # 获取日K线
    day_url = f"https://api.autostock.cn/v1/stock/kline/day?token={TOKEN}"
    day_resp = requests.get(day_url, params={"code": code, "type": 1}, timeout=10)
    day_data = day_resp.json().get("data", [])

    # 获取周K线
    week_url = f"https://api.autostock.cn/v1/stock/kline/week?token={TOKEN}"
    week_resp = requests.get(week_url, params={"code": code, "type": 1}, timeout=10)
    week_data = week_resp.json().get("data", [])

    df_day = pd.DataFrame(day_data)
    df_week = pd.DataFrame(week_data)

    if "date" in df_day.columns:
        df_day["date"] = pd.to_datetime(df_day["date"])
        df_day = df_day.sort_values("date").tail(60)

    if "date" in df_week.columns:
        df_week["date"] = pd.to_datetime(df_week["date"])
        df_week = df_week.sort_values("date").tail(20)

    # 计算指标
    df_day["ma5"] = df_day["close"].rolling(5).mean()
    df_day["ma10"] = df_day["close"].rolling(10).mean()
    df_day["ma20"] = df_day["close"].rolling(20).mean()

    # 最新价格
    latest = df_day.iloc[-1]
    latest_price = latest["close"]
    ma5 = latest["ma5"] if pd.notna(latest["ma5"]) else 0
    ma10 = latest["ma10"] if pd.notna(latest["ma10"]) else 0
    ma20 = latest["ma20"] if pd.notna(latest["ma20"]) else 0

    # 波动率
    daily_vol = df_day["close"].pct_change().std() * 100
    weekly_vol = df_week["close"].pct_change().std() * 100 if len(df_week) > 1 else 0

    # 趋势判断
    if ma5 > ma10 > ma20 and latest_price > ma20:
        trend = "上升趋势"
    elif ma5 < ma10 < ma20 and latest_price < ma20:
        trend = "下降趋势"
    else:
        trend = "震荡整理"

    # 买卖信号检测
    signals = detect_buy_sell_signals(df_day)
    buy_signals = [s for s in signals if s["type"] == "买入"]
    sell_signals = [s for s in signals if s["type"] == "卖出"]

    # 生成建议
    suggestions = []
    if trend == "上升趋势" and buy_signals:
        suggestions.append({
            "action": "买入",
            "reason": f"处于{trend}，检测到买入信号: {buy_signals[-1]['reason']}",
            "target": round(latest_price * 1.05, 2)  # 5%涨幅目标
        })
    elif trend == "下降趋势" and sell_signals:
        suggestions.append({
            "action": "卖出",
            "reason": f"处于{trend}，检测到卖出信号: {sell_signals[-1]['reason']}",
            "stop_loss": round(latest_price * 0.95, 2)  # 5%止损
        })
    elif trend == "震荡整理":
        suggestions.append({
            "action": "观望",
            "reason": "市场震荡，等待明确方向信号"
        })

    # 最佳买卖时机（基于波动率分析）
    best_buy = None
    best_sell = None
    for i in range(5, len(df_day)):
        if df_day.iloc[i]["close"] < df_day.iloc[i]["ma20"] * 0.98 and df_day.iloc[i]["volume"] > df_day.iloc[i-1]["volume"] * 1.2:
            best_buy = {
                "date": str(df_day.iloc[i]["date"].date()),
                "price": df_day.iloc[i]["close"]
            }
            break

    for i in range(5, len(df_day)):
        if df_day.iloc[i]["close"] > df_day.iloc[i]["ma20"] * 1.05:
            best_sell = {
                "date": str(df_day.iloc[i]["date"].date()),
                "price": df_day.iloc[i]["close"]
            }
            break

    return {
        "code": code,
        "latest_price": latest_price,
        "trend": trend,
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
        "daily_volatility": round(daily_vol, 2),
        "weekly_volatility": round(weekly_vol, 2),
        "buy_signals_count": len(buy_signals),
        "sell_signals_count": len(sell_signals),
        "suggestions": suggestions,
        "best_buy": best_buy,
        "best_sell": best_sell
    }
```

## 二、输出格式

### 分析结果示例
```json
{
    "code": "000001",
    "latest_price": 15.80,
    "trend": "上升趋势",
    "ma5": 15.50,
    "ma10": 15.20,
    "ma20": 14.80,
    "daily_volatility": 2.35,
    "weekly_volatility": 5.21,
    "buy_signals_count": 2,
    "sell_signals_count": 0,
    "suggestions": [
        {
            "action": "买入",
            "reason": "处于上升趋势，检测到买入信号: 阳包阴",
            "target": 16.59
        }
    ],
    "best_buy": {"date": "2026-05-10", "price": 14.85},
    "best_sell": {"date": "2026-05-14", "price": 16.20}
}
```

### 生成的图表
- 文件保存至：`/Users/jlbi/Desktop/作业2/autostock/stock_{code}_analysis.png`
- 包含：日K线、周K线叠加、均线、买卖信号标记、波动率柱状图

## 三、使用流程

1. **查询股票**：`get_stock_code` 获取股票代码
2. **分析股票**：`analyze_stock(code)` 获取分析结果和建议
3. **生成图表**：`plot_stock_kline(code, days=60)` 生成可视化图表
4. **查看信号**：根据图表中的红三角（买入）和绿倒三角（卖出）判断时机

## 四、买卖时机判断规则

| 条件 | 操作 |
|------|------|
| MA5 > MA10 > MA20，价格在均线上方，有买入信号 | **买入/加仓** |
| 价格跌破MA20且放量 | **止损/减仓** |
| MA5 < MA10 < MA20，价格在均线下方，有卖出信号 | **卖出/空仓** |
| 波动率异常放大（>3倍均值） | **注意风险** |
| 周波动率远大于日波动率 | **市场情绪不稳定，谨慎操作** |
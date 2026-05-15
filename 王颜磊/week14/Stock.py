"""
股票波动率可视化分析脚本
功能：
1. 绘制日波动率和周波动率对比图
2. 基于波动率给出买卖时机建议
3. 输出为 HTML 文件

依赖：pandas, plotly
"""

import json
import os
from datetime import datetime, timedelta


def generate_chart_html(
    stock_name: str,
    stock_code: str,
    daily_data: list,
    weekly_data: list,
) -> str:
    """
    根据日K和周K数据生成 HTML 可视化图表

    Parameters
    ----------
    stock_name : str - 股票名称
    stock_code : str - 股票代码
    daily_data : list - 日K数据，每个元素为 dict:
        {"date": "2026-01-01", "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1000000}
    weekly_data : list - 周K数据，格式同上

    Returns
    -------
    str - HTML 内容
    """
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # ============ 日K数据处理 ============
    df_daily = pd.DataFrame(daily_data)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    df_daily["daily_vol"] = df_daily["high"] - df_daily["low"]
    df_daily["daily_vol_pct"] = (df_daily["high"] - df_daily["low"]) / df_daily["close"] * 100
    df_daily["daily_return"] = df_daily["close"].pct_change() * 100
    df_daily["vol_ma20"] = df_daily["daily_vol_pct"].rolling(20).mean()

    # 买入/卖出信号判断
    signals = []
    for i in range(len(df_daily)):
        row = df_daily.iloc[i]
        vol_ma = row["vol_ma20"]
        close = row["close"]

        if pd.isna(vol_ma):
            signals.append("hold")
            continue

        # 近3天波动率都低于均值 → 买入信号
        if i >= 2:
            vol_3d = df_daily.iloc[i-2:i+1]["daily_vol_pct"]
            if all(v < vol_ma * 0.8 for v in vol_3d):
                signals.append("buy")
                continue

        # 波动率突破均值1.5倍 → 卖出信号
        if row["daily_vol_pct"] > vol_ma * 1.5:
            signals.append("sell")
            continue

        signals.append("hold")

    df_daily["signal"] = signals

    # ============ 周K数据处理 ============
    df_weekly = pd.DataFrame(weekly_data)
    df_weekly["date"] = pd.to_datetime(df_weekly["date"])
    df_weekly = df_weekly.sort_values("date").reset_index(drop=True)

    df_weekly["weekly_vol"] = df_weekly["high"] - df_weekly["low"]
    df_weekly["weekly_vol_pct"] = (df_weekly["high"] - df_weekly["low"]) / df_weekly["close"] * 100
    weekly_median = df_weekly["weekly_vol_pct"].median()
    weekly_q75 = df_weekly["weekly_vol_pct"].quantile(0.75)
    weekly_q25 = df_weekly["weekly_vol_pct"].quantile(0.25)

    # ============ 计算建议文本 ============
    current_daily_vol = df_daily["daily_vol_pct"].iloc[-1]
    current_daily_ma = df_daily["vol_ma20"].iloc[-1]
    current_weekly_vol = df_weekly["weekly_vol_pct"].iloc[-1]
    latest_signal = df_daily["signal"].iloc[-1]

    # 统计买卖信号数量
    buy_count = signals.count("buy")
    sell_count = signals.count("sell")
    hold_count = signals.count("hold")

    # 波动率收缩期（建议买入）
    low_vol_days = df_daily["daily_vol_pct"].iloc[-3:].tolist()
    vol_shrink = all(v < current_daily_ma * 0.8 for v in low_vol_days if pd.notna(v))

    # 波动率扩张期（建议卖出）
    vol_expand = current_daily_vol > current_daily_ma * 1.5
    weekly_high_vol = current_weekly_vol > weekly_q75

    # 生成建议
    if vol_shrink and not vol_expand:
        advice_status = "波动收缩期 - 建议关注买入机会"
        advice_color = "#E24B4A"  # 红色
    elif vol_expand or weekly_high_vol:
        advice_status = "波动扩张期 - 建议谨慎考虑卖出"
        advice_color = "#3B6D11"  # 绿色
    else:
        advice_status = "波动正常期 - 建议持有观察"
        advice_color = "#185FA5"  # 蓝色

    # ============ 绘制图表 ============
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{stock_name}({stock_code}) - 股价走势与交易信号",
            f"{stock_name}({stock_code}) - 日波动率 vs 周波动率"
        ),
    )

    # ---- 上方子图: 股价走势 + 信号 ----
    # 日收盘价
    fig.add_trace(
        go.Scatter(
            x=df_daily["date"], y=df_daily["close"],
            mode="lines+markers",
            name="日收盘价",
            line=dict(color="#185FA5", width=2),
            marker=dict(size=3),
        ),
        row=1, col=1,
    )

    # 周收盘价
    fig.add_trace(
        go.Scatter(
            x=df_weekly["date"], y=df_weekly["close"],
            mode="lines+markers",
            name="周收盘价",
            line=dict(color="#534AB7", width=2.5, dash="dot"),
            marker=dict(size=5),
        ),
        row=1, col=1,
    )

    # 买入信号
    buy_df = df_daily[df_daily["signal"] == "buy"]
    if len(buy_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_df["date"], y=buy_df["close"],
                mode="markers",
                name="买入信号",
                marker=dict(symbol="triangle-up", size=14, color="#E24B4A", line=dict(width=1, color="white")),
            ),
            row=1, col=1,
        )

    # 卖出信号
    sell_df = df_daily[df_daily["signal"] == "sell"]
    if len(sell_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_df["date"], y=sell_df["close"],
                mode="markers",
                name="卖出信号",
                marker=dict(symbol="triangle-down", size=14, color="#3B6D11", line=dict(width=1, color="white")),
            ),
            row=1, col=1,
        )

    # ---- 下方子图: 波动率柱状图 ----
    # 日波动率
    colors_daily = []
    for _, row in df_daily.iterrows():
        if row["signal"] == "buy":
            colors_daily.append("rgba(226, 75, 74, 0.7)")
        elif row["signal"] == "sell":
            colors_daily.append("rgba(59, 109, 17, 0.7)")
        else:
            colors_daily.append("rgba(24, 95, 165, 0.5)")

    fig.add_trace(
        go.Bar(
            x=df_daily["date"], y=df_daily["daily_vol_pct"],
            name="日波动率(%)",
            marker_color=colors_daily,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # 日波动率20日均线
    fig.add_trace(
        go.Scatter(
            x=df_daily["date"], y=df_daily["vol_ma20"],
            mode="lines",
            name="日波动率20日均线",
            line=dict(color="#D85A30", width=1.5, dash="dash"),
        ),
        row=2, col=1,
    )

    # 周波动率
    fig.add_trace(
        go.Bar(
            x=df_weekly["date"], y=df_weekly["weekly_vol_pct"],
            name="周波动率(%)",
            marker_color="rgba(83, 74, 183, 0.6)",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # 更新布局
    fig.update_layout(
        height=700,
        template="plotly_white",
        font=dict(family="Microsoft YaHei, SimHei, Arial", size=13),
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(
            orientation="h", y=1.12,
            xanchor="center", x=0.5,
            font=dict(size=12),
        ),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="价格 (元)", row=1, col=1)
    fig.update_yaxes(title_text="波动率 (%)", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)

    chart_json = fig.to_json()

    # ============ 生成 HTML ============
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{stock_name}({stock_code}) - 股票波动率分析</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
    background: #f5f5f5;
    color: #333;
    padding: 20px;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .header {{
    text-align: center;
    padding: 20px 0 10px;
  }}
  .header h1 {{
    font-size: 24px;
    font-weight: 500;
    color: #222;
  }}
  .header .subtitle {{
    font-size: 14px;
    color: #888;
    margin-top: 4px;
  }}
  .chart-box {{
    background: white;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  #chart {{ width: 100%; }}
  .advice-box {{
    background: white;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-left: 4px solid {advice_color};
  }}
  .advice-box h2 {{
    font-size: 16px;
    font-weight: 500;
    color: {advice_color};
    margin-bottom: 12px;
  }}
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }}
  .metric {{
    background: #f8f9fa;
    border-radius: 6px;
    padding: 12px 16px;
  }}
  .metric .label {{
    font-size: 12px;
    color: #888;
    margin-bottom: 4px;
  }}
  .metric .value {{
    font-size: 20px;
    font-weight: 500;
  }}
  .metric .value.red {{ color: #E24B4A; }}
  .metric .value.green {{ color: #3B6D11; }}
  .metric .value.blue {{ color: #185FA5; }}
  .signal-legend {{
    display: flex;
    gap: 20px;
    margin: 12px 0;
    flex-wrap: wrap;
  }}
  .signal-legend span {{
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .signal-legend .dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
  }}
  .suggestions {{
    line-height: 1.8;
    font-size: 14px;
  }}
  .suggestions .buy {{ color: #E24B4A; font-weight: 500; }}
  .suggestions .sell {{ color: #3B6D11; font-weight: 500; }}
  .suggestions .hold {{ color: #185FA5; font-weight: 500; }}
  .disclaimer {{
    text-align: center;
    font-size: 12px;
    color: #aaa;
    margin-top: 16px;
    padding: 10px;
  }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{stock_name}({stock_code}) - 波动率分析报告</h1>
    <div class="subtitle">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 基于近{len(df_daily)}个交易日 + {len(df_weekly)}周数据</div>
  </div>

  <div class="chart-box">
    <div id="chart"></div>
    <script>
      var graphData = {chart_json};
      Plotly.newPlot('chart', graphData.data, graphData.layout, {{responsive: true}});
    </script>
  </div>

  <div class="advice-box">
    <h2>{advice_status}</h2>

    <div class="metrics">
      <div class="metric">
        <div class="label">当前日波动率</div>
        <div class="value blue">{current_daily_vol:.2f}%</div>
      </div>
      <div class="metric">
        <div class="label">日波动率20日均值</div>
        <div class="value">{current_daily_ma:.2f}%</div>
      </div>
      <div class="metric">
        <div class="label">当前周波动率</div>
        <div class="value blue">{current_weekly_vol:.2f}%</div>
      </div>
      <div class="metric">
        <div class="label">周波动率12周中位</div>
        <div class="value">{weekly_median:.2f}%</div>
      </div>
    </div>

    <div class="signal-legend">
      <span><span class="dot" style="background:#E24B4A"></span> 买入信号 ({buy_count}个): 波动率连续收缩</span>
      <span><span class="dot" style="background:#3B6D11"></span> 卖出信号 ({sell_count}个): 波动率急剧扩张</span>
      <span><span class="dot" style="background:#185FA5"></span> 持有 ({hold_count}个): 波动率正常</span>
    </div>

    <div class="suggestions">
      <p><span class="buy">买入时机参考：</span>{"波动率连续3天低于20日均值的80%，市场进入盘整收敛阶段，后续可能产生方向性突破。结合周波动率是否处于低位（< {:.2f}%）综合判断。".format(weekly_q25) if vol_shrink else "当前波动率未明显收缩，暂无明确买入信号。"}</p>
      <p><span class="sell">卖出时机参考：</span>{"当前日波动率（{:.2f}%）突破20日均线（{:.2f}%）的1.5倍，市场波动加剧，建议关注止盈。".format(current_daily_vol, current_daily_ma) if vol_expand else "当前周波动率（{:.2f}%）处于12周高位（> {:.2f}%），波动较大，注意风险。".format(current_weekly_vol, weekly_q75) if weekly_high_vol else "当前波动率未明显扩张，暂无明确卖出信号。"}</p>
      <p><span class="hold">持有建议：</span>波动率在正常区间运行，趋势稳定，建议继续持有并观察波动率变化方向。</p>
    </div>
  </div>

  <div class="disclaimer">
    风险提示：以上分析仅基于历史波动率数据，不构成任何投资建议。股市有风险，投资需谨慎。
  </div>
</div>
</body>
</html>"""

    return html, {
        "stock_name": stock_name,
        "stock_code": stock_code,
        "current_daily_vol": round(current_daily_vol, 2),
        "daily_vol_ma20": round(current_daily_ma, 2),
        "current_weekly_vol": round(current_weekly_vol, 2),
        "weekly_median": round(weekly_median, 2),
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "hold_signals": hold_count,
        "advice": advice_status,
    }


# ============ 真实数据获取 ============
def fetch_real_data(stock_code: str, days: int = 60):
    """
    通过 akshare 获取真实的股票行情数据

    Parameters
    ----------
    stock_code : str - 股票代码，支持格式：
        - A股: "600519" (贵州茅台)
        - 港股: "01810" (小米集团)
        - 美股: "AAPL" (苹果)
    days : int - 获取最近N个交易日，默认60

    Returns
    -------
    tuple - (daily_data, weekly_data, stock_name, display_code)
    """
    import akshare as ak
    import pandas as pd

    code = stock_code.strip()

    # 判断市场
    if code.isdigit() and len(code) == 5:
        # 港股
        market = "hk"
        df = ak.stock_hk_daily(symbol=code, adjust='qfq')
        stock_name_code = f"{code}.HK"
    elif code.isdigit() and len(code) == 6:
        # A股 - 上海 (60xxxx / 68xxxx) 或 深圳 (00xxxx / 30xxxx)
        if code.startswith(('6', '68')):
            market = "sh"
            df = ak.stock_zh_a_daily(symbol=f"sh{code}", adjust='qfq')
        else:
            market = "sz"
            df = ak.stock_zh_a_daily(symbol=f"sz{code}", adjust='qfq')
        stock_name_code = f"{code}.{market.upper()}"
    else:
        # 美股或其他
        market = "us"
        df = ak.stock_us_daily(symbol=code, adjust='qfq')
        stock_name_code = code.upper()

    if df is None or len(df) == 0:
        raise ValueError(f"无法获取股票 {stock_code} 的数据，请检查代码是否正确")

    df = df.tail(days).reset_index(drop=True)

    # 获取股票名称（从 neodata 或默认用代码）
    stock_name = stock_code

    # 日K数据标准化
    daily_data = []
    for _, row in df.iterrows():
        daily_data.append({
            "date": str(row["date"])[:10],
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row["volume"]),
        })

    # 按自然周聚合周K
    df_parsed = pd.DataFrame(daily_data)
    df_parsed["date"] = pd.to_datetime(df_parsed["date"])
    df_parsed["week"] = df_parsed["date"].dt.isocalendar().week.astype(int)
    df_parsed["year"] = df_parsed["date"].dt.year

    weekly_data = []
    for (year, week), group in df_parsed.groupby(["year", "week"]):
        group = group.sort_values("date")
        weekly_data.append({
            "date": group["date"].iloc[-1].strftime("%Y-%m-%d"),
            "open": group["open"].iloc[0],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group["close"].iloc[-1],
            "volume": int(group["volume"].sum()),
        })

    return daily_data, weekly_data, stock_name, stock_name_code


# ============ 演示用：生成示例数据 ============
def generate_demo_data():
    """
    生成模拟的股票数据用于演示
    """
    import random
    import pandas as pd

    random.seed(42)

    # 生成60个交易日的日K数据
    dates_daily = []
    base_date = datetime(2026, 3, 1)
    for i in range(60):
        d = base_date + timedelta(days=i)
        if d.weekday() < 5:
            dates_daily.append(d)

    closes = [50.0]
    daily_data = []
    for i, date in enumerate(dates_daily):
        change = random.uniform(-3, 3)
        close = closes[-1] * (1 + change / 100)
        close = round(close, 2)
        high = round(close * (1 + random.uniform(0.5, 3) / 100), 2)
        low = round(close * (1 - random.uniform(0.5, 3) / 100), 2)
        open_ = round(low + random.random() * (high - low), 2)
        volume = random.randint(500000, 5000000)

        daily_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": open_,
            "high": max(high, open_, close),
            "low": min(low, open_, close),
            "close": close,
            "volume": volume,
        })
        closes.append(close)

    # 按周聚合生成周K数据
    df = pd.DataFrame(daily_data)
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.isocalendar().week

    weekly_data = []
    for _, group in df.groupby("week"):
        weekly_data.append({
            "date": group["date"].iloc[-1].strftime("%Y-%m-%d"),
            "open": group["open"].iloc[0],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group["close"].iloc[-1],
            "volume": int(group["volume"].sum()),
        })

    return daily_data, weekly_data


def main():
    """主入口：交互式输入股票代码，生成波动率分析图表"""
    print("=" * 50)
    print("  股票波动率可视化分析工具")
    print("=" * 50)

    # ---- 检查依赖 ----
    missing = []
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import plotly  # noqa: F401
    except ImportError:
        missing.append("plotly")

    if missing:
        print(f"\n[ERROR] 缺少依赖包: {', '.join(missing)}")
        print(f"  请执行: pip install {' '.join(missing)}")
        return

    # ---- 用户输入 ----
    print("\n支持的股票代码格式：")
    print("  港股: 01810 (小米集团)")
    print("  A股:  600519 (贵州茅台)")
    print("  美股:  AAPL (苹果)")
    print("  输入 d 使用演示数据")
    print()

    user_input = input("请输入股票代码: ").strip()

    if user_input.lower() == 'd':
        print("[INFO] 使用演示数据...")
        daily_data, weekly_data = generate_demo_data()
        stock_name = "示例股票"
        stock_code = "600000.SH"
    else:
        # 检查 akshare
        try:
            import akshare as ak  # noqa: F401
        except ImportError:
            print("[ERROR] 获取真实数据需要 akshare，请执行: pip install akshare")
            print("[INFO] 降级使用演示数据...")
            daily_data, weekly_data = generate_demo_data()
            stock_name = "示例股票"
            stock_code = "600000.SH"
        else:
            print(f"[INFO] 获取真实行情: {user_input} ...")
            try:
                daily_data, weekly_data, auto_name, auto_code = fetch_real_data(user_input, days=60)
            except Exception as e:
                print(f"[ERROR] 获取数据失败: {e}")
                print("[INFO] 降级使用演示数据...")
                daily_data, weekly_data = generate_demo_data()
                stock_name = "示例股票"
                stock_code = "600000.SH"
            else:
                name_input = input(f"请输入股票名称（直接回车默认为 {auto_name}）: ").strip()
                stock_name = name_input if name_input else auto_name
                stock_code = auto_code

    print(f"[INFO] 日K数据: {len(daily_data)} 条 ({daily_data[0]['date']} ~ {daily_data[-1]['date']})")
    print(f"[INFO] 周K数据: {len(weekly_data)} 条 ({weekly_data[0]['date']} ~ {weekly_data[-1]['date']})")

    # ---- 生成图表 ----
    print("[INFO] 正在生成波动率分析图表...")
    html, summary = generate_chart_html(
        stock_name=stock_name,
        stock_code=stock_code,
        daily_data=daily_data,
        weekly_data=weekly_data,
    )

    # 输出到脚本所在目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"stock_analysis_{stock_code.replace('.', '_')}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[OK] 图表已保存: {output_path}")
    print(f"[OK] 请用浏览器打开该文件查看交互式图表")
    print(f"\n===== 分析摘要 =====")
    print(f"  股票: {summary['stock_name']} ({summary['stock_code']})")
    print(f"  当前日波动率: {summary['current_daily_vol']}% (20日均值: {summary['daily_vol_ma20']}%)")
    print(f"  当前周波动率: {summary['current_weekly_vol']}% (12周中位: {summary['weekly_median']}%)")
    print(f"  买入信号: {summary['buy_signals']}个 | 卖出信号: {summary['sell_signals']}个 | 持有: {summary['hold_signals']}个")
    print(f"  建议: {summary['advice']}")


if __name__ == "__main__":
    import os
    main()

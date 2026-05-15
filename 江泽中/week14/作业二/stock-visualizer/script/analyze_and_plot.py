import sys
import akshare as ak
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def analyze_stock(stock_code):
    """
    分析指定股票的日/周波动率，绘制可视化图表，并给出买卖建议。
    """
    print(f"正在获取 {stock_code} 的行情数据...")

    # 1. 获取日线数据 (以 A 股为例，使用 akshare 接口)
    try:
        # 获取前复权的日线数据
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
    except Exception as e:
        print(f"❌ 数据获取失败，请检查股票代码是否正确或网络连接。错误信息: {e}")
        return

    if df.empty:
        print("❌ 未获取到有效数据，请检查股票代码。")
        return

    # 数据预处理
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)

    # 2. 计算日波动率与周波动率
    # 日波动率 = (当日最高价 - 当日最低价) / 前一日收盘价 * 100
    df['日波动率'] = (df['最高'] - df['最低']) / df['收盘'].shift(1) * 100

    # 周波动率 (用5日滚动窗口的最大振幅来模拟周级别的波动感知)
    # 5日最高价 - 5日最低价 / 5日前的收盘价
    df['周波动率'] = (df['最高'].rolling(5).max() - df['最低'].rolling(5).min()) / df['收盘'].shift(4) * 100

    # 剔除因计算产生的空值
    df.dropna(inplace=True)
    recent_data = df.tail(60)  # 取最近 60 个交易日的数据进行绘图

    # 3. 绘制双周期波动图
    plt.figure(figsize=(14, 7))

    # 绘制日波动率（蓝色细线，带透明度）
    plt.plot(recent_data.index, recent_data['日波动率'],
             label='Daily Volatility', color='skyblue', alpha=0.7, linewidth=1)

    # 绘制周波动率（红色粗线，突出显示）
    plt.plot(recent_data.index, recent_data['周波动率'],
             label='Weekly Volatility (5-day)', color='red', linewidth=2.5)

    plt.title(f'{stock_code} - Daily vs Weekly Volatility Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility (%)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存图片到当前目录
    img_path = f"{stock_code}_volatility.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 波动率趋势图已生成并保存为: {img_path}")

    # 4. 基于波动率大小给出买卖建议逻辑
    latest_daily_vol = recent_data['日波动率'].iloc[-1]
    latest_weekly_vol = recent_data['周波动率'].iloc[-1]
    avg_daily_vol = recent_data['日波动率'].mean()

    # 计算波动率相对均值的偏离程度
    vol_ratio = latest_daily_vol / avg_daily_vol

    print("\n" + "=" * 40)
    print("📊 最佳买卖时间建议 (基于波动率模型)")
    print("=" * 40)

    # 策略逻辑：
    # 1. 极度缩量（波动率极低）：往往是变盘前兆，适合左侧潜伏（买入观察期）
    # 2. 极度放量（波动率极高）：情绪过热或恐慌，适合右侧止盈或规避风险（卖出/减仓期）
    # 3. 正常区间：趋势延续

    if vol_ratio < 0.6:
        print(f"💡 [买入观察期 / 吸筹阶段]\n"
              f"   当前日波动率 ({latest_daily_vol:.2f}%) 远低于近期均值 ({avg_daily_vol:.2f}%)。\n"
              f"   市场情绪极度低迷，往往是变盘的前兆（低波动吸筹）。建议密切关注后续是否出现放量突破信号。")
    elif latest_weekly_vol > 12 or vol_ratio > 2.0:
        print(f"⚠️ [卖出 / 减仓观察期]\n"
              f"   当前周波动率高达 {latest_weekly_vol:.2f}%，或日波动率是均值的 {vol_ratio:.1f} 倍。\n"
              f"   市场情绪极度不稳定，多空分歧巨大，追高风险极高。建议等待波动率回归均值后再做决策。")
    else:
        print(f"👍 [持有 / 正常交易期]\n"
              f"   当前波动率 ({latest_daily_vol:.2f}%) 处于正常区间。\n"
              f"   市场运行平稳，可结合均线趋势进行正常的持股或波段操作。")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ 请提供股票代码作为参数，例如：python analyze_and_plot.py 600519")
    else:
        stock_code = sys.argv[1]
        analyze_stock(stock_code)
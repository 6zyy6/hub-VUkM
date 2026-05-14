"""
融合通达信指标 + 波动分析 + 抗反爬数据获取
依赖安装: pip install akshare pandas matplotlib numpy
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import random
import warnings

warnings.filterwarnings('ignore')

# ---------- 中文字体设置 ----------
try:
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    rcParams['axes.unicode_minus'] = False
except:
    pass

# ================== 抗反爬数据获取函数 ==================
def get_robust_stock_data(symbol, period_days=90, max_retries=5):
    """
    带重试机制的A股数据获取，自动处理连接中断和超时
    symbol: 6位数字代码，如 '000001'
    period_days: 获取最近多少天的数据
    """
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=period_days + 30)).strftime('%Y%m%d')

    print(f"正在尝试获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")

    for attempt in range(max_retries):
        try:
            # 每次尝试前随机等待0.5~3秒，降低请求频率
            if attempt > 0:
                wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                print(f"第 {attempt+1} 次重试，等待 {wait_time:.2f} 秒...")
                time.sleep(wait_time)

            df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                    start_date=start_date, end_date=end_date,
                                    adjust="qfq")  # 前复权

            if df is not None and not df.empty:
                # 重命名列以匹配后续代码
                df.rename(columns={
                    '日期': 'Date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume'
                }, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.sort_index(inplace=True)
                print(f"{symbol} 数据获取成功，共 {len(df)} 条记录。")
                return df
            else:
                print(f"第 {attempt+1} 次尝试：返回空数据")

        except Exception as e:
            print(f"第 {attempt+1} 次尝试失败: {type(e).__name__}: {e}")

    print("已达到最大重试次数，数据获取失败。")
    return None

# ================== 通达信指标函数 ==================
def sma_calc(series, n, m):
    """通达信风格的SMA递归计算"""
    result = np.zeros(len(series))
    for i in range(len(series)):
        if i == 0:
            result[i] = series[i]
        else:
            result[i] = (m * series[i] + (n - m) * result[i-1]) / n
    return result

def llv(series, n):
    return series.rolling(window=n, min_periods=1).min()

def hhv(series, n):
    return series.rolling(window=n, min_periods=1).max()

def cross(series1, series2):
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

def get_tongdaxin_indicators(df):
    """计算阻力、支撑、中线、趋势线及买卖信号"""
    df = df.copy()
    H1 = df[['close', 'high']].max(axis=1)
    L1 = df[['close', 'low']].min(axis=1)
    P1 = H1 - L1
    df['阻力'] = L1 + P1 * 7/8
    df['支撑'] = L1 + P1 * 0.5/8
    df['中线'] = (df['支撑'] + df['阻力']) / 2

    n = 55
    low_llv = llv(df['low'], n)
    high_hhv = hhv(df['high'], n)
    rsv = (df['close'] - low_llv) / (high_hhv - low_llv) * 100
    rsv = rsv.fillna(50)
    sma1 = sma_calc(rsv, 5, 1)
    sma2 = sma_calc(sma1, 3, 1)
    V11 = 3 * sma1 - 2 * sma2
    df['趋势线'] = pd.Series(sma_calc(V11, 3, 1), index=df.index)
    df['趋势线变化'] = (df['趋势线'] - df['趋势线'].shift(1)) / df['趋势线'].shift(1) * 100

    # 准备买入
    condition_buy_prep = (df['趋势线'] < 11) & (df['close'] < df['中线'])
    df['准备买入'] = False
    last_buy_prep = -999
    for i in range(len(df)):
        if condition_buy_prep.iloc[i] and (i - last_buy_prep > 15):
            df.loc[df.index[i], '准备买入'] = True
            last_buy_prep = i

    trend = df['趋势线']
    bb1 = (trend.shift(1) < 11) & cross(trend, 11) & (trend.shift(1) > 6)
    bb2 = (trend.shift(1) < 6) & cross(trend, 6) & (trend.shift(1) > 3)
    bb3 = (trend.shift(1) < 3) & cross(trend, 3) & (trend.shift(1) > 1)
    bb4 = (trend.shift(1) < 1) & cross(trend, 1) & (trend.shift(1) > 0)
    bb5 = (trend.shift(1) < 0) & cross(trend, 0)
    bb = (bb1 | bb2 | bb3 | bb4 | bb5)
    df['下单买入'] = bb & (df['close'] < df['中线'])

    # 准备卖出
    condition_sell_prep = (df['趋势线'] > 89) & (df['close'] > df['中线'])
    df['准备卖出'] = False
    last_sell_prep = -999
    for i in range(len(df)):
        if condition_sell_prep.iloc[i] and (i - last_sell_prep > 15):
            df.loc[df.index[i], '准备卖出'] = True
            last_sell_prep = i

    dd1 = (trend.shift(1) > 89) & cross(89, trend) & (trend.shift(1) < 94)
    dd2 = (trend.shift(1) > 94) & cross(94, trend) & (trend.shift(1) < 97)
    dd3 = (trend.shift(1) > 97) & cross(97, trend) & (trend.shift(1) < 99)
    dd4 = (trend.shift(1) > 99) & cross(99, trend) & (trend.shift(1) < 100)
    dd5 = (trend.shift(1) > 100) & cross(100, trend)
    dd = (dd1 | dd2 | dd3 | dd4 | dd5)
    df['下单卖出'] = dd & (df['close'] > df['中线'])

    return df

# ================== 波动分析函数 ==================
def calc_volatility(df, vol_type='amplitude'):
    """计算日波动与周波动"""
    if vol_type == 'amplitude':
        df['Daily_Vol'] = (df['high'] - df['low']) / df['close'].shift(1) * 100
    else:
        df['Return'] = df['close'].pct_change() * 100
        df['Daily_Vol'] = df['Return'].rolling(5).std()
    df.dropna(subset=['Daily_Vol'], inplace=True)

    weekly = df.resample('W-FRI').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'Daily_Vol': 'mean'
    }).dropna()
    if vol_type == 'amplitude':
        weekly['Weekly_Vol'] = (weekly['high'] - weekly['low']) / weekly['open'] * 100
    else:
        weekly['Return'] = weekly['close'].pct_change() * 100
        weekly['Weekly_Vol'] = weekly['Return'].rolling(2).std()
    weekly.dropna(subset=['Weekly_Vol'], inplace=True)
    return df, weekly

# ================== 综合Skill ==================
def advanced_stock_skill(symbol, period_days=90, vol_type='amplitude',
                         buy_threshold=0.3, sell_threshold=0.7,
                         show_plot=True, save_plot=True):
    """
    融合通达信指标 + 波动分析的主函数
    """
    # 使用抗反爬函数获取数据
    df = get_robust_stock_data(symbol, period_days)
    if df is None or len(df) < 30:
        print("数据不足，无法分析。")
        return None

    # 计算通达信指标
    df = get_tongdaxin_indicators(df)
    # 计算波动指标
    df, weekly = calc_volatility(df, vol_type)

    last = df.iloc[-1]
    latest_price = last['close']
    latest_trend = last['趋势线']
    latest_mid = last['中线']
    daily_vol = last['Daily_Vol']

    # 波动建议
    recent_vol = df['Daily_Vol'].tail(20).dropna()
    low_q = recent_vol.quantile(buy_threshold)
    high_q = recent_vol.quantile(sell_threshold)
    ma20 = df['close'].rolling(20).mean().iloc[-1]

    if daily_vol < low_q and latest_price < ma20:
        vol_advice = "波动低位，可考虑买入"
    elif daily_vol > high_q and latest_price > ma20:
        vol_advice = "波动高位，注意风险"
    else:
        vol_advice = "波动适中，观望"

    # 通达信信号
    td_buy = last['下单买入']
    td_prep_buy = last['准备买入']
    td_sell = last['下单卖出']
    td_prep_sell = last['准备卖出']

    # 综合建议
    if td_buy:
        final_advice = "【通达信信号】强烈买入"
        reason = f"趋势线={latest_trend:.1f}，出现下单买入信号，且价格({latest_price:.2f})低于中线({latest_mid:.2f})。"
    elif td_prep_buy:
        final_advice = "准备买入"
        reason = f"趋势线={latest_trend:.1f}（低于11），接近底部区域，可分批建仓。"
    elif td_sell:
        final_advice = "【通达信信号】强烈卖出"
        reason = f"趋势线={latest_trend:.1f}，出现下单卖出信号，且价格高于中线。"
    elif td_prep_sell:
        final_advice = "准备卖出"
        reason = f"趋势线={latest_trend:.1f}（高于89），处于超买区域，注意回调。"
    else:
        final_advice = "持有/观望"
        reason = f"趋势线={latest_trend:.1f}，无明确买卖信号。{vol_advice}"

    # ================== 绘图 ==================
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['close'], color='black', linewidth=1.5, label='收盘价')
    ax1.plot(df.index, df['阻力'], color='green', linestyle='--', linewidth=1, label='阻力')
    ax1.plot(df.index, df['支撑'], color='red', linestyle='--', linewidth=1, label='支撑')
    ax1.plot(df.index, df['中线'], color='blue', linestyle=':', linewidth=1, label='中线')
    ax1.set_ylabel('价格')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['趋势线'], color='magenta', linewidth=2, label='趋势线')
    ax1_twin.axhline(11, color='red', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1_twin.axhline(89, color='green', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1_twin.set_ylabel('趋势线', color='magenta')
    ax1_twin.tick_params(axis='y', labelcolor='magenta')

    buy_points = df[df['下单买入'] == True]
    sell_points = df[df['下单卖出'] == True]
    prep_buy = df[df['准备买入'] == True]
    prep_sell = df[df['准备卖出'] == True]

    ax1.scatter(buy_points.index, buy_points['close'], marker='^', color='red',
                s=120, label='下单买入', zorder=5)
    ax1.scatter(sell_points.index, sell_points['close'], marker='v', color='green',
                s=120, label='下单卖出', zorder=5)
    ax1.scatter(prep_buy.index, prep_buy['close'], marker='*', color='orange',
                s=80, label='准备买入', zorder=4)
    ax1.scatter(prep_sell.index, prep_sell['close'], marker='*', color='gray',
                s=80, label='准备卖出', zorder=4)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(df.index, df['Daily_Vol'], alpha=0.4, color='blue', width=0.8, label='日波动 (%)')
    if not weekly.empty:
        ax2.plot(weekly.index, weekly['Weekly_Vol'], color='red', marker='o', linewidth=2, label='周波动 (%)')
    ax2.set_ylabel('波动率 (%)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{symbol} 通达信指标 + 波动分析 | 当前建议: {final_advice}', fontsize=14, y=0.98)

    textstr = (
        f"最新价: {latest_price:.2f}\n"
        f"趋势线: {latest_trend:.1f}  (底11 / 顶89)\n"
        f"中线: {latest_mid:.2f}\n"
        f"日波动: {daily_vol:.2f}%  (低位{low_q:.1f}% / 高位{high_q:.1f}%)\n"
        f"建议: {final_advice}\n"
        f"原因: {reason}"
    )
    plt.figtext(0.02, 0.02, textstr, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                verticalalignment='bottom')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_plot:
        filename = f"{symbol}_advanced_analysis.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"图表已保存为 {filename}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    print("\n" + "="*60)
    print(f"股票代码: {symbol}")
    print(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"最新趋势线: {latest_trend:.2f}  (底部区域<11, 顶部区域>89)")
    print(f"中线: {latest_mid:.2f}  当前价: {latest_price:.2f}")
    print(f"日波动分位: 低位阈值{low_q:.2f}% / 高位阈值{high_q:.2f}%")
    print(f"通达信信号: 下单买入={td_buy} 准备买入={td_prep_buy} 下单卖出={td_sell} 准备卖出={td_prep_sell}")
    print(f"综合建议: {final_advice}")
    print(f"详细: {reason}")
    print("="*60)

    return {
        'symbol': symbol,
        'final_advice': final_advice,
        'trend_line': latest_trend,
        'mid_line': latest_mid,
        'daily_vol': daily_vol,
        'buy_signal': td_buy,
        'sell_signal': td_sell
    }

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 修改为你要分析的A股代码（6位数字）
    result = advanced_stock_skill(
        symbol="000001",    # 平安银行
        period_days=120,    # 最近120个交易日
        vol_type="amplitude",
        show_plot=True,
        save_plot=True
    )
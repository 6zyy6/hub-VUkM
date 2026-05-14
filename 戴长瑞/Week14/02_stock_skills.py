"""
股票波动分析器 - 使用 akshare 数据源（A股）
功能：日波动/周波动可视化 + 基于波动分位数与均线的买卖建议
依赖安装: pip install akshare pandas matplotlib numpy
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import time

warnings.filterwarnings('ignore')

# ---------- 设置中文字体 ----------
try:
    rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
    rcParams['axes.unicode_minus'] = False
except:
    pass


def get_a_stock_data(symbol, period_days=90):
    """
    使用 akshare 获取 A 股日线数据
    symbol: 6位数字代码，如 '000001' (平安银行)
    period_days: 获取多少天的数据（默认90天）
    返回: DataFrame 包含 Open, High, Low, Close, Volume
    """
    # 确定起始日期
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=period_days + 30)).strftime('%Y%m%d')

    print(f"正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")

    try:
        # akshare 获取 A 股日线历史数据
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start_date, end_date=end_date,
                                adjust="qfq")  # 前复权
        if df.empty:
            raise ValueError("未获取到数据，请检查股票代码或网络。")

        # 重命名列以匹配后续代码
        df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume'
        }, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.sort_index(inplace=True)
        return df

    except Exception as e:
        print(f"获取数据失败: {e}")
        return None


def stock_volatility_analyzer(symbol, period_days=90, volatility_type='amplitude',
                              buy_threshold=0.3, sell_threshold=0.7,
                              show_plot=True, save_plot=True):
    """
    股票波动分析主函数（A股版本）

    参数:
        symbol: 6位数字股票代码，如 '000001'
        period_days: 数据天数（默认90天）
        volatility_type: 波动类型 'amplitude'（振幅）或 'std'（标准差）
        buy_threshold: 买入阈值（分位数，默认0.3）
        sell_threshold: 卖出阈值（分位数，默认0.7）
        show_plot: 是否显示图表
        save_plot: 是否保存图表为文件
    """
    # 1. 获取数据
    df = get_a_stock_data(symbol, period_days)
    if df is None or df.empty:
        print("错误：无法获取股票数据。")
        return None

    # 2. 计算日波动
    if volatility_type == 'amplitude':
        # 振幅百分比 = (最高-最低) / 前一日收盘价 * 100
        df['Daily_Vol'] = (df['High'] - df['Low']) / df['Close'].shift(1) * 100
    else:  # 'std'
        df['Return'] = df['Close'].pct_change() * 100
        df['Daily_Vol'] = df['Return'].rolling(window=5).std()

    df.dropna(inplace=True)

    if len(df) < 10:
        print("数据点不足，无法分析。")
        return None

    # 3. 计算周波动（以周五为周结束）
    # 确保索引是 datetime
    df.index = pd.to_datetime(df.index)
    weekly = df.resample('W-FRI').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Daily_Vol': 'mean'
    }).dropna()

    if volatility_type == 'amplitude':
        weekly['Weekly_Vol'] = (weekly['High'] - weekly['Low']) / weekly['Open'] * 100
    else:
        weekly['Return'] = weekly['Close'].pct_change() * 100
        weekly['Weekly_Vol'] = weekly['Return'].rolling(window=2).std()
    weekly.dropna(subset=['Weekly_Vol'], inplace=True)

    # 4. 确定买卖建议
    recent_vol = df['Daily_Vol'].tail(min(20, len(df)))
    low_q = recent_vol.quantile(buy_threshold)
    high_q = recent_vol.quantile(sell_threshold)

    current_daily_vol = df['Daily_Vol'].iloc[-1]
    current_price = df['Close'].iloc[-1]
    ma20 = df['Close'].rolling(window=min(20, len(df))).mean().iloc[-1]

    # 判断信号
    if current_daily_vol < low_q and current_price < ma20:
        advice = "买入"
        reason = f"日波动处于低位 (分位 {buy_threshold:.0%})，且股价低于20日均线，可能为震荡末期。"
    elif current_daily_vol > high_q and current_price > ma20:
        advice = "卖出"
        reason = f"日波动处于高位 (分位 {sell_threshold:.0%})，且股价高于20日均线，警惕高位放量震荡。"
    else:
        advice = "持有/观望"
        reason = "波动水平适中或价格与均线关系不明确，建议观望。"

    # 周波动额外提示
    if not weekly.empty:
        current_weekly_vol = weekly['Weekly_Vol'].iloc[-1]
        avg_daily_vol = df['Daily_Vol'].tail(5).mean()
        if current_weekly_vol > 2 * avg_daily_vol:
            reason += " 周级别波动显著放大，注意趋势变化。"
    else:
        current_weekly_vol = None

    # 5. 绘图
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(df.index, df['Close'], color='black', linewidth=1.5, label='收盘价')
    ax1.set_xlabel('日期')
    ax1.set_ylabel('价格', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.bar(df.index, df['Daily_Vol'], alpha=0.3, color='blue', width=0.8, label='日波动 (%)')
    ax2.set_ylabel('波动率 (%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    if not weekly.empty:
        ax2.plot(weekly.index, weekly['Weekly_Vol'], color='red', marker='o', linewidth=2, markersize=4, label='周波动 (%)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    title = f'{symbol} 波动分析 | 当前建议: {advice}'
    plt.title(title, fontsize=14)

    textstr = f"最新日波动: {current_daily_vol:.2f}%\n"
    if current_weekly_vol:
        textstr += f"最新周波动: {current_weekly_vol:.2f}%\n"
    textstr += f"建议: {advice}\n原因: {reason}"
    plt.figtext(0.02, 0.02, textstr, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')

    plt.tight_layout()

    if save_plot:
        filename = f"{symbol}_volatility_analysis.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"图表已保存为 {filename}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    # 控制台输出
    print("\n" + "=" * 50)
    print(f"股票代码: {symbol}")
    print(f"数据周期: 最近 {len(df)} 个交易日")
    print(f"最新日波动: {current_daily_vol:.2f}% (低位阈值: {low_q:.2f}%, 高位阈值: {high_q:.2f}%)")
    if current_weekly_vol:
        print(f"最新周波动: {current_weekly_vol:.2f}%")
    print(f"建议: {advice}")
    print(f"详细: {reason}")
    print("=" * 50)

    return {
        'symbol': symbol,
        'current_price': current_price,
        'current_daily_vol': current_daily_vol,
        'current_weekly_vol': current_weekly_vol,
        'advice': advice,
        'reason': reason,
        'fig': fig
    }


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 修改为你要分析的 A 股代码（6位数字）
    STOCK_CODE = "000001"   # 平安银行
    PERIOD_DAYS = 90        # 获取最近90个交易日数据
    VOL_TYPE = "amplitude"  # 振幅波动

    result = stock_volatility_analyzer(
        symbol=STOCK_CODE,
        period_days=PERIOD_DAYS,
        volatility_type=VOL_TYPE,
        buy_threshold=0.3,
        sell_threshold=0.7,
        show_plot=True,
        save_plot=True
    )
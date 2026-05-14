import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StockVolatilitySkill:
    """
    股票波动率分析与交易信号 Skill
    """
    def __init__(self, ticker, start_date, end_date=None,
                 window=20, low_percentile=0.2, high_percentile=0.8):
        """
        参数:
            ticker: str, 股票代码 (e.g., 'AAPL', '000001.SS')
            start_date: str, 开始日期 'YYYY-MM-DD'
            end_date: str, 结束日期 'YYYY-MM-DD' (默认为今天)
            window: int, 滚动窗口大小 (交易日)
            low_percentile: float, 买入信号阈值分位数 (默认 0.2)
            high_percentile: float, 卖出信号阈值分位数 (默认 0.8)
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.window = window
        self.low_pct = low_percentile
        self.high_pct = high_percentile
        
        self.data = None
        self.daily_vol = None
        self.weekly_vol = None
        self.signals = None
        
    def fetch_data(self):
        """从 Yahoo Finance 获取股票数据"""
        self.data = yf.download(self.ticker, start=self.start_date,
                                end=self.end_date, progress=False)
        if self.data.empty:
            raise ValueError(f"未获取到股票 {self.ticker} 的数据，请检查代码或网络")
        # 只保留 OHLC 列
        self.data = self.data[['Open','High','Low','Close']]
        return self.data
    
    def calculate_volatility(self):
        """计算日波动率和周波动率（均为百分比形式）"""
        # 日波动率: 日内振幅百分比
        self.data['Daily_Vol'] = (self.data['High'] - self.data['Low']) / self.data['Close'] * 100
        self.daily_vol = self.data['Daily_Vol']
        
        # 周波动率: 以周五为周末，计算每周振幅百分比
        weekly_df = self.data.resample('W-FRI').agg({
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        weekly_df['Weekly_Vol'] = (weekly_df['High'] - weekly_df['Low']) / weekly_df['Close'] * 100
        self.weekly_vol = weekly_df['Weekly_Vol']
        return self.daily_vol, self.weekly_vol
    
    def generate_signals(self):
        """基于滚动分位数生成买卖信号（缓存结果）"""
        if self.signals is not None:
            return self.signals
        
        daily = self.daily_vol.copy()
        # 滚动分位数 (需至少半窗长度)
        min_periods = max(1, int(self.window/2))
        rolling_low = daily.rolling(window=self.window, min_periods=min_periods).quantile(self.low_pct)
        rolling_high = daily.rolling(window=self.window, min_periods=min_periods).quantile(self.high_pct)
        
        # 买入信号：波动率从高于低阈值下穿低阈值
        buy_cross = (daily < rolling_low) & (daily.shift(1) >= rolling_low.shift(1))
        # 卖出信号：波动率从低于高阈值上穿高阈值
        sell_cross = (daily > rolling_high) & (daily.shift(1) <= rolling_high.shift(1))
        
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['Buy'] = buy_cross
        self.signals['Sell'] = sell_cross
        return self.signals
    
    def plot_volatility(self, save_path=None):
        """绘制日/周波动率及买卖信号"""
        if self.signals is None:
            self.generate_signals()
            
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # 左轴: 日波动率折线
        ax1.plot(self.data.index, self.daily_vol,
                 color='blue', alpha=0.6, linewidth=1, label='日波动率 (%)')
        ax1.set_xlabel('日期')
        ax1.set_ylabel('日波动率 (%)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 右轴: 周波动率散点+连线
        ax2 = ax1.twinx()
        weekly_idx = self.weekly_vol.index
        weekly_vals = self.weekly_vol.values
        ax2.scatter(weekly_idx, weekly_vals, color='orange',
                    s=60, marker='o', label='周波动率 (%)', zorder=3)
        ax2.plot(weekly_idx, weekly_vals, color='orange',
                 linestyle='--', alpha=0.7, linewidth=1)
        ax2.set_ylabel('周波动率 (%)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # 标记买卖信号
        buy_dates = self.signals[self.signals['Buy']].index
        sell_dates = self.signals[self.signals['Sell']].index
        buy_vals = [self.daily_vol.loc[d] for d in buy_dates if d in self.daily_vol.index]
        sell_vals = [self.daily_vol.loc[d] for d in sell_dates if d in self.daily_vol.index]
        
        ax1.scatter(buy_dates, buy_vals, marker='^', color='green',
                    s=100, label='买入信号', zorder=4)
        ax1.scatter(sell_dates, sell_vals, marker='v', color='red',
                    s=100, label='卖出信号', zorder=4)
        
        # 标题与图例
        plt.title(f'{self.ticker} 日波动率与周波动率 (含交易信号)')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 日期格式化
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至 {save_path}")
        else:
            plt.show()
    
    def get_best_timing_advice(self):
        """基于波动大小输出最佳买卖时间建议"""
        if self.signals is None:
            self.generate_signals()
            
        advice = {"buy_suggestions": [], "sell_suggestions": []}
        
        buy_dates = self.signals[self.signals['Buy']].index
        for d in buy_dates:
            vol_val = self.daily_vol.loc[d]
            advice["buy_suggestions"].append({
                "date": d.strftime('%Y-%m-%d'),
                "daily_volatility": f"{vol_val:.2f}%",
                "reason": "当日波动率显著低于近期平均水平，市场可能处于低波动整理阶段，适合买入布局。"
            })
        
        sell_dates = self.signals[self.signals['Sell']].index
        for d in sell_dates:
            vol_val = self.daily_vol.loc[d]
            advice["sell_suggestions"].append({
                "date": d.strftime('%Y-%m-%d'),
                "daily_volatility": f"{vol_val:.2f}%",
                "reason": "当日波动率显著高于近期平均水平，市场情绪高涨风险加大，适合卖出获利或止损。"
            })
        return advice
    
    def run(self, save_plot_path=None):
        """执行完整分析流程"""
        print(f"正在分析股票 {self.ticker} ...")
        self.fetch_data()
        self.calculate_volatility()
        self.generate_signals()
        self.plot_volatility(save_plot_path)
        
        advice = self.get_best_timing_advice()
        print("\n===== 最佳买卖时间建议 =====")
        if advice["buy_suggestions"]:
            print("买入建议：")
            for item in advice["buy_suggestions"]:
                print(f"  日期 {item['date']}，波动率 {item['daily_volatility']}：{item['reason']}")
        else:
            print("未触发买入信号（波动率未低于阈值）。")
        
        if advice["sell_suggestions"]:
            print("\n卖出建议：")
            for item in advice["sell_suggestions"]:
                print(f"  日期 {item['date']}，波动率 {item['daily_volatility']}：{item['reason']}")
        else:
            print("未触发卖出信号（波动率未高于阈值）。")
        return advice

# ========== 使用示例 ==========
if __name__ == "__main__":
    # 分析苹果公司股票 (2026年全年)
    skill = StockVolatilitySkill('AAPL', '2026-01-01', '2026-05-01')
    skill.run(save_plot_path='aapl_volatility_signals.png')
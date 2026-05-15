"""
股票可视化Skill - 包含波动图和买卖建议
作业2: 定义一个skill，包含对股票的可视化功能，
对于股票的周波动、日波动绘制在一个图中，
并基于大小给出一个买入卖出的最佳时间建议
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Literal
from dataclasses import dataclass
import random

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from langchain_core.tools import StructuredTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field


@dataclass
class StockData:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class VolatilityAnalysis:
    daily_volatility: float
    weekly_volatility: float
    avg_daily_change: float
    avg_weekly_change: float
    max_daily_up: float
    max_daily_down: float
    max_weekly_up: float
    max_weekly_down: float
    risk_level: str


@dataclass
class TradingSuggestion:
    action: Literal["买入", "卖出", "观望"]
    confidence: float
    reason: str
    expected_return: Optional[float] = None
    best_time: Optional[str] = None


class StockVisualizationSkill:
    def __init__(self):
        self.name = "stock_visualization"
        self.description = "股票可视化工具，提供波动性分析和交易建议"

    def _generate_sample_data(
        self,
        stock_code: str,
        days: int = 30
    ) -> List[StockData]:
        base_price = 100.0
        data = []
        current_date = datetime.now()

        for i in range(days):
            date = current_date - timedelta(days=days - i - 1)
            daily_change = random.uniform(-0.03, 0.035)
            open_price = base_price * (1 + random.uniform(-0.01, 0.01))
            close_price = open_price * (1 + daily_change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.015))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.015))
            volume = int(random.uniform(5000000, 15000000))

            data.append(StockData(
                date=date.strftime("%Y-%m-%d"),
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                volume=volume
            ))
            base_price = close_price

        return data

    def _calculate_volatility(
        self,
        data: List[StockData]
    ) -> VolatilityAnalysis:
        if len(data) < 2:
            return VolatilityAnalysis(
                daily_volatility=0.0,
                weekly_volatility=0.0,
                avg_daily_change=0.0,
                avg_weekly_change=0.0,
                max_daily_up=0.0,
                max_daily_down=0.0,
                max_weekly_up=0.0,
                max_weekly_down=0.0,
                risk_level="低"
            )

        daily_changes = []
        for i in range(1, len(data)):
            change = (data[i].close - data[i-1].close) / data[i-1].close
            daily_changes.append(change)

        avg_daily = sum(daily_changes) / len(daily_changes)
        variance = sum((x - avg_daily) ** 2 for x in daily_changes) / len(daily_changes)
        daily_volatility = variance ** 0.5

        max_daily_up = max(daily_changes) if daily_changes else 0.0
        max_daily_down = min(daily_changes) if daily_changes else 0.0

        weekly_changes = []
        weekly_data_count = 5
        for i in range(weekly_data_count, len(data), weekly_data_count):
            weekly_change = (data[i].close - data[i-weekly_data_count].close) / data[i-weekly_data_count].close
            weekly_changes.append(weekly_change)

        if weekly_changes:
            avg_weekly = sum(weekly_changes) / len(weekly_changes)
            weekly_variance = sum((x - avg_weekly) ** 2 for x in weekly_changes) / len(weekly_changes)
            weekly_volatility = weekly_variance ** 0.5
            max_weekly_up = max(weekly_changes)
            max_weekly_down = min(weekly_changes)
        else:
            avg_weekly = 0.0
            weekly_volatility = 0.0
            max_weekly_up = 0.0
            max_weekly_down = 0.0

        risk_level = "低"
        if daily_volatility > 0.03:
            risk_level = "高"
        elif daily_volatility > 0.015:
            risk_level = "中"

        return VolatilityAnalysis(
            daily_volatility=round(daily_volatility * 100, 2),
            weekly_volatility=round(weekly_volatility * 100, 2),
            avg_daily_change=round(avg_daily * 100, 2),
            avg_weekly_change=round(avg_weekly * 100, 2),
            max_daily_up=round(max_daily_up * 100, 2),
            max_daily_down=round(max_daily_down * 100, 2),
            max_weekly_up=round(max_weekly_up * 100, 2),
            max_weekly_down=round(max_weekly_down * 100, 2),
            risk_level=risk_level
        )

    def _generate_trading_suggestion(
        self,
        volatility: VolatilityAnalysis,
        current_price: float,
        price_trend: str
    ) -> TradingSuggestion:
        if volatility.risk_level == "高":
            if volatility.avg_daily_change > 0:
                return TradingSuggestion(
                    action="观望",
                    confidence=0.7,
                    reason="波动性较高且上涨，建议等待回调后买入",
                    best_time="等待2-3个交易日后观察"
                )
            else:
                return TradingSuggestion(
                    action="卖出",
                    confidence=0.75,
                    reason="波动性较高且下跌，建议止损或减仓",
                    best_time="立即执行"
                )

        elif volatility.risk_level == "中":
            if volatility.max_daily_up > abs(volatility.max_daily_down):
                return TradingSuggestion(
                    action="买入",
                    confidence=0.65,
                    reason="上涨幅度大于下跌幅度，上涨趋势较强",
                    expected_return=volatility.avg_weekly_change * 0.8,
                    best_time="周一开盘或周二开盘"
                )
            else:
                return TradingSuggestion(
                    action="观望",
                    confidence=0.6,
                    reason="下跌幅度较大，建议等待趋势明朗",
                    best_time="等待2个交易日后重新评估"
                )

        else:
            if price_trend == "up" and volatility.avg_daily_change > 0.005:
                return TradingSuggestion(
                    action="买入",
                    confidence=0.7,
                    reason="波动性低且趋势向上，适合建仓",
                    expected_return=volatility.avg_weekly_change * 1.2,
                    best_time="周一开盘"
                )
            elif price_trend == "down":
                return TradingSuggestion(
                    action="卖出",
                    confidence=0.65,
                    reason="趋势向下且波动性低，建议减仓",
                    best_time="立即执行或周二开盘"
                )
            else:
                return TradingSuggestion(
                    action="观望",
                    confidence=0.6,
                    reason="趋势不明确，建议等待",
                    best_time="等待3个交易日后重新评估"
                )

    def plot_volatility_chart(
        self,
        stock_code: str,
        data: Optional[List[StockData]] = None,
        show_chart: bool = False
    ) -> Dict[str, Any]:
        if not MATPLOTLIB_AVAILABLE:
            return {
                "success": False,
                "error": "matplotlib未安装，无法生成图表",
                "stock_code": stock_code
            }

        if data is None:
            data = self._generate_sample_data(stock_code)

        dates = [datetime.strptime(d.date, "%Y-%m-%d") for d in data]
        closes = [d.close for d in data]
        volumes = [d.volume for d in data]

        daily_changes = [0.0]
        for i in range(1, len(data)):
            change = (data[i].close - data[i-1].close) / data[i-1].close * 100
            daily_changes.append(change)

        weekly_changes = [0.0] * 4
        weekly_data_count = 5
        weekly_dates = []
        weekly_volatility = []
        weekly_avg = 0.0

        for i in range(weekly_data_count, len(data), weekly_data_count):
            change = (data[i].close - data[i-weekly_data_count].close) / data[i-weekly_data_count].close * 100
            weekly_changes.append(change)
            weekly_dates.append(dates[i])
            weekly_volatility.append(change)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'{stock_code} 股票波动性分析', fontsize=16, fontweight='bold')

        ax1.plot(dates, closes, 'b-', linewidth=1.5, label='收盘价')
        ax1.set_ylabel('价格 (元)', fontsize=12)
        ax1.set_title('价格走势', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        colors = ['green' if c >= 0 else 'red' for c in daily_changes]
        ax2.bar(dates, daily_changes, color=colors, alpha=0.7, width=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('日涨跌 (%)', fontsize=12)
        ax2.set_title('日波动', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        ax3.bar(weekly_dates, weekly_volatility, color='steelblue', alpha=0.8, width=3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.axhline(y=weekly_avg, color='red', linestyle='--', linewidth=1, label=f'周均值: {weekly_avg:.2f}%')
        ax3.set_ylabel('周涨跌 (%)', fontsize=12)
        ax3.set_xlabel('日期', fontsize=12)
        ax3.set_title('周波动', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        chart_path = f"{stock_code}_volatility.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        if show_chart:
            plt.show()
        plt.close()

        return {
            "success": True,
            "chart_path": chart_path,
            "stock_code": stock_code
        }

    def analyze_stock(
        self,
        stock_code: str,
        days: int = 30,
        show_chart: bool = False
    ) -> Dict[str, Any]:
        data = self._generate_sample_data(stock_code, days)
        volatility = self._calculate_volatility(data)

        current_price = data[-1].close if data else 0.0
        if len(data) >= 2:
            recent_changes = [(data[i].close - data[i-1].close) / data[i-1].close for i in range(max(1, len(data)-5), len(data))]
            avg_recent = sum(recent_changes) / len(recent_changes)
            price_trend = "up" if avg_recent > 0.002 else "down"
        else:
            price_trend = "neutral"

        suggestion = self._generate_trading_suggestion(volatility, current_price, price_trend)

        chart_result = None
        if show_chart or MATPLOTLIB_AVAILABLE:
            chart_result = self.plot_volatility_chart(stock_code, data, show_chart)

        return {
            "stock_code": stock_code,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_price": current_price,
            "data_points": len(data),
            "volatility_analysis": {
                "daily_volatility": volatility.daily_volatility,
                "weekly_volatility": volatility.weekly_volatility,
                "avg_daily_change": volatility.avg_daily_change,
                "avg_weekly_change": volatility.avg_weekly_change,
                "max_daily_up": volatility.max_daily_up,
                "max_daily_down": volatility.max_daily_down,
                "max_weekly_up": volatility.max_weekly_up,
                "max_weekly_down": volatility.max_weekly_down,
                "risk_level": volatility.risk_level
            },
            "trading_suggestion": {
                "action": suggestion.action,
                "confidence": suggestion.confidence,
                "reason": suggestion.reason,
                "expected_return": suggestion.expected_return,
                "best_time": suggestion.best_time
            },
            "chart": chart_result
        }


@tool
def stock_analysis_tool(
    stock_code: str = Field(description="股票代码，如 AAPL、TSLA"),
    days: int = Field(default=30, description="分析天数，默认为30天"),
    show_chart: bool = Field(default=False, description="是否显示图表")
) -> str:
    """
    股票分析工具，用于分析股票的波动性并提供交易建议。

    功能包括：
    1. 计算日波动和周波动
    2. 生成波动性图表（日波动和周波动绘制在同一图中）
    3. 基于波动性大小提供买入/卖出/观望建议
    4. 给出最佳交易时间建议

    适用于需要了解股票风险和交易时机的场景。
    """
    skill = StockVisualizationSkill()
    result = skill.analyze_stock(stock_code, days, show_chart)
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_volatility_chart(
    stock_code: str = Field(description="股票代码"),
    days: int = Field(default=30, description="数据天数")
) -> str:
    """
    获取股票波动性图表，仅生成并保存图表文件。

    返回图表文件路径，图表包含：
    - 上部：价格走势图
    - 中部：日波动柱状图
    - 下部：周波动柱状图
    """
    skill = StockVisualizationSkill()
    result = skill.plot_volatility_chart(stock_code, show_chart=False)
    if result.get("success"):
        return f"图表已保存至: {result['chart_path']}"
    else:
        return f"生成图表失败: {result.get('error', '未知错误')}"


@tool
def get_trading_suggestion(
    stock_code: str = Field(description="股票代码"),
    days: int = Field(default=30, description="用于分析的历史数据天数")
) -> str:
    """
    获取股票交易建议，包括买入、卖出或观望建议。

    基于波动性分析返回：
    - 交易动作（买入/卖出/观望）
    - 置信度
    - 建议理由
    - 预期收益（如果有）
    - 最佳交易时间
    """
    skill = StockVisualizationSkill()
    result = skill.analyze_stock(stock_code, days, show_chart=False)

    suggestion = result["trading_suggestion"]
    volatility = result["volatility_analysis"]

    response = f"""
股票代码: {stock_code}
分析日期: {result['analysis_date']}
当前价格: {result['current_price']}元

【波动性分析】
- 日波动率: {volatility['daily_volatility']}%
- 周波动率: {volatility['weekly_volatility']}%
- 平均日涨跌: {volatility['avg_daily_change']}%
- 平均周涨跌: {volatility['avg_weekly_change']}%
- 最大日涨幅: {volatility['max_daily_up']}%
- 最大日跌幅: {volatility['max_daily_down']}%
- 最大周涨幅: {volatility['max_weekly_up']}%
- 最大周跌幅: {volatility['max_weekly_down']}%
- 风险等级: {volatility['risk_level']}

【交易建议】
- 操作: {suggestion['action']}
- 置信度: {suggestion['confidence']:.0%}
- 理由: {suggestion['reason']}
- 最佳时间: {suggestion['best_time']}
"""

    if suggestion.get('expected_return'):
        response += f"- 预期收益: {suggestion['expected_return']:.2f}%\n"

    return response.strip()


def create_stock_visualization_skill() -> List[StructuredTool]:
    """
    创建股票可视化skill的工具列表
    """
    return [
        stock_analysis_tool,
        get_volatility_chart,
        get_trading_suggestion
    ]


if __name__ == "__main__":
    result = stock_analysis_tool.invoke({
        "stock_code": "AAPL",
        "days": 30,
        "show_chart": False
    })
    print(result)

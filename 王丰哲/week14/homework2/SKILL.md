---
name: stock-volatility-visualizer
description: 用于股票波动率可视化和交易时机分析。Use when Codex needs to fetch stock daily/weekly K-line data, draw daily volatility and weekly volatility in one chart, compare volatility size, and produce conservative buy/sell timing suggestions.
---

# 股票波动可视化

## 主要能力

- 调用 AutoStock K 线接口获取指定股票的日 K 线和周 K 线。
- 计算日波动率和周波动率：`(最高价 - 最低价) / 收盘价 * 100%`。
- 将日波动和周波动绘制在同一张 SVG 图中，便于比较短线和中期波动。
- 根据波动大小给出买入观察窗口和卖出/减仓窗口建议。

## 使用流程

1. 确认股票代码，例如 `000001`。
2. 运行脚本生成图表和建议：

```bash
python3 scripts/stock_volatility_visualizer.py --code 000001 --output 000001_volatility.svg
```

3. 如需限制时间范围，传入开始和结束日期：

```bash
python3 scripts/stock_volatility_visualizer.py \
  --code 000001 \
  --start-date 2025-01-01 \
  --end-date 2026-05-15 \
  --output 000001_volatility.svg
```

4. 如需设置复权方式，使用 `--adjust-type`：

```bash
python3 scripts/stock_volatility_visualizer.py --code 000001 --adjust-type 1
```

`--adjust-type` 含义与参考文件中的 AutoStock 接口一致：

| 参数 | 含义 |
| --- | --- |
| `0` | 不复权 |
| `1` | 前复权 |
| `2` | 后复权 |

## 买卖建议规则

脚本使用波动率分位数做保守判断：

- **买入观察窗口**：优先选择近期日波动处于低位区间，且收盘价相对较低的日期。低波动通常说明价格进入相对稳定区，适合观察或分批买入。
- **卖出/减仓窗口**：优先选择近期日波动处于高位区间，且收盘价相对较高的日期。高波动通常说明情绪放大，适合止盈、减仓或等待回落。
- **当前状态**：对比最新日波动、最新周波动与近期分位数。如果最新波动明显放大，提示谨慎；如果日周波动同步收敛，提示可观察低吸；否则提示继续等待确认。

这些建议只基于历史 K 线波动，不构成投资建议；实际交易还需要结合趋势、成交量、基本面和风险承受能力。

## 脚本位置

- `scripts/stock_volatility_visualizer.py`：获取 K 线、计算波动、生成 SVG 图表并输出 JSON 建议。

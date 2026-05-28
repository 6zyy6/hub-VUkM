"""提示词模板"""
from typing import Dict, List, Optional


SYSTEM_PROMPT = """你是一个狼人杀游戏中的玩家。请根据你的身份和游戏局势做出最优决策。
游戏规则：
- 狼人在夜晚可以杀害一名玩家
- 预言家每晚可以查验一名玩家的身份
- 女巫有解药和毒药各一瓶
- 猎人死亡时可以带走一人
- 村民需要找出狼人
- 白天所有存活玩家轮流发言，然后投票
"""


def format_game_context(context: Dict) -> str:
    """格式化游戏上下文"""
    lines = []
    lines.append("当前游戏状态:")
    lines.append(f"- 存活玩家: {', '.join(context.get('living_players', []))}")
    lines.append(f"- 狼人团队: {', '.join(context.get('werewolf_teammates', []))}")

    if context.get("seer_checks"):
        items = []
        for k, v in context["seer_checks"].items():
            label = "狼人" if v else "好人"
            items.append(f"{k}: {label}")
        lines.append(f"- 查验结果: {', '.join(items)}")

    if context.get("wolf_kill_target"):
        lines.append(f"- 狼人今晚要杀: {context['wolf_kill_target']}")

    return "\n".join(lines)


def villager_prompt(player_name: str, game_context: Dict, my_identity: str) -> str:
    """村民提示词"""
    context_str = format_game_context(game_context)
    return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是村民（好人）。

{context_str}

作为村民，你需要分析局势，找出狼人。请发表你的看法和推理。
你的发言应该：
1. 分析其他玩家的言行
2. 指出可疑玩家
3. 不要暴露自己的村民身份（除非必要）

请直接输出你的发言内容："""


def werewolf_prompt(
    player_name: str,
    living_good: List[str],
    teammates: List[str],
    game_context: Dict,
    is_night: bool = True,
) -> str:
    """狼人提示词"""
    context_str = format_game_context(game_context)

    if is_night:
        good_list = "\n".join(f'- {name}' for name in living_good[:5])
        return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是狼人。

狼人队友: {', '.join(teammates)}
今晚要杀的目标: 从以下好人中选择一人
{good_list}

{context_str}

请选择今晚要杀的人，直接输出玩家名称即可。
"""
    else:
        return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是狼人。

{context_str}

作为狼人，你需要隐藏身份，引导舆论。请发表你的发言。
你的发言应该：
1. 像村民一样分析局势
2. 把嫌疑引向好人
3. 不要暴露自己是狼人

请直接输出你的发言内容："""


def seer_prompt(
    player_name: str,
    candidates: List[str],
    checked: Dict[str, bool],
    is_night: bool = True,
) -> str:
    """预言家提示词"""
    checked_str = ""
    if checked:
        items = []
        for k, v in checked.items():
            label = "狼人" if v else "好人"
            items.append(f"{k}: {label}")
        checked_str = f"已查验: {', '.join(items)}"

    if is_night:
        return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是预言家（好人）。

可选查验目标: {', '.join(candidates)}
已查验结果: {checked_str}

请选择今晚要查验的玩家，直接输出玩家名称即可。
"""
    else:
        return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是预言家（好人）。

已查验结果: {checked_str}

作为预言家，你可以选择是否公开查验结果。如果狼人已经明朗，建议公开。
如果选择公开，请说明查验了谁以及结果；如果不公开，请分析局势。

请直接输出你的发言内容："""


def witch_prompt(
    player_name: str,
    victim: Optional[str],
    heal_remaining: int,
    poison_remaining: int,
    is_day: bool = False,
) -> str:
    """女巫提示词"""
    if is_day:
        return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是女巫（好人）。

解药剩余: {heal_remaining} 瓶
毒药剩余: {poison_remaining} 瓶

请分析局势并发言。
"""

    action_parts = []
    if heal_remaining > 0 and victim:
        action_parts.append(f"救（使用解药救 {victim}）")
    if poison_remaining > 0:
        action_parts.append("毒（使用毒药毒死一名玩家）")
    action_parts.append("等待（不使用任何药）")

    action_list = "\n".join(f"- {a}" for a in action_parts)
    victim_str = victim if victim else "未知"

    return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是女巫（好人）。

今晚狼人杀害目标: {victim_str}
解药剩余: {heal_remaining} 瓶
毒药剩余: {poison_remaining} 瓶

你可以选择:
{action_list}

请直接输出你的选择："""


def hunter_prompt(
    player_name: str,
    game_context: Dict,
    can_shoot: bool,
    is_dying: bool = False,
) -> str:
    """猎人提示词"""
    context_str = format_game_context(game_context)

    if is_dying and can_shoot:
        living_str = ', '.join(game_context.get('living_players', []))
        return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是猎人（好人）。

你即将死亡，可以选择带走一名玩家。
存活玩家: {living_str}

{context_str}

请选择你要带走的人，直接输出玩家名称即可。
"""

    shoot_status = "可以开枪" if can_shoot else "已开枪"
    return f"""{SYSTEM_PROMPT}

你是 {player_name}，身份是猎人（好人）。

{context_str}

作为猎人，你需要分析局势并在适当时机发挥作用。
开枪状态: {shoot_status}

请发表你的发言："""

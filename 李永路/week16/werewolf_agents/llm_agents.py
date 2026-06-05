"""
基于大语言模型的 Agent 实现
每个角色拥有专属的提示词模板和推理逻辑
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from .models import Player, GameState, RoleType
from .llm_client import BaseLLMClient, create_llm_client


class PromptTemplates:
    """提示词模板库"""
    
    # 系统提示词 - 通用
    SYSTEM_PROMPT = """你是一个狼人杀游戏的 AI 玩家。你需要根据你扮演的角色、当前游戏局势和其他玩家的发言，做出最优决策。

游戏规则：
- 狼人阵营：夜间击杀所有好人
- 好人阵营：找出并投票淘汰所有狼人
- 夜间行动保密，白天公开讨论

请严格按照要求的格式输出你的决策。"""

    # 狼人角色提示词
    WEREWOLF_SYSTEM = """你是狼人阵营的一员。你的目标是消灭所有好人。

关键信息：
- 你知道所有狼人队友的身份
- 夜间需要与队友协商击杀目标
- 白天要伪装成好人，混淆视听

策略建议：
- 夜间优先击杀神职角色（预言家、女巫等）
- 白天发言要自然，不要暴露身份
- 可以假装分析局势，引导投票"""

    # 村民角色提示词
    VILLAGER_SYSTEM = """你是一个普通村民，没有任何特殊能力。

关键信息：
- 你不知道任何其他玩家的身份
- 需要通过分析发言和行为来找出狼人
- 你的投票对游戏结果至关重要

策略建议：
- 仔细倾听每个人的发言逻辑
- 寻找发言中的矛盾和不合理之处
- 不要轻易相信跳身份的Player"""

    # 预言家角色提示词
    SEER_SYSTEM = """你是预言家，每晚可以查验一名玩家的身份。

关键信息：
- 你知道已查验玩家的真实身份
- 可以选择何时公开你的身份和查验结果
- 你是好人阵营的核心信息来源

策略建议：
- 尽早跳身份报查验，建立信任
- 优先查验发言可疑的玩家
- 注意保护自己，避免被狼人夜间击杀"""

    # 女巫角色提示词
    WITCH_SYSTEM = """你是女巫，拥有一瓶解药和一瓶毒药。

关键信息：
- 解药可以救夜间被狼人击杀的玩家
- 毒药可以毒杀任意一名玩家
- 每种药只能使用一次

策略建议：
- 第一晚通常使用解药救人
- 毒药要谨慎使用，避免误杀好人
- 可以根据局势选择是否公开身份"""

    # 猎人角色提示词
    HUNTER_SYSTEM = """你是猎人，死亡时可以带走一名玩家。

关键信息：
- 你死亡时可以发动技能带走一人
- 这个技能对狼人有很大威慑力
- 你可以适当暗示自己的身份

策略建议：
- 可以适当强势发言，威慑狼人
- 死亡时优先带走确信的狼人
- 不要轻易暴露身份，避免被针对"""

    # 守卫角色提示词
    GUARD_SYSTEM = """你是守卫，每晚可以守护一名玩家免受夜间伤害。

关键信息：
- 不能连续两晚守护同一人
- 守护成功则该玩家免疫夜间伤害
- 守护和女巫解药同时使用会导致"奶穿"（玩家死亡）

策略建议：
- 优先守护疑似神职或重要玩家
- 可以守护自己保命
- 注意与女巫的配合，避免奶穿"""

    @classmethod
    def get_system_prompt(cls, role: RoleType) -> str:
        """获取角色对应的系统提示词"""
        prompts = {
            RoleType.WEREWOLF: cls.WEREWOLF_SYSTEM,
            RoleType.VILLAGER: cls.VILLAGER_SYSTEM,
            RoleType.SEER: cls.SEER_SYSTEM,
            RoleType.WITCH: cls.WITCH_SYSTEM,
            RoleType.HUNTER: cls.HUNTER_SYSTEM,
            RoleType.GUARD: cls.GUARD_SYSTEM,
        }
        return prompts.get(role, cls.SYSTEM_PROMPT)


class LLMAgent:
    """基于 LLM 的 Agent"""
    
    def __init__(self, player: Player, game_state: GameState, 
                 llm_client: Optional[BaseLLMClient] = None,
                 llm_provider: str = "mock"):
        self.player = player
        self.game_state = game_state
        self.role = player.role
        
        # 初始化 LLM 客户端
        if llm_client:
            self.llm = llm_client
        else:
            self.llm = create_llm_client(provider=llm_provider)
        
        # 获取系统提示词
        self.system_prompt = PromptTemplates.get_system_prompt(self.role)
        
        # 对话历史
        self.conversation_history: List[Dict] = []
    
    def _build_context(self) -> str:
        """构建当前游戏上下文信息"""
        context = f"【当前游戏状态】\n"
        context += f"- 回合数：第 {self.game_state.current_round} 晚 / 第 {self.game_state.day_count} 天\n"
        context += f"- 你的身份：{self.player.name}（{self.role.value}）\n"
        context += f"- 你的状态：{'存活' if self.player.is_alive() else '死亡'}\n\n"
        
        # 存活玩家列表
        alive_players = self.game_state.get_alive_players()
        context += f"【存活玩家】\n"
        for p in alive_players:
            context += f"- {p.player_id}: {p.name}\n"
        context += "\n"
        
        # 已知信息
        if self.player.known_roles:
            context += f"【你已知的信息】\n"
            for pid, role in self.player.known_roles.items():
                player_name = self.game_state.players[pid].name
                context += f"- {player_name} 是 {role.value}\n"
            context += "\n"
        
        # 最近的讨论记录
        if self.game_state.discussion_log:
            context += f"【最近的发言记录】\n"
            recent_logs = self.game_state.discussion_log[-5:]  # 最近5条
            for log in recent_logs:
                speaker_name = self.game_state.players[log.speaker_id].name
                context += f"- {speaker_name}: {log.content}\n"
            context += "\n"
        
        return context
    
    def night_action(self) -> Optional[int]:
        """
        夜间行动 - 使用 LLM 决策
        
        Returns:
            目标玩家 ID，None 表示不行动
        """
        if not self.player.is_alive():
            return None
        
        context = self._build_context()
        
        # 根据不同角色构建不同的提示词
        if self.role == RoleType.WEREWOLF:
            prompt = self._build_werewolf_night_prompt(context)
        elif self.role == RoleType.SEER:
            prompt = self._build_seer_night_prompt(context)
        elif self.role == RoleType.WITCH:
            prompt = self._build_witch_night_prompt(context)
        elif self.role == RoleType.GUARD:
            prompt = self._build_guard_night_prompt(context)
        else:
            return None  # 村民和猎人夜间无行动
        
        # 调用 LLM
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=100
        )
        
        # 解析响应
        target_id = self._parse_player_id(response)
        
        if target_id:
            self.player.night_actions.append(f"LLM决策: {response}")
        
        return target_id
    
    def witch_night_action(self) -> Tuple[Optional[int], Optional[int]]:
        """
        女巫夜间行动（特殊处理，返回两个动作）
        
        Returns:
            (save_target, poison_target)
        """
        if not self.player.is_alive():
            return None, None
        
        context = self._build_context()
        prompt = self._build_witch_night_prompt(context)
        
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=150
        )
        
        # 解析响应，期望格式：救:ID, 毒:ID 或 救:无, 毒:ID
        save_target = self._parse_action_from_response(response, "救")
        poison_target = self._parse_action_from_response(response, "毒")
        
        return save_target, poison_target
    
    def day_speech(self) -> str:
        """
        白天发言 - 使用 LLM 生成
        
        Returns:
            发言内容
        """
        if not self.player.is_alive():
            return ""
        
        context = self._build_context()
        
        prompt = f"""{context}

【任务】
请为你的角色生成一段白天发言内容。

【要求】
1. 发言长度在 50-150 字之间
2. 符合你的角色身份和策略
3. 可以分析局势、表达观点、质疑他人
4. 不要直接暴露敏感信息（如狼人要伪装）
5. 语言自然流畅，像真人一样发言

【输出格式】
直接输出发言内容，不要包含其他说明。"""

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.8,
            max_tokens=200
        )
        
        # 清理响应
        speech = response.strip().strip('"').strip("'")
        
        # 限制长度
        if len(speech) > 200:
            speech = speech[:200] + "..."
        
        self.player.day_actions.append(f"LLM发言: {speech}")
        
        return speech
    
    def vote_target(self) -> Optional[int]:
        """
        投票目标 - 使用 LLM 决策
        
        Returns:
            投票给的玩家 ID，None 表示弃票
        """
        if not self.player.is_alive():
            return None
        
        context = self._build_context()
        
        prompt = f"""{context}

【任务】
请决定你要投票淘汰的玩家。

【可用选项】
"""
        alive_others = [p for p in self.game_state.get_alive_players() 
                       if p.player_id != self.player.player_id]
        
        for p in alive_others:
            prompt += f"- {p.player_id}: {p.name}\n"
        
        prompt += """
- 0: 弃票

【要求】
1. 根据你的角色身份和已知信息做出决策
2. 考虑之前的发言和局势
3. 只输出玩家编号数字

【输出格式】
只输出一个数字（玩家ID或0表示弃票）"""

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.5,
            max_tokens=50
        )
        
        # 解析响应
        target_id = self._parse_player_id(response)
        
        # 确保目标是存活的且不是自己
        if target_id and target_id != 0:
            if target_id not in [p.player_id for p in alive_others]:
                target_id = None
        
        return target_id
    
    def hunter_on_death(self) -> Optional[int]:
        """
        猎人死亡时的技能
        
        Returns:
            带走的玩家 ID
        """
        context = self._build_context()
        
        prompt = f"""{context}

【任务】
你作为猎人已死亡，请选择要带走的玩家。

【要求】
1. 优先带走确信的狼人
2. 如果不确定，可以选择弃票
3. 只输出玩家编号数字

【输出格式】
只输出一个数字（玩家ID或0表示不带走任何人）"""

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.5,
            max_tokens=50
        )
        
        return self._parse_player_id(response)
    
    # ========== 辅助方法 ==========
    
    def _build_werewolf_night_prompt(self, context: str) -> str:
        """构建狼人夜间提示词"""
        prompt = f"""{context}

【任务】
作为狼人，请选择今晚要击杀的目标。

【策略】
- 优先击杀神职角色（预言家、女巫等）
- 避免击杀狼人队友
- 考虑白天的发言表现

【输出格式】
只输出要击杀的玩家编号数字"""
        
        return prompt
    
    def _build_seer_night_prompt(self, context: str) -> str:
        """构建预言家夜间提示词"""
        prompt = f"""{context}

【任务】
作为预言家，请选择今晚要查验的玩家。

【策略】
- 优先查验发言可疑的玩家
- 避免重复查验已查验的玩家
- 可以考虑查验沉默寡言的玩家

【输出格式】
只输出要查验的玩家编号数字"""
        
        return prompt
    
    def _build_witch_night_prompt(self, context: str) -> str:
        """构建女巫夜间提示词"""
        prompt = f"""{context}

【任务】
作为女巫，请决定是否使用解药或毒药。

【当前状态】
- 解药剩余：{'有' if not self.game_state.witch_antidote_used else '已用完'}
- 毒药剩余：{'有' if not self.game_state.witch_poison_used else '已用完'}

【输出格式】
请按以下格式输出：
救:玩家ID 或 救:无
毒:玩家ID 或 毒:无

示例：
救:3
毒:无"""
        
        return prompt
    
    def _build_guard_night_prompt(self, context: str) -> str:
        """构建守卫夜间提示词"""
        last_protected = self.game_state.guard_last_protected
        protected_info = f"\n注意：昨晚你守护了玩家 {last_protected}，今晚不能再次守护他。" if last_protected else ""
        
        prompt = f"""{context}{protected_info}

【任务】
作为守卫，请选择今晚要守护的玩家。

【策略】
- 可以守护自己保命
- 优先守护疑似神职的玩家
- 不能连续两晚守护同一人

【输出格式】
只输出要守护的玩家编号数字"""
        
        return prompt
    
    def _parse_player_id(self, text: str) -> Optional[int]:
        """从文本中解析玩家 ID"""
        # 尝试提取数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            try:
                return int(numbers[0])
            except:
                pass
        return None
    
    def _parse_action_from_response(self, text: str, action_type: str) -> Optional[int]:
        """从响应中解析特定动作的目标"""
        # 查找类似 "救:3" 或 "毒:无" 的模式
        pattern = rf"{action_type}[:：]\s*(\d+|无)"
        match = re.search(pattern, text)
        
        if match:
            value = match.group(1)
            if value == "无":
                return None
            try:
                return int(value)
            except:
                pass
        
        return None

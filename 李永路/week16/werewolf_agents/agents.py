"""
Agent 基类和角色特定 Agent 实现
每个 Agent 根据其扮演角色拥有独立的目标、策略与行动空间
"""

import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from .models import Player, GameState, RoleType, NightAction, DayAction


class BaseAgent(ABC):
    """Agent 基类"""
    
    def __init__(self, player: Player, game_state: GameState):
        self.player = player
        self.game_state = game_state
        self.role = player.role
    
    @abstractmethod
    def night_action(self) -> Optional[int]:
        """
        夜间行动
        Returns: 目标玩家 ID，None 表示不行动
        """
        pass
    
    @abstractmethod
    def day_speech(self) -> str:
        """
        白天发言
        Returns: 发言内容
        """
        pass
    
    @abstractmethod
    def vote_target(self) -> Optional[int]:
        """
        投票目标
        Returns: 投票给的玩家 ID，None 表示弃票
        """
        pass
    
    def get_alive_players_except_self(self) -> List[Player]:
        """获取除自己外的存活玩家"""
        return [p for p in self.game_state.get_alive_players() 
                if p.player_id != self.player.player_id]
    
    def get_info_summary(self) -> str:
        """获取当前已知信息摘要"""
        summary = f"我是 {self.player.name}（{self.role.value}）\n"
        summary += f"已知信息：\n"
        
        if self.player.known_roles:
            for pid, role in self.player.known_roles.items():
                player_name = self.game_state.players[pid].name
                summary += f"- {player_name} 是 {role.value}\n"
        
        alive = self.game_state.get_alive_players()
        summary += f"\n存活玩家：{', '.join([p.name for p in alive])}\n"
        
        return summary


class WerewolfAgent(BaseAgent):
    """狼人 Agent - 目标是消灭所有好人"""
    
    def __init__(self, player: Player, game_state: GameState, werewolf_team: List[int]):
        super().__init__(player, game_state)
        self.werewolf_team = werewolf_team  # 狼人队友 ID 列表
        
        # 狼人知道所有队友的身份
        for wid in werewolf_team:
            if wid != player.player_id:
                player.add_known_role(wid, RoleType.WEREWOLF)
    
    def night_action(self) -> Optional[int]:
        """狼人夜间选择击杀目标"""
        alive_villagers = [p for p in self.game_state.get_alive_players() 
                          if p.player_id not in self.werewolf_team]
        
        if not alive_villagers:
            return None
        
        # 策略：优先击杀已知的神职，否则随机
        # 这里简化为随机选择
        target = random.choice(alive_villagers)
        self.player.night_actions.append(f"击杀 {target.name}")
        
        return target.player_id
    
    def day_speech(self) -> str:
        """狼人白天发言 - 伪装成好人"""
        strategies = [
            f"我觉得昨晚的情况很奇怪，需要大家仔细分析。",
            f"我是一个普通村民，希望预言家能出来报一下查验信息。",
            f"我怀疑某个玩家的发言有问题，但还需要更多信息。",
            f"建议大家多听听每个人的发言逻辑，找出矛盾点。",
        ]
        
        speech = random.choice(strategies)
        self.player.day_actions.append(f"发言: {speech}")
        
        return speech
    
    def vote_target(self) -> Optional[int]:
        """狼人投票 - 尽量投好人"""
        alive_villagers = [p for p in self.game_state.get_alive_players() 
                          if p.player_id not in self.werewolf_team]
        
        if not alive_villagers:
            return None
        
        # 策略：集中投票给同一个好人
        target = random.choice(alive_villagers)
        return target.player_id


class VillagerAgent(BaseAgent):
    """村民 Agent - 没有特殊能力，通过推理找出狼人"""
    
    def night_action(self) -> Optional[int]:
        """村民夜间无行动"""
        return None
    
    def day_speech(self) -> str:
        """村民白天发言 - 分析局势"""
        suspicious = []
        for action in self.game_state.discussion_log:
            # 简单分析：记录可疑发言
            if "怀疑" in action.content or "狼" in action.content:
                suspicious.append(action.speaker_id)
        
        if suspicious:
            speech = f"我注意到有些玩家的发言比较可疑，我们需要更多讨论来确认身份。"
        else:
            speech = f"我是一名普通村民，目前信息还不足，想听听大家的分析。"
        
        self.player.day_actions.append(f"发言: {speech}")
        return speech
    
    def vote_target(self) -> Optional[int]:
        """村民投票 - 基于讨论投票"""
        # 简化策略：随机投票给非自己的玩家
        others = self.get_alive_players_except_self()
        if others:
            return random.choice(others).player_id
        return None


class SeerAgent(BaseAgent):
    """预言家 Agent - 每晚可以查验一名玩家的身份"""
    
    def __init__(self, player: Player, game_state: GameState):
        super().__init__(player, game_state)
        self.verified_players: List[int] = []  # 已查验的玩家
    
    def night_action(self) -> Optional[int]:
        """预言家夜间查验"""
        # 选择未查验的存活玩家
        unverified = [p for p in self.get_alive_players_except_self() 
                     if p.player_id not in self.verified_players]
        
        if not unverified:
            return None
        
        # 策略：优先查验发言可疑的玩家，否则随机
        target = random.choice(unverified)
        self.verified_players.append(target.player_id)
        
        # 查验结果
        is_werewolf = target.role == RoleType.WEREWOLF
        result = "狼人" if is_werewolf else "好人"
        self.player.add_known_role(target.player_id, target.role)
        self.player.night_actions.append(f"查验 {target.name}，结果是 {result}")
        
        return target.player_id
    
    def day_speech(self) -> str:
        """预言家白天发言 - 可以选择跳身份或不跳"""
        if self.verified_players:
            last_verified = self.verified_players[-1]
            player = self.game_state.players[last_verified]
            role = "狼人" if player.role == RoleType.WEREWOLF else "好人"
            
            # 策略：50% 概率跳身份报查验
            if random.random() > 0.5:
                speech = f"我是预言家，昨晚查验了 {player.name}，他是 {role}。"
            else:
                speech = f"我有一些信息，但现在还不是透露的时候。"
        else:
            speech = f"我还在学习局势，希望大家多分享信息。"
        
        self.player.day_actions.append(f"发言: {speech}")
        return speech
    
    def vote_target(self) -> Optional[int]:
        """预言家投票 - 优先投已知的狼人"""
        known_werewolves = [pid for pid, role in self.player.known_roles.items() 
                           if role == RoleType.WEREWOLF 
                           and self.game_state.players[pid].is_alive()]
        
        if known_werewolves:
            return known_werewolves[0]
        
        # 否则投可疑玩家
        others = self.get_alive_players_except_self()
        if others:
            return random.choice(others).player_id
        return None


class WitchAgent(BaseAgent):
    """女巫 Agent - 有一瓶解药和一瓶毒药"""
    
    def __init__(self, player: Player, game_state: GameState):
        super().__init__(player, game_state)
        self.antidote_used = False
        self.poison_used = False
        self.night_kill_target: Optional[int] = None  # 昨晚被杀的人
    
    def set_night_kill_target(self, target_id: Optional[int]):
        """设置昨晚被狼人击杀的目标"""
        self.night_kill_target = target_id
    
    def night_action(self) -> Tuple[Optional[int], Optional[int]]:
        """
        女巫夜间行动
        Returns: (save_target, poison_target)
        """
        save_target = None
        poison_target = None
        
        # 使用解药救人
        if (self.night_kill_target is not None and 
            not self.antidote_used and 
            random.random() > 0.3):  # 70% 概率救人
            save_target = self.night_kill_target
            self.antidote_used = True
            self.game_state.witch_antidote_used = True
            self.player.night_actions.append(f"使用解药救了玩家")
        
        # 使用毒药毒人
        if not self.poison_used and random.random() > 0.7:  # 30% 概率毒人
            alive_others = self.get_alive_players_except_self()
            # 排除刚救的人
            if save_target:
                alive_others = [p for p in alive_others if p.player_id != save_target]
            
            if alive_others:
                target = random.choice(alive_others)
                poison_target = target.player_id
                self.poison_used = True
                self.game_state.witch_poison_used = True
                self.player.night_actions.append(f"使用毒药毒了 {target.name}")
        
        return save_target, poison_target
    
    def day_speech(self) -> str:
        """女巫白天发言"""
        if self.antidote_used:
            speech = f"我已经用过解药了，现在只剩毒药，会谨慎使用。"
        elif self.poison_used:
            speech = f"我已经用过毒药了，希望大家好好讨论。"
        else:
            speech = f"我手里还有药，会根据局势决定是否使用。"
        
        self.player.day_actions.append(f"发言: {speech}")
        return speech
    
    def vote_target(self) -> Optional[int]:
        """女巫投票"""
        others = self.get_alive_players_except_self()
        if others:
            return random.choice(others).player_id
        return None


class HunterAgent(BaseAgent):
    """猎人 Agent - 死亡时可以带走一名玩家"""
    
    def night_action(self) -> Optional[int]:
        """猎人夜间无行动"""
        return None
    
    def on_death(self) -> Optional[int]:
        """
        猎人死亡时的技能
        Returns: 带走的玩家 ID
        """
        others = self.get_alive_players_except_self()
        if others:
            target = random.choice(others)
            self.player.day_actions.append(f"发动技能带走 {target.name}")
            return target.player_id
        return None
    
    def day_speech(self) -> str:
        """猎人白天发言"""
        speech = f"我不是狼人，希望大家理性分析，不要轻易投票。"
        self.player.day_actions.append(f"发言: {speech}")
        return speech
    
    def vote_target(self) -> Optional[int]:
        """猎人投票"""
        others = self.get_alive_players_except_self()
        if others:
            return random.choice(others).player_id
        return None


class GuardAgent(BaseAgent):
    """守卫 Agent - 每晚可以守护一名玩家（不能连续两晚守护同一人）"""
    
    def night_action(self) -> Optional[int]:
        """守卫夜间守护"""
        alive_players = self.get_alive_players_except_self()
        
        # 不能连续两晚守护同一人
        if self.game_state.guard_last_protected:
            alive_players = [p for p in alive_players 
                           if p.player_id != self.game_state.guard_last_protected]
        
        if not alive_players:
            return None
        
        # 策略：随机守护或守护自己
        if random.random() > 0.5:
            target = self.player  # 守护自己
        else:
            target = random.choice(alive_players)
        
        self.game_state.guard_last_protected = target.player_id
        self.player.night_actions.append(f"守护 {target.name}")
        
        return target.player_id
    
    def day_speech(self) -> str:
        """守卫白天发言"""
        speech = f"昨晚平安夜或者有人死亡，我需要更多信息来判断局势。"
        self.player.day_actions.append(f"发言: {speech}")
        return speech
    
    def vote_target(self) -> Optional[int]:
        """守卫投票"""
        others = self.get_alive_players_except_self()
        if others:
            return random.choice(others).player_id
        return None


def create_agent(player: Player, game_state: GameState, werewolf_teams: Dict[int, List[int]]) -> BaseAgent:
    """工厂函数：根据角色创建对应的 Agent"""
    
    if player.role == RoleType.WEREWOLF:
        werewolf_team = werewolf_teams.get(player.player_id, [])
        return WerewolfAgent(player, game_state, werewolf_team)
    
    elif player.role == RoleType.VILLAGER:
        return VillagerAgent(player, game_state)
    
    elif player.role == RoleType.SEER:
        return SeerAgent(player, game_state)
    
    elif player.role == RoleType.WITCH:
        return WitchAgent(player, game_state)
    
    elif player.role == RoleType.HUNTER:
        return HunterAgent(player, game_state)
    
    elif player.role == RoleType.GUARD:
        return GuardAgent(player, game_state)
    
    else:
        raise ValueError(f"Unsupported role: {player.role}")

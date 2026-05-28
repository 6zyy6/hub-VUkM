"""
狼人杀对局引擎
负责回合流转、胜负裁决、信息隔离等核心逻辑
"""

import uuid
import random
from typing import List, Dict, Optional
from collections import Counter
from .models import (
    Player, GameState, RoleType, PlayerStatus, 
    GamePhase, NightAction, DayAction
)
from .agents import BaseAgent, create_agent
from .llm_agents import LLMAgent
from .llm_client import BaseLLMClient


class WerewolfGameEngine:
    """狼人杀游戏引擎"""
    
    def __init__(self, player_configs: List[Dict], use_llm: bool = False, 
                 llm_provider: str = "mock", llm_client: Optional[BaseLLMClient] = None):
        """
        初始化游戏引擎
        
        Args:
            player_configs: 玩家配置列表，每个元素包含 name, role, is_ai
            use_llm: 是否使用 LLM Agent
            llm_provider: LLM 提供商 ("openai", "qwen", "deepseek", "mock")
            llm_client: 自定义 LLM 客户端（可选）
        """
        self.game_id = str(uuid.uuid4())[:8]
        self.game_state = GameState(game_id=self.game_id)
        self.agents: Dict[int, BaseAgent] = {}
        self.werewolf_teams: Dict[int, List[int]] = {}  # 狼人团队映射
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_client = llm_client
        
        # 初始化玩家
        self._initialize_players(player_configs)
        
        # 创建 Agent
        self._create_agents()
        
        # 初始化狼人团队信息
        self._setup_werewolf_teams()
    
    def _initialize_players(self, player_configs: List[Dict]):
        """初始化所有玩家"""
        for idx, config in enumerate(player_configs):
            player = Player(
                player_id=idx + 1,
                name=config.get("name", f"玩家{idx + 1}"),
                role=RoleType(config.get("role", "villager")),
                is_ai=config.get("is_ai", True)
            )
            self.game_state.players[player.player_id] = player
    
    def _create_agents(self):
        """为所有玩家创建对应的 Agent"""
        for player in self.game_state.players.values():
            if self.use_llm:
                # 使用 LLM Agent
                agent = LLMAgent(
                    player, 
                    self.game_state,
                    llm_client=self.llm_client,
                    llm_provider=self.llm_provider
                )
            else:
                # 使用规则 Agent
                agent = create_agent(player, self.game_state, self.werewolf_teams)
            
            self.agents[player.player_id] = agent
    
    def _setup_werewolf_teams(self):
        """设置狼人团队信息（狼人互相知道身份）"""
        werewolf_ids = [
            p.player_id for p in self.game_state.players.values() 
            if p.role == RoleType.WEREWOLF
        ]
        
        for wid in werewolf_ids:
            self.werewolf_teams[wid] = werewolf_ids
    
    def start_game(self):
        """开始游戏"""
        self.game_state.add_log(f"===== 游戏 {self.game_id} 开始 =====")
        self.game_state.add_log(f"共有 {len(self.game_state.players)} 名玩家参与")
        
        # 显示角色分配（仅供调试，实际游戏中应保密）
        role_summary = []
        for player in self.game_state.players.values():
            role_summary.append(f"{player.name}: {player.role.value}")
        
        self.game_state.add_log(f"角色分配: {', '.join(role_summary)}")
        self.game_state.add_log("=" * 50)
    
    def run_night_phase(self):
        """执行夜间阶段"""
        self.game_state.current_phase = GamePhase.NIGHT
        self.game_state.current_round += 1
        self.game_state.night_actions.clear()
        
        self.game_state.add_log(f"\n--- 第 {self.game_state.current_round} 晚 ---")
        
        # 重置守卫保护状态
        protected_player = None
        witch_save_target = None
        witch_poison_target = None
        werewolf_kill_target = None
        
        # 1. 守卫行动
        for player_id, agent in self.agents.items():
            if isinstance(agent, type(self.agents[player_id])):
                from .agents import GuardAgent
                if isinstance(agent, GuardAgent) and agent.player.is_alive():
                    protect_target = agent.night_action()
                    if protect_target:
                        protected_player = protect_target
                        self.game_state.add_log(f"[守卫] 守护了玩家 {self.game_state.players[protect_target].name}")
        
        # 2. 狼人行动
        werewolf_kills = []
        for player_id, agent in self.agents.items():
            from .agents import WerewolfAgent
            if isinstance(agent, WerewolfAgent) and agent.player.is_alive():
                kill_target = agent.night_action()
                if kill_target and kill_target not in werewolf_kills:
                    werewolf_kills.append(kill_target)
        
        if werewolf_kills:
            # 狼人投票决定击杀目标（简化：选择第一个目标）
            werewolf_kill_target = werewolf_kills[0]
            target_name = self.game_state.players[werewolf_kill_target].name
            self.game_state.add_log(f"[狼人] 选择击杀 {target_name}")
        
        # 3. 女巫行动
        for player_id, agent in self.agents.items():
            from .agents import WitchAgent
            if isinstance(agent, WitchAgent) and agent.player.is_alive():
                # 告诉女巫昨晚谁被杀了
                agent.set_night_kill_target(werewolf_kill_target)
                save_target, poison_target = agent.night_action()
                
                if save_target:
                    witch_save_target = save_target
                    self.game_state.add_log(f"[女巫] 使用了解药")
                
                if poison_target:
                    witch_poison_target = poison_target
                    target_name = self.game_state.players[poison_target].name
                    self.game_state.add_log(f"[女巫] 使用了毒药毒杀 {target_name}")
            
            # 处理 LLM Agent 的女巫
            elif isinstance(agent, LLMAgent) and agent.role == RoleType.WITCH and agent.player.is_alive():
                # 告诉女巫昨晚谁被杀了
                agent.game_state.night_actions.append(f"昨晚 {self.game_state.players[werewolf_kill_target].name if werewolf_kill_target else '无人'} 被击杀")
                save_target, poison_target = agent.witch_night_action()
                
                if save_target:
                    witch_save_target = save_target
                    self.game_state.witch_antidote_used = True
                    self.game_state.add_log(f"[女巫] 使用了解药")
                
                if poison_target:
                    witch_poison_target = poison_target
                    self.game_state.witch_poison_used = True
                    target_name = self.game_state.players[poison_target].name
                    self.game_state.add_log(f"[女巫] 使用了毒药毒杀 {target_name}")
        
        # 4. 预言家行动
        for player_id, agent in self.agents.items():
            from .agents import SeerAgent
            if isinstance(agent, SeerAgent) and agent.player.is_alive():
                verify_target = agent.night_action()
                if verify_target:
                    target_name = self.game_state.players[verify_target].name
                    self.game_state.add_log(f"[预言家] 查验了 {target_name}")
        
        # 结算夜间死亡
        deaths = []
        
        # 处理狼人击杀
        if werewolf_kill_target:
            if werewolf_kill_target == witch_save_target:
                self.game_state.add_log(f"[结果] {self.game_state.players[werewolf_kill_target].name} 被救，平安夜！")
            elif werewolf_kill_target == protected_player:
                self.game_state.add_log(f"[结果] {self.game_state.players[werewolf_kill_target].name} 被守护，平安夜！")
            else:
                player = self.game_state.players[werewolf_kill_target]
                player.die()
                deaths.append(player)
                self.game_state.add_log(f"[结果] {player.name} 被狼人杀害！")
        
        # 处理女巫毒杀
        if witch_poison_target and witch_poison_target != witch_save_target:
            player = self.game_state.players[witch_poison_target]
            if player.is_alive():
                player.die()
                deaths.append(player)
                self.game_state.add_log(f"[结果] {player.name} 被女巫毒杀！")
        
        # 猎人死亡触发技能
        for dead_player in deaths:
            if dead_player.role == RoleType.HUNTER:
                from .agents import HunterAgent
                agent = self.agents[dead_player.player_id]
                if isinstance(agent, HunterAgent):
                    take_away = agent.on_death()
                    if take_away:
                        taken_player = self.game_state.players[take_away]
                        taken_player.die()
                        self.game_state.add_log(f"[猎人] {dead_player.name} 发动技能带走了 {taken_player.name}！")
    
    def run_day_phase(self):
        """执行白天阶段"""
        self.game_state.current_phase = GamePhase.DAY
        self.game_state.day_count += 1
        
        self.game_state.add_log(f"\n--- 第 {self.game_state.day_count} 天 ---")
        
        # 公布昨晚死亡情况
        if not self.game_state.night_actions:
            self.game_state.add_log("[天亮] 昨晚是平安夜，没有人死亡。")
        else:
            alive_count = len(self.game_state.get_alive_players())
            self.game_state.add_log(f"[天亮] 天亮了，当前存活 {alive_count} 人。")
        
        # 讨论阶段
        self.run_discussion_phase()
        
        # 投票阶段
        self.run_voting_phase()
    
    def run_discussion_phase(self):
        """执行讨论阶段"""
        self.game_state.current_phase = GamePhase.DISCUSSION
        self.game_state.discussion_log.clear()
        
        self.game_state.add_log("\n[讨论] 开始发言环节...")
        
        # 按顺序让每个存活玩家发言
        alive_players = self.game_state.get_alive_players()
        random.shuffle(alive_players)  # 随机发言顺序
        
        for player in alive_players:
            agent = self.agents[player.player_id]
            speech = agent.day_speech()
            
            day_action = DayAction(
                speaker_id=player.player_id,
                content=speech,
                action_type="speech"
            )
            self.game_state.discussion_log.append(day_action)
            
            self.game_state.add_log(f"[{player.name}] {speech}")
    
    def run_voting_phase(self):
        """执行投票阶段"""
        self.game_state.current_phase = GamePhase.VOTING
        self.game_state.votes.clear()
        
        self.game_state.add_log("\n[投票] 开始投票环节...")
        
        # 每个存活玩家投票
        alive_players = self.game_state.get_alive_players()
        
        for player in alive_players:
            agent = self.agents[player.player_id]
            vote_target = agent.vote_target()
            
            if vote_target:
                self.game_state.votes[player.player_id] = vote_target
                target_name = self.game_state.players[vote_target].name
                self.game_state.add_log(f"[{player.name}] 投票给 {target_name}")
            else:
                self.game_state.add_log(f"[{player.name}] 弃票")
        
        # 统计票数
        if self.game_state.votes:
            vote_counts = Counter(self.game_state.votes.values())
            most_voted = vote_counts.most_common(1)[0]
            eliminated_id = most_voted[0]
            vote_count = most_voted[1]
            
            eliminated_player = self.game_state.players[eliminated_id]
            eliminated_player.die()
            
            self.game_state.add_log(f"\n[结果] {eliminated_player.name} 以 {vote_count} 票被淘汰！")
            self.game_state.add_log(f"[身份] {eliminated_player.name} 的身份是：{eliminated_player.role.value}")
            
            # 猎人被投票出局触发技能
            if eliminated_player.role == RoleType.HUNTER:
                from .agents import HunterAgent
                agent = self.agents[eliminated_player.player_id]
                if isinstance(agent, HunterAgent):
                    take_away = agent.on_death()
                    if take_away:
                        taken_player = self.game_state.players[take_away]
                        taken_player.die()
                        self.game_state.add_log(f"[猎人] {eliminated_player.name} 发动技能带走了 {taken_player.name}！")
        else:
            self.game_state.add_log("\n[结果] 无人得票，平票！")
    
    def check_and_end_game(self) -> bool:
        """检查游戏是否结束"""
        if self.game_state.check_game_over():
            self.game_state.current_phase = GamePhase.GAME_OVER
            self.game_state.add_log(f"\n{'='*50}")
            self.game_state.add_log(f"游戏结束！获胜方：{self.game_state.winner}")
            self.game_state.add_log(f"{'='*50}")
            return True
        return False
    
    def play_one_round(self):
        """执行一个完整的游戏回合（夜晚 + 白天）"""
        # 夜间阶段
        self.run_night_phase()
        
        # 检查游戏是否结束
        if self.check_and_end_game():
            return
        
        # 白天阶段
        self.run_day_phase()
        
        # 检查游戏是否结束
        self.check_and_end_game()
    
    def run_game(self, max_rounds: int = 10):
        """
        运行完整游戏
        
        Args:
            max_rounds: 最大回合数限制
        """
        self.start_game()
        
        for round_num in range(max_rounds):
            if self.game_state.current_phase == GamePhase.GAME_OVER:
                break
            
            self.play_one_round()
        
        # 如果达到最大回合数仍未结束，强制结束
        if self.game_state.current_phase != GamePhase.GAME_OVER:
            self.game_state.add_log("\n达到最大回合数限制，游戏强制结束。")
            # 根据存活人数判定胜负
            alive_werewolves = len(self.game_state.get_alive_werewolves())
            alive_villagers = len(self.game_state.get_alive_villagers())
            
            if alive_werewolves >= alive_villagers:
                self.game_state.winner = "werewolf"
            else:
                self.game_state.winner = "villager"
            
            self.game_state.current_phase = GamePhase.GAME_OVER
            self.game_state.add_log(f"最终获胜方：{self.game_state.winner}")
    
    def get_game_report(self) -> Dict:
        """生成游戏报告"""
        report = {
            "game_id": self.game_id,
            "winner": self.game_state.winner,
            "total_rounds": self.game_state.current_round,
            "players": [],
            "game_log": self.game_state.game_log
        }
        
        for player in self.game_state.players.values():
            player_report = {
                "player_id": player.player_id,
                "name": player.name,
                "role": player.role.value,
                "status": player.status.value,
                "known_roles": {
                    str(pid): role.value 
                    for pid, role in player.known_roles.items()
                }
            }
            report["players"].append(player_report)
        
        return report

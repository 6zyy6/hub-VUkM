"""
Agent Team 管理器
负责创建、管理、协调所有智能体，实现多智能体协作与博弈
"""

import asyncio
import json
import random
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

try:
    from .game_engine import WerewolfGameEngine, Role, Team, Phase, GameEvent, GameLog
    from .agents import AgentFactory, BaseAgent, AgentMessage, MessageType
except ImportError:
    from game_engine import WerewolfGameEngine, Role, Team, Phase, GameEvent, GameLog
    from agents import AgentFactory, BaseAgent, AgentMessage, MessageType


class AgentManager:
    """智能体管理器：连接游戏引擎与智能体"""

    def __init__(self, game_engine: WerewolfGameEngine, 
                 log_callback: Optional[Callable[[dict], None]] = None,
                 event_callback: Optional[Callable[[dict], None]] = None):
        self.game = game_engine
        self.agents: Dict[int, BaseAgent] = {}
        self.log_callback = log_callback
        self.event_callback = event_callback
        self.message_history: List[dict] = []
        self.agent_actions: Dict[int, List[dict]] = {}
        self.round_stats: List[dict] = []

        # 初始化所有Agent
        self._init_agents()

    def _init_agents(self):
        """根据游戏引擎的玩家状态初始化智能体"""
        for player_id, player_state in self.game.players.items():
            agent = AgentFactory.create_agent(
                player_id=player_id,
                role=player_state.role,
                name=player_state.name
            )
            agent.send_message_callback = self._broadcast_message
            self.agents[player_id] = agent
            self.agent_actions[player_id] = []

        # 设置狼人队友
        wolves = [pid for pid, agent in self.agents.items() if agent.role == Role.WEREWOLF]
        for pid in wolves:
            if hasattr(self.agents[pid], 'set_wolf_partners'):
                self.agents[pid].set_wolf_partners([w for w in wolves if w != pid])

    def _broadcast_message(self, message: AgentMessage):
        """广播消息给所有Agent"""
        self.message_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": message.to_dict()
        })

        # 根据消息类型和目标分发
        if not message.target_ids:  # 广播
            for agent in self.agents.values():
                if agent.alive and agent.player_id != message.sender_id:
                    agent.receive_message(message)
        else:  # 私聊
            for target_id in message.target_ids:
                if target_id in self.agents and self.agents[target_id].alive:
                    self.agents[target_id].receive_message(message)

        # 触发事件回调
        if self.event_callback:
            self.event_callback({
                "type": "agent_message",
                "message": message.to_dict()
            })

    def get_game_state_for_agent(self, player_id: int) -> dict:
        """获取指定玩家视角的游戏状态"""
        return self.game.public_state_dict(player_id)

    def run_night_phase(self) -> dict:
        """执行夜晚阶段：收集所有Agent的夜间行动"""
        self.game.run_night_phase()
        night_actions = {}

        # 狼人行动
        wolves = [pid for pid, agent in self.agents.items() 
                  if agent.role == Role.WEREWOLF and agent.alive]
        if wolves:
            # 狼人协调：选一个狼人作为代表
            lead_wolf = wolves[0]
            wolf_agent = self.agents[lead_wolf]
            # 获取狼人统一目标
            game_state = self.get_game_state_for_agent(lead_wolf)
            if hasattr(wolf_agent, 'coordinate_with_wolves'):
                target = wolf_agent.coordinate_with_wolves(game_state)
                if target:
                    self.game.set_werewolf_kill(target)
                    night_actions["werewolf_kill"] = target
                    self._log_agent_action(lead_wolf, "werewolf_kill", target)

        # 预言家行动
        seer = next((pid for pid, agent in self.agents.items() 
                     if agent.role == Role.SEER and agent.alive), None)
        if seer:
            seer_agent = self.agents[seer]
            game_state = self.get_game_state_for_agent(seer)
            action = seer_agent.decide_action(game_state, Phase.NIGHT, [])
            if action.get("action") == "check" and "target_id" in action:
                target = action["target_id"]
                is_wolf = self.game.set_seer_check(target)
                if is_wolf is not None:
                    seer_agent.record_check(target, is_wolf)
                    night_actions["seer_check"] = {"target": target, "is_wolf": is_wolf}
                    self._log_agent_action(seer, "seer_check", target, is_wolf)

        # 女巫行动（需要知道狼人刀谁）
        witch = next((pid for pid, agent in self.agents.items() 
                      if agent.role == Role.WITCH and agent.alive), None)
        if witch:
            witch_agent = self.agents[witch]
            game_state = self.get_game_state_for_agent(witch)
            # 女巫需要知道狼人刀谁
            killed = night_actions.get("werewolf_kill")
            action = witch_agent.decide_action(game_state, Phase.NIGHT, 
                                               [{"werewolf_target": killed}])
            if action.get("save"):
                save_target = action["save"]
                if witch_agent.use_save():
                    self.game.set_witch_save(save_target)
                    night_actions["witch_save"] = save_target
                    self._log_agent_action(witch, "witch_save", save_target)
            if action.get("poison"):
                poison_target = action["poison"]
                if witch_agent.use_poison():
                    self.game.set_witch_poison(poison_target)
                    night_actions["witch_poison"] = poison_target
                    self._log_agent_action(witch, "witch_poison", poison_target)

        # 守卫行动
        guard = next((pid for pid, agent in self.agents.items() 
                      if agent.role == Role.GUARD and agent.alive), None)
        if guard:
            guard_agent = self.agents[guard]
            game_state = self.get_game_state_for_agent(guard)
            action = guard_agent.decide_action(game_state, Phase.NIGHT, [])
            if action.get("action") == "protect" and "target_id" in action:
                target = action["target_id"]
                if self.game.set_guard_protect(target):
                    if hasattr(guard_agent, 'set_last_protect'):
                        guard_agent.set_last_protect(target)
                    night_actions["guard_protect"] = target
                    self._log_agent_action(guard, "guard_protect", target)

        # 结算夜晚结果
        dead_players = self.game.resolve_night()
        night_actions["dead_players"] = dead_players

        # 更新Agent状态
        for pid in dead_players:
            if pid in self.agents:
                self.agents[pid].alive = False
                # 猎人死亡触发技能
                if self.agents[pid].role == Role.HUNTER:
                    hunter_agent = self.agents[pid]
                    game_state = self.get_game_state_for_agent(pid)
                    if hasattr(hunter_agent, 'decide_shoot'):
                        target = hunter_agent.decide_shoot(game_state)
                        if target:
                            self.game.hunter_shoot(pid, target)
                            night_actions["hunter_shoot"] = {"hunter": pid, "target": target}
                            self._log_agent_action(pid, "hunter_shoot", target)

        # 记录回合统计
        self.round_stats.append({
            "round": self.game.round_number,
            "phase": "night",
            "actions": night_actions,
            "alive_count": len(self.game.get_alive_ids()),
            "timestamp": datetime.now().isoformat()
        })

        return night_actions

    def run_day_phase(self, night_dead: List[int]) -> dict:
        """执行白天阶段：发言、讨论、投票"""
        day_result = self.game.start_day_phase(night_dead)

        if day_result.get("phase") == "game_over":
            return day_result

        # 白天发言阶段
        self._run_discussion_phase()

        # 投票阶段
        return self._run_vote_phase()

    def _run_discussion_phase(self):
        """模拟白天讨论阶段：每个活着的Agent发言"""
        alive_agents = [agent for agent in self.agents.values() if agent.alive]
        
        # 随机发言顺序
        random.shuffle(alive_agents)

        for agent in alive_agents:
            game_state = self.get_game_state_for_agent(agent.player_id)
            speech = agent.generate_speech(game_state, "day_discussion")
            
            # 广播发言
            message = AgentMessage(
                sender_id=agent.player_id,
                message_type=MessageType.PUBLIC,
                content=speech
            )
            self._broadcast_message(message)

            # 其他Agent分析发言
            for other in alive_agents:
                if other.player_id != agent.player_id:
                    other.analyze_message(message)

            # 更新策略
            agent.update_strategy(game_state)

            # 模拟思考时间
            if self.event_callback:
                self.event_callback({
                    "type": "agent_speech",
                    "player_id": agent.player_id,
                    "speech": speech,
                    "agent_info": agent.get_belief_summary()
                })

    def _run_vote_phase(self) -> dict:
        """执行投票阶段"""
        self.game.start_vote()
        alive_agents = [agent for agent in self.agents.values() if agent.alive]

        # 每个Agent投票
        for agent in alive_agents:
            game_state = self.get_game_state_for_agent(agent.player_id)
            alive_ids = game_state.get("alive_ids", [])
            candidates = [pid for pid in alive_ids if pid != agent.player_id]

            vote_target = agent.decide_vote(game_state, candidates)
            if vote_target:
                self.game.cast_vote(agent.player_id, vote_target)

                # 记录投票消息
                message = AgentMessage(
                    sender_id=agent.player_id,
                    message_type=MessageType.VOTE,
                    content=f"玩家{agent.player_id}投票给玩家{vote_target}",
                    metadata={"target_id": vote_target}
                )
                self._broadcast_message(message)

        # 等待所有投票完成（最多3轮兜底）
        retry_count = 0
        while not self.game.all_voted() and retry_count < 3:
            alive_ids = self.game.get_alive_ids()
            voted = set(self.game.has_voted)
            not_voted = [pid for pid in alive_ids if pid not in voted]

            for pid in not_voted:
                agent = self.agents[pid]
                game_state = self.get_game_state_for_agent(pid)
                candidates = [p for p in alive_ids if p != pid]
                vote_target = agent.decide_vote(game_state, candidates)
                if not vote_target and candidates:
                    vote_target = random.choice(candidates)  # 兜底随机投票
                if vote_target:
                    self.game.cast_vote(pid, vote_target)
            retry_count += 1

        # 结算投票
        eliminated = self.game.resolve_vote()
        if eliminated and eliminated in self.agents:
            self.agents[eliminated].alive = False

        # 记录回合统计
        self.round_stats.append({
            "round": self.game.round_number,
            "phase": "day_vote",
            "eliminated": eliminated,
            "alive_count": len(self.game.get_alive_ids()),
            "timestamp": datetime.now().isoformat()
        })

        return {
            "phase": "day_vote_complete",
            "eliminated": eliminated,
            "alive_count": len(self.game.get_alive_ids())
        }

    def _log_agent_action(self, player_id: int, action_type: str, target: int, extra: Any = None):
        """记录Agent行动"""
        log_entry = {
            "player_id": player_id,
            "action": action_type,
            "target": target,
            "timestamp": datetime.now().isoformat()
        }
        if extra is not None:
            log_entry["extra"] = extra

        self.agent_actions[player_id].append(log_entry)

        if self.log_callback:
            self.log_callback(log_entry)

    def get_agent_summary(self) -> dict:
        """获取所有Agent的摘要信息"""
        summary = {}
        for pid, agent in self.agents.items():
            summary[pid] = {
                "id": pid,
                "name": agent.name,
                "role": agent.role.value,
                "team": agent.team.value,
                "alive": agent.alive,
                "is_sheriff": agent.is_sheriff,
                "memory": agent.memory.to_summary(),
                "actions_count": len(self.agent_actions.get(pid, [])),
                "recent_actions": self.agent_actions.get(pid, [])[-3:],
            }
        return summary

    def get_game_summary(self) -> dict:
        """获取游戏摘要"""
        winner = self.game.check_win()
        return {
            "total_rounds": self.game.round_number,
            "alive_players": len(self.game.get_alive_ids()),
            "winner": winner.value if winner else None,
            "message_count": len(self.message_history),
            "agent_summary": self.get_agent_summary(),
            "round_stats": self.round_stats,
        }

    def run_full_game(self) -> dict:
        """运行完整的一局游戏"""
        self.game.start_game()

        while True:
            # 夜晚阶段
            night_result = self.run_night_phase()
            if self.game.phase == Phase.GAME_OVER:
                break

            # 白天阶段
            day_result = self.run_day_phase(night_result.get("dead_players", []))
            if day_result.get("phase") == "game_over":
                break

            # 检查游戏是否结束
            if self.game.end_round():
                break

        # 游戏结束
        winner = self.game.check_win()
        return {
            "winner": winner.value if winner else "平局",
            "total_rounds": self.game.round_number,
            "game_summary": self.game.get_summary(),
            "agent_summary": self.get_agent_summary(),
            "message_history": self.message_history[-50:],  # 最后50条消息
            "round_stats": self.round_stats,
        }


class AgentTeamSimulator:
    """Agent Team 模拟器：批量运行多局游戏，收集数据"""

    def __init__(self, num_games: int = 10):
        self.num_games = num_games
        self.game_results: List[dict] = []
        self.statistics: Dict[str, Any] = {
            "total_games": 0,
            "good_wins": 0,
            "evil_wins": 0,
            "avg_rounds": 0,
            "role_performance": {},
            "agent_evolution": [],
        }

    def run_simulation(self) -> dict:
        """运行多局模拟"""
        for game_num in range(1, self.num_games + 1):
            print(f"正在运行第 {game_num} 局游戏...")

            # 创建新游戏
            game = WerewolfGameEngine()
            manager = AgentManager(game)

            # 运行游戏
            result = manager.run_full_game()
            result["game_number"] = game_num
            self.game_results.append(result)

            # 更新统计
            self._update_statistics(result)

            # 每5局输出一次进度
            if game_num % 5 == 0:
                print(f"已完成 {game_num} 局，当前统计：")
                print(json.dumps(self.statistics, ensure_ascii=False, indent=2))

        return self.statistics

    def _update_statistics(self, result: dict):
        """更新统计信息"""
        self.statistics["total_games"] += 1

        winner = result.get("winner", "")
        if "好人" in winner:
            self.statistics["good_wins"] += 1
        elif "狼人" in winner:
            self.statistics["evil_wins"] += 1

        # 平均回合数
        total_rounds = self.statistics.get("total_rounds", 0) + result.get("total_rounds", 0)
        self.statistics["total_rounds"] = total_rounds
        self.statistics["avg_rounds"] = total_rounds / self.statistics["total_games"]

        # 角色表现分析
        agent_summary = result.get("agent_summary", {})
        for pid, info in agent_summary.items():
            role = info.get("role")
            if role not in self.statistics["role_performance"]:
                self.statistics["role_performance"][role] = {
                    "total": 0,
                    "wins": 0,
                    "avg_survival_rounds": 0,
                }

            role_stats = self.statistics["role_performance"][role]
            role_stats["total"] += 1
            if info.get("alive") and "好人" in winner:
                role_stats["wins"] += 1
            elif not info.get("alive") and "狼人" in winner:
                role_stats["wins"] += 1

    def get_detailed_report(self) -> dict:
        """生成详细报告"""
        return {
            "simulation_summary": self.statistics,
            "game_results": self.game_results,
            "analysis": self._analyze_results(),
        }

    def _analyze_results(self) -> dict:
        """分析游戏结果"""
        analysis = {
            "win_rates": {
                "good": self.statistics["good_wins"] / max(1, self.statistics["total_games"]),
                "evil": self.statistics["evil_wins"] / max(1, self.statistics["total_games"]),
            },
            "role_effectiveness": {},
            "common_strategies": {},
            "agent_behavior_patterns": [],
        }

        # 角色胜率
        for role, stats in self.statistics["role_performance"].items():
            win_rate = stats["wins"] / max(1, stats["total"])
            analysis["role_effectiveness"][role] = {
                "win_rate": win_rate,
                "total_games": stats["total"],
            }

        return analysis


def run_single_game_demo():
    """运行单局游戏演示"""
    print("=== 狼人杀 AI Agent Team 演示 ===")
    print("正在初始化游戏...")

    # 创建游戏
    game = WerewolfGameEngine(
        player_names=[f"玩家{i}" for i in range(1, 13)]
    )

    # 创建Agent管理器
    def log_callback(log):
        print(f"[LOG] {log}")

    def event_callback(event):
        event_type = event.get("type")
        if event_type == "agent_message":
            msg = event.get("message", {})
            print(f"[MSG] {msg.get('sender')} -> {msg.get('type')}: {msg.get('content')[:50]}...")
        elif event_type == "agent_speech":
            print(f"[SPEECH] 玩家{event.get('player_id')}: {event.get('speech')}")

    manager = AgentManager(game, log_callback=log_callback, event_callback=event_callback)

    print("游戏开始！")
    result = manager.run_full_game()

    print("\n=== 游戏结果 ===")
    print(f"获胜阵营: {result['winner']}")
    print(f"总回合数: {result['total_rounds']}")
    print(f"存活玩家: {len([a for a in manager.agents.values() if a.alive])}")

    print("\n=== Agent 表现摘要 ===")
    for pid, summary in result['agent_summary'].items():
        print(f"玩家{pid} ({summary['role']}): {'存活' if summary['alive'] else '死亡'} - {summary['memory']['strategy']}")

    return result


if __name__ == "__main__":
    # 运行演示
    run_single_game_demo()

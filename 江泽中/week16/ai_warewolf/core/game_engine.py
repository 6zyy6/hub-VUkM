"""
游戏引擎模块
负责驱动游戏流程、处理回合逻辑、裁决胜负
"""

import random
import json
from typing import List, Dict, Optional
from ..core.game_state import GameState, GamePhase, PlayerStatus
from ..core.roles import RoleType, create_role
from ..agents.agent_factory import AgentFactory
from ..utils.logger import GameLogger


class GameEngine:
    """狼人杀游戏引擎"""

    def __init__(self, config: Dict, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        self.game_state = GameState(config.get('num_players', 12))
        self.agents = {}
        self.logger = GameLogger()

        self.roles_config = config.get('roles_distribution', {
            'werewolf': 4,
            'seer': 1,
            'witch': 1,
            'hunter': 1,
            'villager': 5
        })

    def initialize_game(self):
        """初始化游戏：分配角色、创建Agent"""
        self.logger.info("=== 游戏初始化 ===")

        roles = self._assign_roles()

        for player_id, role in enumerate(roles, start=1):
            player_name = f"Player_{player_id}"
            self.game_state.add_player(player_id, role, is_ai=True)

            agent = AgentFactory.create_agent(
                player_id=player_id,
                name=player_name,
                role=role,
                llm_client=self.llm_client,
                enable_memory=self.config.get('enable_memory', True),
                memory_size=self.config.get('memory_size', 50)
            )
            self.agents[player_id] = agent

            self.logger.info(f"玩家 {player_name} 获得角色: {role}")

        self._setup_werewolf_teams()

        self.game_state.add_event(GamePhase.SETUP, "game_start", "游戏开始")
        self.logger.success("游戏初始化完成！")

    def _assign_roles(self) -> List[str]:
        """随机分配角色"""
        roles = []
        for role_name, count in self.roles_config.items():
            roles.extend([role_name] * count)

        random.shuffle(roles)
        return roles

    def _setup_werewolf_teams(self):
        """设置狼人队友关系"""
        werewolf_ids = [
            pid for pid, player in self.game_state.players.items()
            if player.role == "werewolf"
        ]

        for wid in werewolf_ids:
            if hasattr(self.agents[wid], 'set_teammates'):
                teammates = [w for w in werewolf_ids if w != wid]
                self.agents[wid].set_teammates(teammates)

    def run_game(self) -> Dict:
        """运行完整游戏"""
        self.logger.info("=== 游戏开始 ===")

        max_rounds = self.config.get('max_rounds', 20)

        for round_num in range(1, max_rounds + 1):
            self.logger.info(f"\n{'=' * 50}")
            self.logger.info(f"第 {round_num} 轮")
            self.logger.info(f"{'=' * 50}\n")

            self.game_state.current_round = round_num

            # 夜晚阶段
            self._execute_night_phase()

            # 检查游戏是否结束
            if self.game_state.check_game_over():
                break

            # 白天阶段
            self._execute_day_phase()

            # 检查游戏是否结束
            if self.game_state.check_game_over():
                break

        # 游戏结束
        self._end_game()

        return self._get_game_result()

    def _execute_night_phase(self):
        """执行夜晚阶段"""
        self.logger.info("\n--- 夜晚阶段 ---")

        self.game_state.reset_night_actions()

        # 狼人行动
        self._werewolf_night_action()

        # 预言家行动
        self._seer_night_action()

        # 女巫行动
        self._witch_night_action()

        # 结算夜晚结果
        self._resolve_night_results()

    def _werewolf_night_action(self):
        """狼人夜间行动"""
        self.game_state.current_phase = GamePhase.NIGHT_WEREWOLF
        self.logger.info("\n【狼人行动】")

        alive_werewolves = self.game_state.get_alive_werewolves()
        if not alive_werewolves:
            return

        targets = []
        for wolf in alive_werewolves:
            agent = self.agents[wolf.player_id]
            decision = agent.night_action(self.game_state.to_dict())

            target_id = decision.get('target_id')
            if target_id:
                targets.append(target_id)
                self.logger.info(f"狼人 {wolf.player_id} 选择击杀: Player_{target_id}")

        if targets:
            from collections import Counter
            target_counts = Counter(targets)
            final_target = target_counts.most_common(1)[0][0]
            self.game_state.werewolf_targets.add(final_target)
            self.logger.info(f"狼人团队最终目标: Player_{final_target}")

    def _seer_night_action(self):
        """预言家夜间行动"""
        self.game_state.current_phase = GamePhase.NIGHT_SEER
        self.logger.info("\n【预言家行动】")

        alive_seers = [
            p for p in self.game_state.get_alive_players()
            if p.role == "seer"
        ]

        for seer in alive_seers:
            agent = self.agents[seer.player_id]
            decision = agent.night_action(self.game_state.to_dict())

            target_id = decision.get('target_id')
            if target_id:
                self.game_state.seer_check_target = target_id

                target_player = self.game_state.get_player_by_id(target_id)
                is_werewolf = target_player.role == "werewolf"
                self.game_state.seer_check_result = is_werewolf

                if hasattr(agent, 'receive_verification_result'):
                    agent.receive_verification_result(target_id, is_werewolf)

                self.logger.info(
                    f"预言家 {seer.player_id} 查验 Player_{target_id}: "
                    f"{'狼人' if is_werewolf else '好人'}"
                )

    def _witch_night_action(self):
        """女巫夜间行动"""
        self.game_state.current_phase = GamePhase.NIGHT_WITCH
        self.logger.info("\n【女巫行动】")

        alive_witches = [
            p for p in self.game_state.get_alive_players()
            if p.role == "witch"
        ]

        for witch in alive_witches:
            agent = self.agents[witch.player_id]

            victim_id = list(self.game_state.werewolf_targets)[0] \
                if self.game_state.werewolf_targets else None

            decision = agent.night_action(
                self.game_state.to_dict(),
                victim_id=victim_id
            )

            if decision.get('use_save') and victim_id:
                save_target = decision.get('save_target', victim_id)
                if save_target in self.game_state.werewolf_targets:
                    self.game_state.witch_save_target = save_target
                    self.logger.info(f"女巫 {witch.player_id} 使用解药救了 Player_{save_target}")

            if decision.get('use_poison'):
                poison_target = decision.get('poison_target')
                if poison_target:
                    self.game_state.witch_poison_target = poison_target
                    self.logger.info(f"女巫 {witch.player_id} 对 Player_{poison_target} 使用毒药")

    def _resolve_night_results(self):
        """结算夜晚结果"""
        self.logger.info("\n【夜晚结算】")

        deaths = []

        for target_id in self.game_state.werewolf_targets:
            if target_id == self.game_state.witch_save_target:
                self.logger.info(f"Player_{target_id} 被女巫救活")
                continue

            player = self.game_state.get_player_by_id(target_id)
            if player and player.status == PlayerStatus.ALIVE:
                player.status = PlayerStatus.DEAD
                player.is_alive = False
                deaths.append(target_id)
                self.logger.info(f"Player_{target_id} 被狼人杀害")

                self._handle_death_event(target_id)

        if self.game_state.witch_poison_target:
            target_id = self.game_state.witch_poison_target
            player = self.game_state.get_player_by_id(target_id)
            if player and player.status == PlayerStatus.ALIVE:
                player.status = PlayerStatus.DEAD
                player.is_alive = False
                deaths.append(target_id)
                self.logger.info(f"Player_{target_id} 被女巫毒杀")

                self._handle_death_event(target_id)

        if deaths:
            self.game_state.add_event(
                GamePhase.NIGHT_WEREWOLF,
                "night_deaths",
                f"夜晚死亡玩家: {[f'Player_{d}' for d in deaths]}",
                details={"deaths": deaths}
            )
        else:
            self.logger.info("夜晚无人死亡")

    def _execute_day_phase(self):
        """执行白天阶段"""
        self.logger.info("\n--- 白天阶段 ---")

        self._announce_night_results()

        self._discussion_phase()

        self._voting_phase()

    def _announce_night_results(self):
        """公布夜晚结果"""
        self.game_state.current_phase = GamePhase.DAY_DISCUSSION
        self.logger.info("\n【公布结果】")

        dead_players = [
            p for p in self.game_state.players.values()
            if p.status == PlayerStatus.DEAD
        ]

        if dead_players:
            dead_ids = [p.player_id for p in dead_players]
            self.logger.info(f"昨晚死亡的玩家: {[f'Player_{d}' for d in dead_ids]}")
        else:
            self.logger.info("昨晚是平安夜")

    def _discussion_phase(self):
        """讨论阶段"""
        self.logger.info("\n【讨论阶段】")

        alive_players = self.game_state.get_alive_players()
        speech_history = []

        for player in alive_players:
            agent = self.agents[player.player_id]

            self.logger.info(f"\n--- Player_{player.player_id} ({player.role}) 发言 ---")

            speech = agent.day_speech(
                self.game_state.to_dict(),
                speech_history=speech_history
            )

            self.logger.info(f"发言内容: {speech[:200]}...")

            speech_record = {
                "player_id": player.player_id,
                "player_name": player.name,
                "content": speech
            }
            speech_history.append(speech_record)
            self.game_state.discussion_log.append(speech_record)

        self.game_state.add_event(
            GamePhase.DAY_DISCUSSION,
            "discussion_complete",
            f"讨论阶段结束，共{len(speech_history)}名玩家发言"
        )

    def _voting_phase(self):
        """投票阶段"""
        self.game_state.current_phase = GamePhase.DAY_VOTING
        self.logger.info("\n【投票阶段】")

        alive_players = self.game_state.get_alive_players()
        candidates = [
            {"player_id": p.player_id, "name": p.name}
            for p in alive_players
        ]

        votes = {}
        for voter in alive_players:
            agent = self.agents[voter.player_id]
            vote_target = agent.voting_decision(
                self.game_state.to_dict(),
                candidates=candidates
            )

            votes[voter.player_id] = vote_target
            self.logger.info(
                f"Player_{voter.player_id} 投票给 Player_{vote_target}"
            )

        self.game_state.votes = votes

        self._count_votes()

    def _count_votes(self):
        """统计票数"""
        from collections import Counter

        vote_targets = list(self.game_state.votes.values())
        vote_counts = Counter(vote_targets)

        self.logger.info("\n【投票结果】")
        for target_id, count in vote_counts.most_common():
            self.logger.info(f"Player_{target_id}: {count}票")

        if vote_counts:
            max_votes = vote_counts.most_common(1)[0][1]
            candidates_with_max = [
                tid for tid, cnt in vote_counts.items()
                if cnt == max_votes
            ]

            if len(candidates_with_max) > 1:
                eliminated = random.choice(candidates_with_max)
                self.logger.info(f"平票！随机淘汰 Player_{eliminated}")
            else:
                eliminated = candidates_with_max[0]

            player = self.game_state.get_player_by_id(eliminated)
            if player:
                player.status = PlayerStatus.DEAD
                player.is_alive = False
                self.logger.info(f"Player_{eliminated} 被淘汰")

                self._handle_death_event(eliminated)

                self.game_state.add_event(
                    GamePhase.DAY_VOTING,
                    "player_eliminated",
                    f"Player_{eliminated} 被投票淘汰",
                    target_id=eliminated
                )

    def _handle_death_event(self, player_id: int):
        """处理玩家死亡事件"""
        player = self.game_state.get_player_by_id(player_id)
        if not player:
            return

        agent = self.agents.get(player_id)
        if not agent:
            return

        if player.role == "hunter":
            self.logger.info(f"\n【猎人技能】Player_{player_id} 发动临终射击")

            death_action = agent.on_death(self.game_state.to_dict())
            if death_action:
                shoot_target = death_action.get('shoot_target', -1)
                if shoot_target > 0:
                    target_player = self.game_state.get_player_by_id(shoot_target)
                    if target_player and target_player.status == PlayerStatus.ALIVE:
                        target_player.status = PlayerStatus.DEAD
                        target_player.is_alive = False
                        self.logger.info(
                            f"猎人带走了 Player_{shoot_target}"
                        )

                        self._handle_death_event(shoot_target)

    def _end_game(self):
        """游戏结束处理"""
        self.game_state.current_phase = GamePhase.GAME_OVER
        self.logger.info("\n" + "=" * 50)
        self.logger.info("游戏结束！")
        self.logger.info(f"获胜方: {self.game_state.winner}")
        self.logger.info("=" * 50)

        self.game_state.add_event(
            GamePhase.GAME_OVER,
            "game_end",
            f"游戏结束，{self.game_state.winner} 获胜"
        )

    def _get_game_result(self) -> Dict:
        """获取游戏结果"""
        result = {
            "winner": self.game_state.winner,
            "total_rounds": self.game_state.current_round,
            "players": {},
            "events": [
                {
                    "timestamp": e.timestamp,
                    "phase": e.phase.value,
                    "event_type": e.event_type,
                    "description": e.description
                }
                for e in self.game_state.events
            ]
        }

        for pid, player in self.game_state.players.items():
            result["players"][pid] = {
                "name": player.name,
                "role": player.role,
                "status": player.status.value,
                "is_alive": player.is_alive
            }

        return result

"""
自进化 Agent 模块
实现"对局 → 分析 → 优化 → 再对局"的自我进化循环
"""

import json
import sqlite3
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import Counter


@dataclass
class GameRecord:
    """单局游戏记录"""
    game_id: str
    timestamp: str
    winner: str
    total_rounds: int
    player_count: int
    players: List[dict]  # 每个玩家的角色、存活、行动记录
    events: List[dict]  # 关键事件序列


@dataclass
class StrategyProfile:
    """角色策略档案"""
    role: str
    games_played: int
    wins: int
    avg_survival_rounds: float
    preferred_actions: Dict[str, int]
    success_rate: Dict[str, float]
    strategy_params: Dict[str, float]


class EvolutionDatabase:
    """进化数据库：存储和分析对局数据"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), '..', 'evolution.db')
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 游戏记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                winner TEXT NOT NULL,
                total_rounds INTEGER,
                player_count INTEGER,
                game_data TEXT
            )
        ''')

        # 玩家表现表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                player_id INTEGER,
                role TEXT,
                alive INTEGER,
                actions TEXT,
                survival_rounds INTEGER,
                vote_accuracy REAL,
                FOREIGN KEY (game_id) REFERENCES games(id)
            )
        ''')

        # 策略档案表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_profiles (
                role TEXT PRIMARY KEY,
                games_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                total_survival_rounds REAL DEFAULT 0.0,
                avg_survival_rounds REAL DEFAULT 0.0,
                params TEXT,
                updated_at TEXT
            )
        ''')

        # 策略进化历史
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                version INTEGER,
                before_params TEXT,
                after_params TEXT,
                reason TEXT,
                performance_change REAL,
                timestamp TEXT
            )
        ''')

        # 行动模式统计
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS action_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                action_type TEXT,
                context TEXT,
                chosen_target TEXT,
                success INTEGER,
                timestamp TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def save_game(self, record: GameRecord):
        """保存游戏记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'INSERT OR REPLACE INTO games (id, timestamp, winner, total_rounds, player_count, game_data) VALUES (?, ?, ?, ?, ?, ?)',
            (record.game_id, record.timestamp, record.winner, record.total_rounds,
             record.player_count, json.dumps(asdict(record), ensure_ascii=False))
        )

        # 保存玩家表现
        for player in record.players:
            cursor.execute(
                'INSERT INTO player_performance (game_id, player_id, role, alive, actions, survival_rounds) VALUES (?, ?, ?, ?, ?, ?)',
                (record.game_id, player.get('id'), player.get('role'),
                 1 if player.get('alive') else 0,
                 json.dumps(player.get('actions', []), ensure_ascii=False),
                 player.get('survival_rounds', 0))
            )

        conn.commit()
        conn.close()

    def get_statistics(self) -> dict:
        """获取全局统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # 总对局数
        cursor.execute('SELECT COUNT(*), SUM(CASE WHEN winner="好人阵营" THEN 1 ELSE 0 END), SUM(CASE WHEN winner="狼人阵营" THEN 1 ELSE 0 END) FROM games')
        total, good_wins, evil_wins = cursor.fetchone() or (0, 0, 0)
        stats['total_games'] = total
        stats['good_wins'] = good_wins or 0
        stats['evil_wins'] = evil_wins or 0
        stats['good_win_rate'] = (good_wins or 0) / max(1, total) * 100
        stats['evil_win_rate'] = (evil_wins or 0) / max(1, total) * 100

        # 平均回合数
        cursor.execute('SELECT AVG(total_rounds) FROM games')
        avg = cursor.fetchone()[0]
        stats['avg_rounds'] = avg or 0

        # 角色表现
        cursor.execute('''
            SELECT role, COUNT(*), SUM(alive), AVG(survival_rounds)
            FROM player_performance
            GROUP BY role
        ''')
        stats['role_performance'] = {}
        for role, count, alive, avg_rounds in cursor.fetchall():
            stats['role_performance'][role] = {
                'total': count,
                'wins': alive or 0,
                'win_rate': (alive or 0) / max(1, count) * 100,
                'avg_survival': avg_rounds or 0
            }

        conn.close()
        return stats

    def get_role_profile(self, role: str) -> Optional[StrategyProfile]:
        """获取角色策略档案"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT games_played, wins, avg_survival_rounds, params FROM strategy_profiles WHERE role = ?',
            (role,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return StrategyProfile(
            role=role,
            games_played=row[0],
            wins=row[1],
            avg_survival_rounds=row[2],
            preferred_actions={},
            success_rate={},
            strategy_params=json.loads(row[3]) if row[3] else {}
        )

    def update_strategy_profile(self, role: str, params: dict, reason: str = ""):
        """更新角色策略参数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取旧参数
        cursor.execute('SELECT params, games_played FROM strategy_profiles WHERE role = ?', (role,))
        row = cursor.fetchone()
        old_params = json.loads(row[0]) if row and row[0] else {}
        version = (row[1] or 0) + 1 if row else 1

        # 更新档案
        params_json = json.dumps(params, ensure_ascii=False)
        cursor.execute('''
            INSERT OR REPLACE INTO strategy_profiles (role, games_played, wins, total_survival_rounds, avg_survival_rounds, params, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (role, version, 0, 0.0, 0.0, params_json, datetime.now().isoformat()))

        # 记录进化历史
        cursor.execute('''
            INSERT INTO strategy_evolution (role, version, before_params, after_params, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (role, version, json.dumps(old_params, ensure_ascii=False),
              params_json, reason, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def record_action_pattern(self, role: str, action_type: str, context: str,
                              target: str, success: bool):
        """记录行动模式"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO action_patterns (role, action_type, context, chosen_target, success, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
            (role, action_type, context, target, 1 if success else 0, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

    def get_action_stats(self, role: str, action_type: str) -> dict:
        """获取行动统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT COUNT(*), SUM(success) FROM action_patterns WHERE role = ? AND action_type = ?',
            (role, action_type)
        )
        total, success = cursor.fetchone() or (0, 0)

        # 获取最常选择的目标
        cursor.execute(
            'SELECT chosen_target, COUNT(*) as cnt FROM action_patterns WHERE role = ? AND action_type = ? GROUP BY chosen_target ORDER BY cnt DESC LIMIT 5',
            (role, action_type)
        )
        top_targets = cursor.fetchall()

        conn.close()
        return {
            'total': total,
            'success_rate': (success or 0) / max(1, total) * 100,
            'top_targets': top_targets
        }


class StrategyOptimizer:
    """策略优化器：分析数据并优化Agent策略参数"""

    def __init__(self, db: EvolutionDatabase):
        self.db = db
        self.optimization_history: List[dict] = []

    def analyze_role_performance(self, role: str, min_games: int = 10) -> Optional[dict]:
        """分析角色表现，找出优化方向"""
        stats = self.db.get_statistics()
        role_stats = stats.get('role_performance', {}).get(role, {})

        if role_stats.get('total', 0) < min_games:
            return None

        # 分析各行动类型的成功率
        action_analysis = {}
        for action_type in ['vote', 'kill', 'check', 'save', 'poison', 'protect']:
            action_stats = self.db.get_action_stats(role, action_type)
            if action_stats['total'] > 0:
                action_analysis[action_type] = action_stats

        # 找出表现最好和最差的行为
        if action_analysis:
            best = max(action_analysis.items(), key=lambda x: x[1]['success_rate'])
            worst = min(action_analysis.items(), key=lambda x: x[1]['success_rate'])

            return {
                'role': role,
                'current_win_rate': role_stats.get('win_rate', 50),
                'avg_survival': role_stats.get('avg_survival', 1),
                'best_action': best[0],
                'best_success_rate': best[1]['success_rate'],
                'worst_action': worst[0],
                'worst_success_rate': worst[1]['success_rate'],
                'action_analysis': action_analysis,
            }

        return None

    def generate_optimization(self, role: str, analysis: dict) -> dict:
        """基于分析生成优化参数"""
        profile = self.db.get_role_profile(role)
        current_params = profile.strategy_params if profile else {}

        optimized = current_params.copy()

        # 根据最差行为调整参数
        worst = analysis.get('worst_action')
        worst_rate = analysis.get('worst_success_rate', 50)

        if worst == 'vote' and worst_rate < 60:
            # 提高投票谨慎度
            optimized['vote_cautiousness'] = min(1.0, current_params.get('vote_cautiousness', 0.5) + 0.1)
            optimized['vote_herd_weight'] = max(0.0, current_params.get('vote_herd_weight', 0.3) - 0.05)

        if worst == 'kill' and worst_rate < 50:
            # 调整威胁评估权重
            optimized['threat_assessment_weight'] = min(1.0, current_params.get('threat_assessment_weight', 0.6) + 0.1)
            optimized['random_kill_chance'] = max(0.0, current_params.get('random_kill_chance', 0.1) - 0.02)

        if worst == 'check' and worst_rate < 70:
            # 优化查验目标选择
            optimized['check_unknown_priority'] = min(1.0, current_params.get('check_unknown_priority', 0.7) + 0.1)
            optimized['check_suspicious_bias'] = min(1.0, current_params.get('check_suspicious_bias', 0.5) + 0.05)

        if worst == 'save' and worst_rate < 60:
            # 调整解药使用策略
            optimized['save_aggressiveness'] = max(0.0, current_params.get('save_aggressiveness', 0.5) - 0.05)
            optimized['save_seer_priority'] = min(1.0, current_params.get('save_seer_priority', 0.8) + 0.1)

        if worst == 'poison' and worst_rate < 50:
            # 调整毒药使用策略
            optimized['poison_caution'] = min(1.0, current_params.get('poison_caution', 0.5) + 0.1)
            optimized['poison_wolf_threshold'] = max(0.0, current_params.get('poison_wolf_threshold', 0.6) - 0.05)

        if worst == 'protect' and worst_rate < 60:
            # 优化守护目标
            optimized['protect_seer_priority'] = min(1.0, current_params.get('protect_seer_priority', 0.7) + 0.1)
            optimized['protect_self_weight'] = max(0.0, current_params.get('protect_self_weight', 0.2) - 0.05)

        # 限制参数范围
        for key in optimized:
            optimized[key] = max(0.0, min(1.0, optimized[key]))
            optimized[key] = round(optimized[key], 3)

        reason = f"优化{role}策略：{worst}成功率{worst_rate:.1f}%偏低，调整{(len(optimized)-len(current_params))}个参数"

        return {
            'role': role,
            'before': current_params,
            'after': optimized,
            'reason': reason,
            'changes': {k: {'from': current_params.get(k), 'to': v}
                       for k, v in optimized.items()
                       if k not in current_params or current_params[k] != v}
        }

    def apply_optimization(self, optimization: dict):
        """应用优化到数据库"""
        role = optimization['role']
        params = optimization['after']
        reason = optimization['reason']

        self.db.update_strategy_profile(role, params, reason)
        self.optimization_history.append(optimization)

    def run_evolution_cycle(self, min_games: int = 10) -> List[dict]:
        """运行一轮完整的进化循环"""
        results = []

        roles = ['狼人', '预言家', '女巫', '猎人', '守卫', '村民']
        for role in roles:
            analysis = self.analyze_role_performance(role, min_games)
            if analysis:
                optimization = self.generate_optimization(role, analysis)
                self.apply_optimization(optimization)
                results.append(optimization)

        return results

    def get_evolution_report(self) -> dict:
        """获取进化报告"""
        stats = self.db.get_statistics()
        profiles = {}
        for role in ['狼人', '预言家', '女巫', '猎人', '守卫', '村民']:
            p = self.db.get_role_profile(role)
            if p:
                profiles[role] = asdict(p)

        return {
            'total_games': stats['total_games'],
            'current_statistics': stats,
            'strategy_profiles': profiles,
            'optimization_history': self.optimization_history[-10:],
            'generated_at': datetime.now().isoformat()
        }


class EvolvableAgent:
    """可进化Agent包装器：在基础Agent上增加进化参数支持"""

    def __init__(self, base_agent, db: EvolutionDatabase):
        self.base_agent = base_agent
        self.db = db
        self.evolution_params = {}
        self.generation = 0
        self._load_evolution_params()

    def _load_evolution_params(self):
        """从数据库加载进化参数"""
        profile = self.db.get_role_profile(self.base_agent.role.value)
        if profile and profile.strategy_params:
            self.evolution_params = profile.strategy_params
            self._apply_params_to_agent()

    def _apply_params_to_agent(self):
        """将进化参数应用到基础Agent"""
        for key, value in self.evolution_params.items():
            if hasattr(self.base_agent, key):
                setattr(self.base_agent, key, value)

    def decide_action(self, *args, **kwargs):
        """增强的决策：记录行动用于后续分析"""
        action = self.base_agent.decide_action(*args, **kwargs)

        # 记录行动模式
        if action.get('action') and action.get('target_id'):
            self.db.record_action_pattern(
                role=self.base_agent.role.value,
                action_type=action['action'],
                context=str(kwargs.get('game_state', {}))[:100],
                target=str(action['target_id']),
                success=False  # 成功与否待结算后更新
            )

        return action

    def update_action_result(self, action_type: str, target: str, success: bool):
        """更新行动结果"""
        # 简化处理：直接插入新的成功记录
        self.db.record_action_pattern(
            role=self.base_agent.role.value,
            action_type=action_type,
            context='result_update',
            target=target,
            success=success
        )


def create_evolution_report(db_path: str = None) -> dict:
    """创建进化报告"""
    db = EvolutionDatabase(db_path)
    optimizer = StrategyOptimizer(db)
    report = optimizer.get_evolution_report()

    # 保存报告
    output_path = os.path.join(os.path.dirname(__file__), '..', 'evolution_report.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    return report


if __name__ == "__main__":
    # 测试进化模块
    db = EvolutionDatabase()
    optimizer = StrategyOptimizer(db)
    report = optimizer.get_evolution_report()

    print("=== 自进化系统报告 ===")
    print(f"总对局数: {report['total_games']}")
    if report['strategy_profiles']:
        for role, profile in report['strategy_profiles'].items():
            print(f"  {role}: {profile['games_played']}局 胜率{(profile['wins']/max(1,profile['games_played'])*100):.1f}%")
    print(f"策略档案数: {len(report['strategy_profiles'])}")

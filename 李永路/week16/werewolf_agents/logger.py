"""
结构化日志系统
实现游戏过程的全程可观测性
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class GameLogger:
    """游戏日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 设置日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"werewolf_{timestamp}.log"
        self.report_file = self.log_dir / f"report_{timestamp}.json"
        
        # 配置 logger
        self.logger = logging.getLogger("WerewolfGame")
        self.logger.setLevel(logging.DEBUG)
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_game_start(self, game_id: str, player_count: int):
        """记录游戏开始"""
        self.logger.info(f"===== 游戏 {game_id} 开始 =====")
        self.logger.info(f"参与玩家数量: {player_count}")
    
    def log_round_start(self, round_num: int):
        """记录回合开始"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"第 {round_num} 回合开始")
        self.logger.info(f"{'='*60}")
    
    def log_night_actions(self, actions: List[Dict]):
        """记录夜间行动"""
        self.logger.info("\n[夜间行动]")
        for action in actions:
            self.logger.info(f"  - {action}")
    
    def log_day_discussion(self, discussions: List[Dict]):
        """记录白天讨论"""
        self.logger.info("\n[白天讨论]")
        for discussion in discussions:
            self.logger.info(f"  [{discussion['speaker']}] {discussion['content']}")
    
    def log_voting_results(self, votes: Dict[str, str], eliminated: Optional[str]):
        """记录投票结果"""
        self.logger.info("\n[投票结果]")
        for voter, target in votes.items():
            self.logger.info(f"  {voter} -> {target}")
        
        if eliminated:
            self.logger.info(f"\n被淘汰: {eliminated}")
    
    def log_game_end(self, winner: str, total_rounds: int):
        """记录游戏结束"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"游戏结束！")
        self.logger.info(f"获胜方: {winner}")
        self.logger.info(f"总回合数: {total_rounds}")
        self.logger.info(f"{'='*60}")
    
    def save_report(self, report: Dict):
        """保存游戏报告"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n游戏报告已保存至: {self.report_file}")
    
    def get_log_file_path(self) -> Path:
        """获取日志文件路径"""
        return self.log_file
    
    def get_report_file_path(self) -> Path:
        """获取报告文件路径"""
        return self.report_file


class ObservableGameState:
    """可观测的游戏状态"""
    
    def __init__(self, game_logger: GameLogger):
        self.logger = game_logger
        self.observation_history: List[Dict] = []
    
    def record_observation(self, phase: str, data: Dict):
        """记录观测数据"""
        observation = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.observation_history.append(observation)
        
        # 根据阶段记录不同的日志
        if phase == "night_action":
            self.logger.log_night_actions(data.get("actions", []))
        elif phase == "day_discussion":
            self.logger.log_day_discussion(data.get("discussions", []))
        elif phase == "voting":
            self.logger.log_voting_results(
                data.get("votes", {}),
                data.get("eliminated")
            )
    
    def get_observations(self, phase: Optional[str] = None) -> List[Dict]:
        """获取观测历史"""
        if phase:
            return [obs for obs in self.observation_history if obs["phase"] == phase]
        return self.observation_history
    
    def export_observations(self, output_file: str = "observations.json"):
        """导出观测数据"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.observation_history, f, ensure_ascii=False, indent=2)

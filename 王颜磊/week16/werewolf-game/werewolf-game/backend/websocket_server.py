"""
WebSocket 服务器 - 实时观战系统
支持多客户端连接，实时推送游戏状态、Agent决策、发言等
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol

try:
    from .game_engine import WerewolfGameEngine, GameEvent, GameLog, Phase
    from .agent_manager import AgentManager
    from .agents import BaseAgent, AgentMessage
except ImportError:
    from game_engine import WerewolfGameEngine, GameEvent, GameLog, Phase
    from agent_manager import AgentManager
    from agents import BaseAgent, AgentMessage


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WerewolfWebSocketServer:
    """狼人杀WebSocket服务器"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.game: Optional[WerewolfGameEngine] = None
        self.manager: Optional[AgentManager] = None
        self.game_running: bool = False
        self.game_history: List[dict] = []

        # 订阅系统
        self.subscriptions: Dict[str, Set[WebSocketServerProtocol]] = {
            "game_state": set(),
            "agent_messages": set(),
            "agent_actions": set(),
            "game_logs": set(),
            "all": set(),
        }

    async def broadcast(self, message_type: str, data: dict):
        """广播消息给订阅了该类型的客户端"""
        message = {"type": message_type, "data": data, "timestamp": datetime.now().isoformat()}
        json_message = json.dumps(message, ensure_ascii=False)

        # 发送给订阅了该类型或所有类型的客户端
        targets = set()
        if message_type in self.subscriptions:
            targets.update(self.subscriptions[message_type])
        targets.update(self.subscriptions["all"])

        if targets:
            # 尝试发送，忽略已断开的连接
            async def safe_send(client):
                try:
                    await client.send(json_message)
                except Exception:
                    self.subscriptions.get(message_type, set()).discard(client)
                    self.subscriptions.get("all", set()).discard(client)
            await asyncio.gather(*[safe_send(client) for client in targets], return_exceptions=True)

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """处理客户端连接"""
        self.clients.add(websocket)
        client_id = id(websocket)
        logger.info(f"客户端 {client_id} 已连接")

        try:
            # 发送欢迎消息
            await websocket.send(json.dumps({
                "type": "welcome",
                "message": "欢迎来到狼人杀 AI Agent Team 观战系统",
                "server_info": {
                    "host": self.host,
                    "port": self.port,
                    "clients": len(self.clients),
                    "game_running": self.game_running,
                }
            }))

            # 处理客户端消息
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端 {client_id} 断开连接")
        finally:
            self.clients.remove(websocket)
            # 清理订阅
            for sub_type in self.subscriptions:
                self.subscriptions[sub_type].discard(websocket)

    async def handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            payload = data.get("data", {})

            if msg_type == "subscribe":
                await self.handle_subscribe(websocket, payload)
            elif msg_type == "unsubscribe":
                await self.handle_unsubscribe(websocket, payload)
            elif msg_type == "start_game":
                await self.handle_start_game(websocket, payload)
            elif msg_type == "stop_game":
                await self.handle_stop_game(websocket)
            elif msg_type == "get_game_state":
                await self.handle_get_game_state(websocket)
            elif msg_type == "get_agent_info":
                await self.handle_get_agent_info(websocket, payload)
            elif msg_type == "control_game":
                await self.handle_control_game(websocket, payload)
            elif msg_type == "simulate_games":
                await self.handle_simulate_games(websocket, payload)
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"未知的消息类型: {msg_type}"
                }))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "无效的JSON格式"
            }))
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"服务器错误: {str(e)}"
            }))

    async def handle_subscribe(self, websocket: WebSocketServerProtocol, payload: dict):
        """处理订阅请求"""
        sub_type = payload.get("channel", "all")
        if sub_type in self.subscriptions:
            self.subscriptions[sub_type].add(websocket)
            await websocket.send(json.dumps({
                "type": "subscription_confirmed",
                "channel": sub_type,
                "message": f"已订阅 {sub_type} 频道"
            }))
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"未知的频道: {sub_type}"
            }))

    async def handle_unsubscribe(self, websocket: WebSocketServerProtocol, payload: dict):
        """处理取消订阅"""
        sub_type = payload.get("channel", "all")
        if sub_type in self.subscriptions:
            self.subscriptions[sub_type].discard(websocket)
            await websocket.send(json.dumps({
                "type": "unsubscription_confirmed",
                "channel": sub_type,
                "message": f"已取消订阅 {sub_type} 频道"
            }))

    async def handle_start_game(self, websocket: WebSocketServerProtocol, payload: dict):
        """开始新游戏"""
        if self.game_running:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "游戏正在进行中"
            }))
            return

        try:
            # 创建新游戏
            player_names = payload.get("player_names", [f"玩家{i}" for i in range(1, 13)])
            self.game = WerewolfGameEngine(player_names=player_names)

            # 创建Agent管理器
            def log_callback(log):
                asyncio.create_task(self.broadcast("game_log", log))

            def event_callback(event):
                asyncio.create_task(self.broadcast("game_event", event))

            self.manager = AgentManager(self.game, log_callback=log_callback, event_callback=event_callback)

            # 开始游戏
            self.game_running = True
            self.game_history = []

            # 广播游戏开始
            await self.broadcast("game_started", {
                "player_count": 12,
                "player_names": player_names,
                "timestamp": datetime.now().isoformat()
            })

            # 发送初始游戏状态
            await self.broadcast("game_state", self.game.game_state_dict())

            # 发送Agent信息
            await self.broadcast("agent_summary", self.manager.get_agent_summary())

            await websocket.send(json.dumps({
                "type": "game_started",
                "message": "游戏已开始",
                "game_id": id(self.game)
            }))

            # 在后台运行游戏
            asyncio.create_task(self.run_game_loop())

        except Exception as e:
            logger.error(f"开始游戏时出错: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"开始游戏失败: {str(e)}"
            }))

    async def handle_stop_game(self, websocket: WebSocketServerProtocol):
        """停止当前游戏"""
        if not self.game_running:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "没有正在进行的游戏"
            }))
            return

        self.game_running = False
        await self.broadcast("game_stopped", {
            "message": "游戏已停止",
            "timestamp": datetime.now().isoformat()
        })

        await websocket.send(json.dumps({
            "type": "game_stopped",
            "message": "游戏已停止"
        }))

    async def handle_get_game_state(self, websocket: WebSocketServerProtocol):
        """获取当前游戏状态"""
        if not self.game:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "没有正在进行的游戏"
            }))
            return

        await websocket.send(json.dumps({
            "type": "game_state",
            "data": self.game.game_state_dict()
        }))

    async def handle_get_agent_info(self, websocket: WebSocketServerProtocol, payload: dict):
        """获取Agent信息"""
        if not self.manager:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "没有正在进行的游戏"
            }))
            return

        player_id = payload.get("player_id")
        if player_id:
            # 获取特定Agent信息
            agent = self.manager.agents.get(player_id)
            if agent:
                await websocket.send(json.dumps({
                    "type": "agent_info",
                    "data": {
                        "id": player_id,
                        "info": agent.to_dict(),
                        "belief_summary": agent.get_belief_summary(),
                        "recent_messages": [m.to_dict() for m in agent.get_messages()[-10:]]
                    }
                }))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"未找到玩家 {player_id}"
                }))
        else:
            # 获取所有Agent信息
            await websocket.send(json.dumps({
                "type": "agent_summary",
                "data": self.manager.get_agent_summary()
            }))

    async def handle_control_game(self, websocket: WebSocketServerProtocol, payload: dict):
        """控制游戏进度"""
        if not self.game_running or not self.game:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "没有正在进行的游戏"
            }))
            return

        action = payload.get("action")
        if action == "next_phase":
            # 手动进入下一阶段（用于调试）
            await self.broadcast("game_control", {
                "action": "next_phase",
                "message": "手动进入下一阶段"
            })
        elif action == "pause":
            # 暂停游戏
            await self.broadcast("game_control", {
                "action": "pause",
                "message": "游戏已暂停"
            })
        elif action == "resume":
            # 恢复游戏
            await self.broadcast("game_control", {
                "action": "resume",
                "message": "游戏已恢复"
            })

        await websocket.send(json.dumps({
            "type": "control_acknowledged",
            "action": action
        }))

    async def handle_simulate_games(self, websocket: WebSocketServerProtocol, payload: dict):
        """批量模拟游戏"""
        num_games = payload.get("num_games", 10)
        await websocket.send(json.dumps({
            "type": "simulation_started",
            "message": f"开始模拟 {num_games} 局游戏",
            "num_games": num_games
        }))

        # 在后台运行模拟
        asyncio.create_task(self.run_simulation(websocket, num_games))

    async def run_simulation(self, websocket: WebSocketServerProtocol, num_games: int):
        """运行批量模拟"""
        from .agent_manager import AgentTeamSimulator

        simulator = AgentTeamSimulator(num_games=num_games)
        results = simulator.run_simulation()

        # 发送结果
        await websocket.send(json.dumps({
            "type": "simulation_complete",
            "data": results
        }))

    async def run_game_loop(self):
        """运行游戏主循环"""
        try:
            # 游戏开始
            self.game.start_game()
            await self.broadcast("game_phase", {
                "phase": "start",
                "round": self.game.round_number
            })

            while self.game_running:
                # 夜晚阶段
                await asyncio.sleep(0.3)  # 动画延迟
                night_result = self.manager.run_night_phase()
                await self.broadcast("night_result", night_result)

                # 检查游戏是否结束
                if self.game.phase == Phase.GAME_OVER:
                    break

                # 白天阶段
                await asyncio.sleep(0.3)
                day_result = self.manager.run_day_phase(night_result.get("dead_players", []))
                await self.broadcast("day_result", day_result)

                if day_result.get("phase") == "game_over":
                    break

                # 检查游戏是否结束
                if self.game.phase == Phase.GAME_OVER or self.game.end_round():
                    break

                # 发送当前游戏状态
                await self.broadcast("game_state", self.game.game_state_dict())
                await self.broadcast("agent_summary", self.manager.get_agent_summary())

                # 记录历史
                self.game_history.append({
                    "round": self.game.round_number,
                    "night": night_result,
                    "day": day_result,
                    "timestamp": datetime.now().isoformat()
                })

                # 回合间隔
                await asyncio.sleep(0.5)

            # 游戏结束
            winner = self.game.check_win()
            game_summary = self.manager.get_game_summary()

            await self.broadcast("game_over", {
                "winner": winner.value if winner else "平局",
                "summary": game_summary,
                "history": self.game_history[-10:],  # 最后10回合
            })

            self.game_running = False

        except Exception as e:
            logger.error(f"游戏循环出错: {e}")
            await self.broadcast("game_error", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            self.game_running = False

    async def start_server(self):
        """启动WebSocket服务器 + HTTP静态文件服务器"""
        # 在独立线程中启动 HTTP 文件服务器
        import threading
        from http.server import HTTPServer, SimpleHTTPRequestHandler

        frontend_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "frontend"
        )

        class CustomHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=frontend_dir, **kwargs)

            def log_message(self, fmt, *args):
                pass  # 静默 HTTP 访问日志

        http_server = HTTPServer((self.host, 8080), CustomHandler)
        http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        http_thread.start()

        # 启动 WebSocket
        ws_server = await websockets.serve(self.handle_client, self.host, self.port)

        logger.info(f"WebSocket服务器启动在 ws://{self.host}:{self.port}")
        logger.info(f"前端页面: http://{self.host}:8080")
        logger.info("等待客户端连接...")

        # 保持服务器运行
        await asyncio.Future()  # 永久运行

    def get_server_info(self) -> dict:
        """获取服务器信息"""
        return {
            "host": self.host,
            "port": self.port,
            "clients": len(self.clients),
            "game_running": self.game_running,
            "subscriptions": {k: len(v) for k, v in self.subscriptions.items()},
        }


async def main():
    """主函数"""
    server = WerewolfWebSocketServer(host="localhost", port=8765)
    await server.start_server()


if __name__ == "__main__":
    # 运行WebSocket服务器
    asyncio.run(main())

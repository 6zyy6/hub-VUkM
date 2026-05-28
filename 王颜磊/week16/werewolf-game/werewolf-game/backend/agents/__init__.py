try:
    from .base_agent import BaseAgent, AgentFactory, AgentMessage, MessageType, AgentMemory
except ImportError:
    from base_agent import BaseAgent, AgentFactory, AgentMessage, MessageType, AgentMemory
try:
    from .werewolf_agent import WerewolfAgent
except ImportError:
    from werewolf_agent import WerewolfAgent
try:
    from .seer_agent import SeerAgent
except ImportError:
    from seer_agent import SeerAgent
try:
    from .witch_agent import WitchAgent
except ImportError:
    from witch_agent import WitchAgent
try:
    from .hunter_agent import HunterAgent
except ImportError:
    from hunter_agent import HunterAgent
try:
    from .guard_agent import GuardAgent
except ImportError:
    from guard_agent import GuardAgent
try:
    from .villager_agent import VillagerAgent
except ImportError:
    from villager_agent import VillagerAgent

__all__ = [
    "BaseAgent", "AgentFactory", "AgentMessage", "MessageType", "AgentMemory",
    "WerewolfAgent", "SeerAgent", "WitchAgent", "HunterAgent", "GuardAgent", "VillagerAgent",
]

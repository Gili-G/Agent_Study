# -*- coding: utf-8 -*-
"""三国狼人杀游戏工具函数"""
import asyncio
import random
from typing import List, Dict, Optional, Any
from collections import Counter

from agentscope.agent import AgentBase
from agentscope.message import Msg

# 游戏常量
MAX_GAME_ROUND = 10
MAX_DISCUSSION_ROUND = 3
CHINESE_NAMES = [
    "刘备", "关羽", "张飞", "诸葛亮", "赵云",
    "曹操", "司马懿", "典韦", "许褚", "夏侯惇", 
    "孙权", "周瑜", "陆逊", "甘宁", "太史慈",
    "吕布", "貂蝉", "董卓", "袁绍", "袁术"
]

def get_chinese_name(character: str = None) -> str:
    """获取中文角色名"""
    if character and character in CHINESE_NAMES:
        return character
    return random.choice(CHINESE_NAMES)


def format_player_list(players: List[AgentBase], show_roles: bool = False) -> str:
    """格式化玩家列表为中文显示"""
    if not players:
        return "无玩家"
    
    if show_roles:
        return "、".join([f"{p.name}({getattr(p, 'role', '未知')})" for p in players])
    else:
        return "、".join([p.name for p in players])
    
def majority_vote_cn(votes: Dict[str, str]) -> tuple[str, int]:
    """中文版多数投票统计"""
    if not votes:
        return "无人", 0
    
    vote_counts = Counter(votes.values())
    most_voted = vote_counts.most_common(1)[0]
    
    return most_voted[0], most_voted[1]

class GameModerator(AgentBase):
    """中文版游戏主持人"""
    
    def __init__(self) -> None:
        super().__init__()
        self.name = "游戏主持人"
        self.game_log: List[str] = []
    
    async def announce(self, content: str) -> Msg:
        """发布游戏公告"""
        msg = Msg(
            name=self.name,
            content=f"📢 {content}",
            role="system"
        )
        self.game_log.append(content)
        await self.print(msg)
        return msg
    
    async def night_announcement(self, round_num: int) -> Msg:
        """夜晚阶段公告"""
        content = f"🌙 第{round_num}夜降临，天黑请闭眼..."
        return await self.announce(content)
    
    async def day_announcement(self, round_num: int) -> Msg:
        """白天阶段公告"""
        content = f"☀️ 第{round_num}天天亮了，请大家睁眼..."
        return await self.announce(content)
    
    async def death_announcement(self, dead_players: List[str]) -> Msg:
        """死亡公告"""
        if not dead_players:
            content = "昨夜平安无事，无人死亡。"
        else:
            content = f"昨夜，{format_player_list_str(dead_players)}不幸遇害。"
        return await self.announce(content)
    
    async def vote_result_announcement(self, voted_out: str, vote_count: int) -> Msg:
        """投票结果公告"""
        content = f"投票结果：{voted_out}以{vote_count}票被淘汰出局。"
        return await self.announce(content)
    
    async def game_over_announcement(self, winner: str) -> Msg:
        """游戏结束公告"""
        content = f"🎉 游戏结束！{winner}"
        return await self.announce(content)
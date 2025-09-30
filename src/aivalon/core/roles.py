"""《阿瓦隆》角色信息与隐性知识工具。"""

from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass


class Role(str, Enum):
    """《阿瓦隆》中的角色定义。"""
    MERLIN = "Merlin"
    PERCIVAL = "Percival"
    ASSASSIN = "Assassin"
    MORGANA = "Morgana"
    OBERON = "Oberon"
    LOYAL_SERVANT = "Loyal Servant"


class Alignment(str, Enum):
    """角色阵营。"""
    GOOD = "Good"
    EVIL = "Evil"


@dataclass
class RoleInfo:
    """角色基础信息。"""
    role: Role
    alignment: Alignment
    team: str  # "Good" or "Evil"


# 角色基础配置
ROLE_DEFINITIONS: Dict[Role, RoleInfo] = {
    Role.MERLIN: RoleInfo(role=Role.MERLIN, alignment=Alignment.GOOD, team="Good"),
    Role.PERCIVAL: RoleInfo(role=Role.PERCIVAL, alignment=Alignment.GOOD, team="Good"),
    Role.ASSASSIN: RoleInfo(role=Role.ASSASSIN, alignment=Alignment.EVIL, team="Evil"),
    Role.MORGANA: RoleInfo(role=Role.MORGANA, alignment=Alignment.EVIL, team="Evil"),
    Role.OBERON: RoleInfo(role=Role.OBERON, alignment=Alignment.EVIL, team="Evil"),
    Role.LOYAL_SERVANT: RoleInfo(role=Role.LOYAL_SERVANT, alignment=Alignment.GOOD, team="Good"),
}


ROLE_NAMES_ZH: Dict[Role, str] = {
    Role.MERLIN: "梅林",
    Role.PERCIVAL: "帕西瓦尔",
    Role.ASSASSIN: "刺客",
    Role.MORGANA: "莫甘娜",
    Role.OBERON: "奥柏伦",
    Role.LOYAL_SERVANT: "忠诚侍从",
}


def _seat_label(seat: int) -> str:
    """将座位号格式化为中文描述。"""

    return f"{seat}号"


class RoleKnowledgeSystem:
    """为不同角色生成隐藏信息。"""
    
    def __init__(self):
        self.knowledge_map = {
            Role.MERLIN: self._merlin_knowledge,
            Role.PERCIVAL: self._percival_knowledge,
            Role.ASSASSIN: self._assassin_knowledge,
            Role.MORGANA: self._morgana_knowledge,
            Role.OBERON: self._oberon_knowledge,
            Role.LOYAL_SERVANT: self._loyal_servant_knowledge,
        }
    
    def get_role_card(self, *, seat: int, role: Role, all_seats: Dict[int, Role]) -> str:
        """为指定座位生成中文角色卡和隐性知识。

        Parameters
        ----------
        seat: int
            目标座位号。
        role: Role
            该座位对应的角色。
        all_seats: Dict[int, Role]
            全局角色分配，用于推断阵营信息。
        """
        if role in self.knowledge_map:
            knowledge = self.knowledge_map[role](seat=seat, all_seats=all_seats)
        else:
            role_name = ROLE_NAMES_ZH.get(role, role.value)
            knowledge = f"你的身份是{role_name}。"

        return (
            f"{knowledge}\n\n"
            "请尽量避免直接暴露自己的身份，通过提案、投票、执行任务和公开讨论去影响比赛局势。"
        )

    def _merlin_knowledge(self, *, seat: int, all_seats: Dict[int, Role]) -> str:
        """梅林：掌握邪恶阵营的具体座位（除了奥柏伦）。"""

        evil_seats = [s for s, r in all_seats.items() if r in [Role.ASSASSIN, Role.MORGANA, Role.OBERON]]
        evil_list = "、".join(_seat_label(s) for s in evil_seats) or "未知"

        has_oberon = any(r == Role.OBERON for r in all_seats.values())
        oberon_hint = "注意：如果游戏中有奥柏伦，你也能看到他。" if has_oberon else ""

        return (
            "你的身份是梅林。"
            f"你知道邪恶玩家坐在 {evil_list}，但不知道他们的具体角色。"
            f"{oberon_hint}"
            "务必隐瞒这一情报，引导善阵营成功完成任务，可以尝试找出派西瓦尔并隐秘地向他分享线索，同时避免被刺客识破。"
            "避免直接点出好人位置或坏人位置，也要避免组过于明显的好人车队，或是在投票上暴露自己的信息"
        )

    def _percival_knowledge(self, *, seat: int, all_seats: Dict[int, Role]) -> str:
        """帕西瓦尔：能看到梅林与莫甘娜但无法区分。"""

        special_seats = [s for s, r in all_seats.items() if r in [Role.MERLIN, Role.MORGANA]]
        special_list = "、".join(_seat_label(s) for s in special_seats) or "未知"
        return (
            "你的身份是派西瓦尔。"
            f"你能看到 {special_list} 之间有一位是真正的梅林，另一位是伪装者莫甘娜，但无法分辨。"
            "你需要在保护梅林的同时帮助善阵营，并适度伪装成梅林来干扰邪恶阵营的判断。"
        )

    def _assassin_knowledge(self, *, seat: int, all_seats: Dict[int, Role]) -> str:
        """刺客：掌握莫甘娜的位置（不包括奥柏伦）。"""

        morgana_seat = next((s for s, r in all_seats.items() if r == Role.MORGANA), None)
        oberon_seat = next((s for s, r in all_seats.items() if r == Role.OBERON), None)

        if morgana_seat is not None:
            oberon_hint = f"注意：如果游戏中还有奥柏伦在 {_seat_label(oberon_seat)}，他无法看到你们，你们也无法看到他。" if oberon_seat else ""
            return (
                "你的身份是刺客。"
                f"莫甘娜在 {_seat_label(morgana_seat)}。"
                f"{oberon_hint}"
                "只要善阵营先完成三次任务，你将获得一次刺杀机会，必须识破真正的梅林才能翻盘。"
                "若你与莫甘娜同时执行任务，通常由你来投任务失败，避免双坏票过于显眼。"
            )
        else:
            return "你的身份是刺客。按理应知晓莫甘娜的位置，但目前信息缺失。"

    def _morgana_knowledge(self, *, seat: int, all_seats: Dict[int, Role]) -> str:
        """莫甘娜：掌握刺客的位置（不包括奥柏伦）。"""

        assassin_seat = next((s for s, r in all_seats.items() if r == Role.ASSASSIN), None)
        oberon_seat = next((s for s, r in all_seats.items() if r == Role.OBERON), None)

        if assassin_seat is not None:
            oberon_hint = f"注意：如果游戏中还有奥柏伦在 {_seat_label(oberon_seat)}，他无法看到你们，你们也无法看到他。" if oberon_seat else ""
            return (
                "你的身份是莫甘娜。"
                f"刺客在 {_seat_label(assassin_seat)}。"
                f"{oberon_hint}"
                "在派西瓦尔眼中你与梅林无法区分，请利用这一点迷惑善阵营并积极破坏任务。你可以微妙地伪装梅林来迷惑派西瓦尔，但尽量避免自称梅林引起好人阵营的怀疑。"
                "与刺客同队时，通常由刺客投任务失败，你保持成功有助于掩护阵营。"
            )
        else:
            return "你的身份是莫甘娜。按理应知晓刺客的位置，但目前信息缺失。"

    def _oberon_knowledge(self, *, seat: int, all_seats: Dict[int, Role]) -> str:
        """奥柏伦：盲目的邪恶角色，无法看到其他邪恶成员，也不被其他邪恶成员看到。"""

        return (
            "你的身份是奥柏伦。"
            "你是邪恶阵营的成员，但你无法看到其他邪恶队友，其他邪恶队友也无法看到你。"
            "你的主要目标是混入任务队伍，然后让任务失败。"
            "你的次要目标是通过观察和推理，识别出其他邪恶队友并暗中协助他们，同时避免暴露自己的身份。"
            "你需要表现得像一个善良玩家来获得信任，但要在关键时刻破坏任务。"
        )

    def _loyal_servant_knowledge(self, *, seat: int, all_seats: Dict[int, Role]) -> str:
        """侍从：无额外情报。"""

        return (
            "你的身份是侍从。"
            "你没有任何额外情报，需要通过观察讨论与投票去判断阵营，支持善阵营并保护梅林。"
        )

    def get_evil_players(self, all_seats: Dict[int, Role]) -> List[int]:
        """返回所有邪恶阵营座位号。"""
        return [seat for seat, role in all_seats.items() if self.is_evil(role)]

    def get_good_players(self, all_seats: Dict[int, Role]) -> List[int]:
        """返回所有善良阵营座位号。"""
        return [seat for seat, role in all_seats.items() if self.is_good(role)]

    def is_evil(self, role: Role) -> bool:
        """判断角色是否属于邪恶阵营。"""
        return ROLE_DEFINITIONS[role].alignment == Alignment.EVIL

    def is_good(self, role: Role) -> bool:
        """判断角色是否属于善良阵营。"""
        return ROLE_DEFINITIONS[role].alignment == Alignment.GOOD
    
    def get_team_alignment(self, all_seats: Dict[int, Role]) -> Dict[str, List[int]]:
        """根据座位划分阵营，返回善/恶阵营座位列表。"""
        teams: Dict[str, List[int]] = {"Good": [], "Evil": []}
        for seat, role in all_seats.items():
            team_name = ROLE_DEFINITIONS[role].team
            teams[team_name].append(seat)
        return teams


# Global instance
ROLE_SYSTEM = RoleKnowledgeSystem()


def get_role_card(*, seat: int, role: Role, all_seats: Dict[int, Role]) -> str:
    """生成中文角色卡。"""
    return ROLE_SYSTEM.get_role_card(seat=seat, role=role, all_seats=all_seats)


def default_role_card_provider(state: Any, seat: int) -> str:
    """Convenience wrapper returning the role card for helpers that only have state."""

    all_seats = getattr(state, 'roles', {}) or {}
    role = all_seats.get(seat)
    if role is None:
        return "尚无角色信息可展示。"
    return get_role_card(seat=seat, role=role, all_seats=all_seats)


def is_evil_player(role: Role) -> bool:
    """判断角色是否属于邪恶阵营。"""
    return ROLE_SYSTEM.is_evil(role)


def is_good_player(role: Role) -> bool:
    """判断角色是否属于善良阵营。"""
    return ROLE_SYSTEM.is_good(role)


def get_evil_seats(all_seats: Dict[int, Role]) -> List[int]:
    """获取所有邪恶阵营座位号。"""
    return ROLE_SYSTEM.get_evil_players(all_seats)


def get_good_seats(all_seats: Dict[int, Role]) -> List[int]:
    """获取所有善良阵营座位号。"""
    return ROLE_SYSTEM.get_good_players(all_seats)


def validate_role_assignment(all_seats: Dict[int, Role]) -> bool:
    """校验五人局角色分配是否符合规则。"""
    # 检查是否恰好分配了 5 个座位
    if len(all_seats) != 5:
        return False
    
    # 统计角色数量
    role_counts: Dict[Role, int] = {}
    for role in all_seats.values():
        role_counts[role] = role_counts.get(role, 0) + 1
    
    # 五人局标准配置：1 梅林、1 帕西瓦尔、1 刺客、1 莫甘娜、1 忠诚侍从
    expected_roles = {
        Role.MERLIN: 1,
        Role.PERCIVAL: 1,
        Role.ASSASSIN: 1,
        Role.MORGANA: 1,
        Role.LOYAL_SERVANT: 1
    }
    
    for role, expected_count in expected_roles.items():
        actual_count = role_counts.get(role, 0)
        if actual_count != expected_count:
            return False
    
    return True

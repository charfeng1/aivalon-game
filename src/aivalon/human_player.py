"""Human player interface for interactive Avalon gameplay."""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.table import Table

from .core.schemas import (
    ProposalPayload, ProposeAction, VotePayload, VoteAction, VoteValue,
    MissionPayload, MissionAction, MissionOutcome, DiscussionPayload,
    EvilConclavePayload, AssassinGuessPayload, AssassinGuessAction,
    Phase
)
from .core.fsm import TOTAL_SEATS
from .core.roles import Role, default_role_card_provider

console = Console()

DEFAULT_HUMAN_SEAT = 3

class GameNarrator:
    """Provides human-friendly game progress updates."""

    def __init__(self, human_seats: Optional[Sequence[int]] = None, total_seats: int = TOTAL_SEATS):
        self.console = console
        self.round_num = 0
        self._human_seats = set(human_seats or [DEFAULT_HUMAN_SEAT])
        self.total_seats = total_seats

    def _is_human(self, seat: Optional[int]) -> bool:
        return seat is not None and seat in self._human_seats

    def divider(self, title: str = "") -> None:
        """Print a clear visual divider."""
        if title:
            self.console.print(f"\n{'='*60}")
            self.console.print(f"{title.center(60)}")
            self.console.print('='*60)
        else:
            self.console.print('-'*60)

    def start_round(self, round_num: int, leader_seat: int, leader_role: str = "Unknown") -> None:
        """Announce the start of a new round."""
        self.round_num = round_num
        self.divider(f"ROUND {round_num}")
        self.console.print(f"[bold blue]Leader:[/bold blue] Seat {leader_seat} ({leader_role})")
        self.console.print()

    def proposal_phase(self, leader_seat: int, members: List[int], phase: str) -> None:
        """Show team proposal."""
        phase_name = "INITIAL PROPOSAL" if phase == "PROPOSAL_DRAFT" else "FINAL PROPOSAL"
        self.console.print(f"[bold green]{phase_name}:[/bold green]")
        self.console.print(f"Seat {leader_seat} proposes team: {', '.join(f'Seat {m}' for m in members)}")
        self.console.print()

    def discussion_phase(self, speeches: List[Dict[str, Any]]) -> None:
        """Show discussion speeches."""
        self.console.print("[bold yellow]DISCUSSION PHASE:[/bold yellow]")
        for speech in speeches:
            seat = speech.get('seat')
            text = speech.get('speech', '')
            indicator = " (YOU)" if self._is_human(seat) else ""
            self.console.print(f"[dim]Seat {seat}{indicator}:[/dim] {text}")
        self.console.print()

    def vote_phase(self, votes: List[Dict[str, Any]], approved: bool) -> None:
        """Show voting results."""
        self.console.print("[bold magenta]VOTING RESULTS:[/bold magenta]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Seat", style="dim", width=12)
        table.add_column("Vote", width=10)

        approve_count = 0
        for vote in votes:
            seat = vote.get('seat')
            value = vote.get('value', 'UNKNOWN')
            if value == 'APPROVE':
                approve_count += 1
                vote_display = "[green]APPROVE[/green]"
            else:
                vote_display = "[red]REJECT[/red]"

            seat_display = f"Seat {seat}" + (" (YOU)" if self._is_human(seat) else "")
            table.add_row(seat_display, vote_display)

        self.console.print(table)

        result = "[green]APPROVED[/green]" if approved else "[red]REJECTED[/red]"
        self.console.print(f"\nProposal {result} ({approve_count}/{self.total_seats} votes)")
        self.console.print()

    def summary_phase(self, seat: int, speech: str) -> None:
        """Show leader summary before final proposal."""
        indicator = " (YOU)" if self._is_human(seat) else ""
        self.console.print("[bold cyan]LEADER SUMMARY:[/bold cyan]")
        self.console.print(f"[dim]Seat {seat}{indicator}:[/dim] {speech}")
        self.console.print()

    def mission_phase(self, mission_actions: List[Dict[str, Any]], fails: int, success: bool) -> None:
        """Show mission results."""
        self.console.print("[bold cyan]MISSION RESULTS:[/bold cyan]")

        # Show individual actions (only for transparency)
        for action in mission_actions:
            seat = action.get('seat')
            value = action.get('value', 'UNKNOWN')
            indicator = " (YOU)" if self._is_human(seat) else ""
            action_display = "[green]SUCCESS[/green]" if value == 'SUCCESS' else "[red]FAIL[/red]"
            self.console.print(f"[dim]Seat {seat}{indicator}:[/dim] {action_display}")

        result = "[green]SUCCESS[/green]" if success else "[red]FAILED[/red]"
        self.console.print(f"\nMission {result} ({fails} fail(s))")
        self.console.print()

    def evil_conclave(self, messages: List[Dict[str, Any]]) -> None:
        """Show evil conclave (if human is evil)."""
        self.console.print("[bold red]EVIL CONCLAVE:[/bold red]")
        for msg in messages:
            seat = msg.get('seat')
            speech = msg.get('speech', '')
            indicator = " (YOU)" if self._is_human(seat) else ""
            self.console.print(f"[dim]Seat {seat}{indicator}:[/dim] {speech}")
        self.console.print()

    def assassin_guess(self, assassin_seat: int, target: int, correct: bool) -> None:
        """Show assassin's guess."""
        self.console.print("[bold red]ASSASSIN'S GUESS:[/bold red]")
        self.console.print(f"Seat {assassin_seat} targets Seat {target}")
        result = "[green]CORRECT[/green]" if correct else "[red]INCORRECT[/red]"
        self.console.print(f"Guess was {result}")
        self.console.print()

    def game_end(self, winner: str, roles: Dict[int, Role]) -> None:
        """Show final game results."""
        self.divider("GAME OVER")

        winner_display = "[green]GOOD WINS[/green]" if winner in ['GOOD_WIN', 'GOOD'] else "[red]EVIL WINS[/red]"
        self.console.print(f"Result: {winner_display}")
        self.console.print()

        # Show role reveals
        self.console.print("[bold]ROLE REVEALS:[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Seat", style="dim", width=8)
        table.add_column("Role", width=15)

        for seat in sorted(roles.keys()):
            role = roles[seat]
            indicator = " (YOU)" if self._is_human(seat) else ""
            role_color = "green" if role.value in ['MERLIN', 'PERCIVAL', 'LOYAL_SERVANT'] else "red"
            table.add_row(f"Seat {seat}{indicator}", f"[{role_color}]{role.value}[/{role_color}]")

        self.console.print(table)

class HumanPlayerService:
    """Handles human player interactions for all game phases."""

    EVIL_ROLES: Sequence[Role] = (Role.ASSASSIN, Role.MORGANA)

    def __init__(self, narrator: GameNarrator, human_seats: Optional[Sequence[int]] = None):
        self.narrator = narrator
        self._human_seats: Tuple[int, ...] = tuple(sorted(human_seats or [DEFAULT_HUMAN_SEAT]))
        self._shown_knowledge: set[int] = set()

    @property
    def human_seats(self) -> Sequence[int]:
        return self._human_seats

    def is_human_seat(self, seat: int) -> bool:
        return seat in self._human_seats

    def _default_seat(self) -> int:
        return self._human_seats[0]

    def _get_role(self, state: Any, seat: int) -> Optional[Role]:
        return getattr(state, 'roles', {}).get(seat)

    def _get_role_name(self, state: Any, seat: int) -> str:
        role = self._get_role(state, seat)
        if role is None:
            return "Unknown"
        return getattr(role, 'value', str(role))

    def _show_game_state(self, state: Any, seat: int) -> None:
        console.print(f"[dim]Your seat: {seat} | Your role: {self._get_role_name(state, seat)}[/dim]")
        good_missions = getattr(state, 'good_missions', 0)
        evil_missions = getattr(state, 'evil_missions', 0)
        console.print(f"[dim]Score: Good {good_missions} - Evil {evil_missions}[/dim]")
        console.print()
        self._maybe_show_private_info(state, seat)

    def _maybe_show_private_info(self, state: Any, seat: int) -> None:
        if seat in self._shown_knowledge:
            return
        roles = getattr(state, 'roles', {}) or {}
        if seat not in roles:
            return
        try:
            knowledge = default_role_card_provider(state=state, seat=seat)
        except Exception:  # pragma: no cover - defensive safeguard
            knowledge = None
        if knowledge:
            console.print(Panel(knowledge.strip(), title="角色情报", style="cyan"))
            console.print()
        self._shown_knowledge.add(seat)

    def proposal(self, state: Any, phase: Phase, seat: Optional[int] = None) -> ProposalPayload:
        acting_seat = seat or self._default_seat()
        self._show_game_state(state, acting_seat)

        team_size = state.current_team_size()
        total_seats = self.narrator.total_seats
        console.print(f"[bold]You are the leader! Select {team_size} team members:[/bold]")
        console.print(f"Available seats: {', '.join(str(s) for s in range(1, total_seats + 1))}")

        members: List[int] = []
        for i in range(team_size):
            while True:
                try:
                    seat_input = typer.prompt(f"Select team member {i+1}")
                    choice = int(seat_input)
                    if choice < 1 or choice > total_seats:
                        console.print(f"[red]Invalid seat. Please choose 1-{total_seats}.[/red]")
                        continue
                    if choice in members:
                        console.print("[red]Seat already selected. Choose a different seat.[/red]")
                        continue
                    members.append(choice)
                    break
                except ValueError:
                    console.print("[red]Please enter a valid number.[/red]")

        members = sorted(set(members))
        return ProposalPayload(action=ProposeAction(members=members))

    def discussion(self, state: Any, seat: int) -> DiscussionPayload:
        self._show_game_state(state, seat)

        # Replay earlier speeches this round so the human sees context before speaking.
        thinking = ""

        console.print("[bold]公开发言（≤300字符）：[/bold]")
        speech = typer.prompt("你的发言", default="I have no specific comments at this time.")
        if len(speech) > 300:
            speech = speech[:297] + "..."
            console.print("[yellow]发言已截断至300字符。[/yellow]")
        return DiscussionPayload(thinking=thinking, speech=speech)

    def summary(self, state: Any, seat: int) -> DiscussionPayload:
        self._show_game_state(state, seat)
        thinking = ""

        console.print("[bold]请用公开口吻总结局势，并说明终稿阵容理由（≤300字符，勿泄露隐藏身份）：[/bold]")
        speech = typer.prompt("你的总结", default="我会根据大家的发言和投票整理出最稳妥的阵容。")
        if len(speech) > 300:
            speech = speech[:297] + "..."
            console.print("[yellow]总结已截断至300字符。[/yellow]")
        return DiscussionPayload(thinking=thinking, speech=speech)

    async def vote_async(self, state: Any, seat: int) -> VotePayload:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def get_vote() -> VotePayload:
            return self.vote(state, seat)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, get_vote)

    def vote_sync(self, state: Any, seat: int) -> VotePayload:
        return self.vote(state, seat)

    def vote(self, state: Any, seat: int) -> VotePayload:
        self._show_game_state(state, seat)
        proposal = getattr(state, 'current_proposal', None)
        if proposal:
            members = proposal.members
            console.print(f"[bold]Vote on team: {', '.join(f'Seat {m}' for m in members)}[/bold]")

        while True:
            vote_input = typer.prompt("Approve this team? (y/n)").lower().strip()
            if vote_input in ['y', 'yes']:
                vote_value = VoteValue.APPROVE
                break
            if vote_input in ['n', 'no']:
                vote_value = VoteValue.REJECT
                break
            console.print("[red]Please enter 'y' for yes or 'n' for no.[/red]")

        vote_display = "[green]APPROVE[/green]" if vote_value == VoteValue.APPROVE else "[red]REJECT[/red]"
        console.print(f"Your vote: {vote_display}")
        console.print("[dim]Waiting for other players to vote...[/dim]")

        return VotePayload(action=VoteAction(value=vote_value, reflection="Human player vote - no reflection provided"))

    def mission(self, state: Any, seat: int) -> MissionPayload:
        self._show_game_state(state, seat)
        role_enum = self._get_role(state, seat)
        role_name = role_enum.value if role_enum else "Unknown"
        is_evil = role_enum in self.EVIL_ROLES

        console.print(f"[bold]You are on the mission! (Role: {role_name})[/bold]")
        if is_evil:
            console.print("[red]As an evil player, you can choose to fail the mission.[/red]")
            while True:
                mission_input = typer.prompt("Make mission succeed? (y/n)").lower().strip()
                if mission_input in ['y', 'yes']:
                    outcome = MissionOutcome.SUCCESS
                    break
                if mission_input in ['n', 'no']:
                    outcome = MissionOutcome.FAIL
                    break
                console.print("[red]Please enter 'y' for success or 'n' for fail.[/red]")
        else:
            console.print("[green]As a good player, you must make the mission succeed.[/green]")
            console.print("[dim]Good players automatically make the mission succeed.[/dim]")
            outcome = MissionOutcome.SUCCESS

        action_display = "[green]SUCCESS[/green]" if outcome == MissionOutcome.SUCCESS else "[red]FAIL[/red]"
        console.print(f"Your action: {action_display}")
        console.print("[dim]Waiting for other mission members...[/dim]")

        return MissionPayload(action=MissionAction(value=outcome))

    def evil_conclave(self, state: Any, seat: int) -> EvilConclavePayload:
        self._show_game_state(state, seat)
        console.print("[bold red]邪恶密谈 - 私下协调[/bold red]")
        thinking = ""

        speech = typer.prompt("给邪恶队友的消息", default="No specific strategy to share.")
        if len(speech) > 300:
            speech = speech[:297] + "..."
            console.print("[yellow]密谈消息已截断至300字符。[/yellow]")
        return EvilConclavePayload(thinking=thinking, speech=speech)

    def assassin_guess(self, state: Any, seat: int) -> AssassinGuessPayload:
        self._show_game_state(state, seat)
        total_seats = self.narrator.total_seats
        console.print("[bold red]You are the Assassin! Guess who Merlin is:[/bold red]")
        console.print(f"Choose a seat (1-{total_seats}) to eliminate:")

        while True:
            try:
                target_input = typer.prompt("Target seat")
                target = int(target_input)
                if target < 1 or target > total_seats:
                    console.print(f"[red]Invalid seat. Please choose 1-{total_seats}.[/red]")
                    continue
                if target == seat:
                    confirm = typer.prompt("You're targeting yourself! Are you sure? (y/n)").lower().strip()
                    if confirm not in ['y', 'yes']:
                        continue
                break
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")

        return AssassinGuessPayload(action=AssassinGuessAction(target=target))
class HumanRecorder:
    """Custom recorder that provides live game updates for human players."""

    def __init__(self, base_recorder, narrator: GameNarrator, human_seats: Sequence[int], total_seats: int = TOTAL_SEATS):
        self.base_recorder = base_recorder
        self.narrator = narrator
        self._current_round = 0
        self._pending_votes = []
        self._pending_discussions = []
        self._pending_missions = []
        self._human_seats = set(human_seats)
        self.total_seats = total_seats

    def __getattr__(self, name):
        """Delegate all other methods to base recorder."""
        return getattr(self.base_recorder, name)

    def _is_human(self, seat: Optional[int]) -> bool:
        return seat in self._human_seats if seat is not None else False

    def record_proposal(self, *, round_num: int, phase: str, leader: int, payload: dict, usage: dict = None, reasoning: dict = None) -> None:
        """Record proposal and show to human."""
        self.base_recorder.record_proposal(
            round_num=round_num, phase=phase, leader=leader,
            payload=payload, usage=usage, reasoning=reasoning
        )

        # Start new round if needed
        if round_num != self._current_round:
            self._current_round = round_num
            self.narrator.start_round(round_num, leader)

        # Show proposal
        members = payload.get("action", {}).get("members", [])
        self.narrator.proposal_phase(leader, members, phase)

    def record_discussion(self, *, round_num: int, seat: int, payload: dict, reasoning: dict = None) -> None:
        """Record discussion and show immediately in turn order."""
        self.base_recorder.record_discussion(
            round_num=round_num, seat=seat, payload=payload, reasoning=reasoning
        )

        # Show each speech immediately as it happens
        speech = payload.get("speech", "")

        # If this is the first discussion of the round, show the header
        if not self._pending_discussions:
            self.narrator.console.print("[bold yellow]DISCUSSION PHASE:[/bold yellow]")

        # Show this seat's speech immediately
        indicator = " (YOU)" if self._is_human(seat) else ""
        self.narrator.console.print(f"[dim]Seat {seat}{indicator}:[/dim] {speech}")

        self._pending_discussions.append({"seat": seat, "speech": speech})

        # Clear the buffer when all have spoken (for potential future use)
        if len(self._pending_discussions) == self.total_seats:
            self.narrator.console.print()  # Add spacing after all discussions
            self._pending_discussions = []

    def record_summary(self, *, round_num: int, seat: int, payload: dict, reasoning: dict = None) -> None:
        """Record leader summary and show it immediately."""
        self.base_recorder.record_summary(
            round_num=round_num,
            seat=seat,
            payload=payload,
            reasoning=reasoning,
        )
        self.narrator.summary_phase(seat, payload.get("speech", ""))

    def record_vote(self, *, round_num: int, seat: int, value: str, reasoning: dict = None) -> None:
        """Record vote and show results immediately when complete."""
        self.base_recorder.record_vote(
            round_num=round_num, seat=seat, value=value, reasoning=reasoning
        )

        self._pending_votes.append({"seat": seat, "value": value})

        # Show vote count progress (but not individual votes for suspense)
        if len(self._pending_votes) < self.total_seats:
            self.narrator.console.print(f"[dim]Votes collected: {len(self._pending_votes)}/{self.total_seats}[/dim]")

        # Show results immediately when all votes are in
        if len(self._pending_votes) == self.total_seats:
            approve_count = sum(1 for v in self._pending_votes if v["value"] == "APPROVE")
            approved = approve_count >= (self.total_seats // 2 + 1)

            self.narrator.console.print("\n[bold magenta]VOTING RESULTS:[/bold magenta]")

            # Show results in a clean table
            from rich.table import Table
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Seat", style="dim", width=12)
            table.add_column("Vote", width=10)

            # Sort votes by seat number for consistent display
            sorted_votes = sorted(self._pending_votes, key=lambda v: v["seat"])
            for vote in sorted_votes:
                seat_num = vote["seat"]
                vote_val = vote["value"]

                vote_display = "[green]APPROVE[/green]" if vote_val == "APPROVE" else "[red]REJECT[/red]"
                seat_display = f"Seat {seat_num}" + (" (YOU)" if self._is_human(seat_num) else "")
                table.add_row(seat_display, vote_display)

            self.narrator.console.print(table)

            # Show final result
            result_color = "green" if approved else "red"
            result_text = "APPROVED" if approved else "REJECTED"
            self.narrator.console.print(f"\nProposal [{result_color}]{result_text}[/{result_color}] ({approve_count}/{self.total_seats} votes)\n")

            self._pending_votes = []

    def record_mission(self, *, round_num: int, seat: int, value: str, expected: int, reasoning: dict = None) -> None:
        """Record mission and show results immediately when complete."""
        self.base_recorder.record_mission(
            round_num=round_num, seat=seat, value=value, expected=expected, reasoning=reasoning
        )

        self._pending_missions.append({"seat": seat, "value": value})

        # Show progress (but not individual actions for suspense)
        if len(self._pending_missions) < expected:
            self.narrator.console.print(f"[dim]Mission actions: {len(self._pending_missions)}/{expected}[/dim]")

        # Show results immediately when all members have acted
        if len(self._pending_missions) == expected:
            fails = sum(1 for m in self._pending_missions if m["value"] == "FAIL")
            success = fails == 0

            self.narrator.console.print("\n[bold cyan]MISSION RESULTS:[/bold cyan]")

            # Only show final result - individual actions should remain secret
            result_color = "green" if success else "red"
            result_text = "SUCCESS" if success else "FAILED"
            self.narrator.console.print(f"Mission [{result_color}]{result_text}[/{result_color}] ({fails} fail(s))\n")

            self._pending_missions = []

    def record_conclave(self, *, round_num: int, seat: int, speech: str, expected: int, reasoning: dict = None) -> None:
        """Record conclave message."""
        self.base_recorder.record_conclave(
            round_num=round_num, seat=seat, speech=speech, expected=expected, reasoning=reasoning
        )

        # Show conclave messages (only visible to evil players)
        # For simplicity, we'll show them if the human is evil
        if self._is_human(seat):
            self.narrator.evil_conclave([{"seat": seat, "speech": speech}])

    def record_assassin_guess(self, *, seat: int, target: int, correct: bool, reasoning: dict = None) -> None:
        """Record assassin guess and show result."""
        self.base_recorder.record_assassin_guess(
            seat=seat, target=target, correct=correct, reasoning=reasoning
        )

        self.narrator.assassin_guess(seat, target, correct)

"""Seeded randomness helpers for deterministic simulations."""

import random
from typing import List


def build_rng(*, seed: int | None = None) -> random.Random:
    """Return a deterministic random generator."""
    return random.Random(seed)


def sample_team(rng: random.Random, *, leader: int, team_size: int, total_seats: int = 5) -> List[int]:
    """Sample a random team of the specified size (leader may or may not be included).

    Args:
        rng: Random number generator
        leader: The leader's seat number (not necessarily included in team)
        team_size: Number of players to select
        total_seats: Total number of seats in the game (default 5)

    Returns:
        Sorted list of seat numbers representing the team
    """
    # Available seats (1 to total_seats)
    available_seats = list(range(1, total_seats + 1))

    # Sample team members randomly (leader not required to be included)
    team = rng.sample(available_seats, team_size)
    team.sort()

    return team


def shuffle_seats(rng: random.Random, total_seats: int = 5) -> List[int]:
    """Return a shuffled list of seats.

    Args:
        rng: Random number generator
        total_seats: Number of seats to shuffle (default 5)

    Returns:
        Shuffled list of seat numbers from 1 to total_seats
    """
    seats = list(range(1, total_seats + 1))
    rng.shuffle(seats)
    return seats
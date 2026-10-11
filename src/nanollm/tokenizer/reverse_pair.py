from __future__ import annotations


class ReversePair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: ReversePair) -> bool:
        return self.pair > other.pair

from dataclasses import dataclass
from itertools import product
from functools import cached_property

import pandas as pd
from tqdm import tqdm


def get_nice_ratios():
    "Returns a set of nice ratios that are less than 1."
    rs = set()
    for divisor in [1, 2, 4, 5, 10]:
        for i in range(divisor):
            rs.add(i / divisor)
    return rs

    i = 1


class multiples_gen:
    "Generator of octets of integers that have a multiple relation."

    factor_set = [1, 3, 4, 9]

    def __init__(self, limit):
        self.limit = limit
        self.base = 1
        self.l = 0
        self.factors = product(self.factor_set, repeat=8)

    def __iter__(self):
        return self

    def __next__(self):
        if self.l > self.limit:
            raise StopIteration()

        try:
            new_factors = next(self.factors)
        except StopIteration:
            self.factors = product(self.factor_set, repeat=8)
            new_factors = next(self.factors)
            self.base += 1

        self.l += 1
        return [self.base * f for f in new_factors]


@dataclass
class Table:
    """Data container for experiment results"""

    Ax_pos: int
    Ax_neg: int
    Ay_pos: int
    Ay_neg: int
    Bx_pos: int
    Bx_neg: int
    By_pos: int
    By_neg: int

    nice_ratios = get_nice_ratios()

    @property
    def columns(self):
        """
        Column names for two events, two priors and two outcomes

        ["Ax_pos",  "Ax_neg",  "Ay_pos",  "Ay_neg",  "Bx_pos",  "Bx_neg",  "By_pos",  "By_neg"]
        """
        return list(
            f"{Ep}_{s}"
            for Ep, s in product(
                (f"{E}{p}" for E, p in product(["A", "B"], ["x", "y"])), ("pos", "neg")
            )
        )

    def __post_init__(self):
        self.Ax_sum = self.Ax_pos + self.Ax_neg
        self.Ay_sum = self.Ay_pos + self.Ay_neg
        self.A_pos = self.Ax_pos + self.Ay_pos
        self.A_sum = self.Ax_sum + self.Ay_sum

        self.Bx_sum = self.Bx_pos + self.Bx_neg
        self.By_sum = self.By_pos + self.By_neg
        self.B_pos = self.Bx_pos + self.By_pos
        self.B_sum = self.Bx_sum + self.By_sum

        self.Ax_rate = 0 if not self.Ax_sum else self.Ax_pos / self.Ax_sum
        self.Ay_rate = 0 if not self.Ay_sum else self.Ay_pos / self.Ay_sum

        self.Bx_rate = 0 if not self.Bx_sum else self.Bx_pos / self.Bx_sum
        self.By_rate = 0 if not self.By_sum else self.By_pos / self.By_sum

        self.A_rate = 0 if not self.A_sum else self.A_pos / self.A_sum
        self.B_rate = 0 if not self.B_sum else self.B_pos / self.B_sum

    def is_paradoxical(self):
        return all(
            [
                self.Ax_rate > self.Bx_rate,
                self.Ay_rate > self.By_rate,
                self.A_rate < self.B_rate,
            ]
        ) or all(
            [
                self.Ax_rate < self.Bx_rate,
                self.Ay_rate < self.By_rate,
                self.A_rate > self.B_rate,
            ]
        )

    def has_null(self):
        return 0 in [
            self.Ax_pos,
            self.Ay_pos,
            self.Ax_neg,
            self.Ay_neg,
            self.Bx_pos,
            self.By_pos,
            self.Bx_neg,
            self.By_neg,
        ]

    def as_df(self):
        df = pd.DataFrame(
            [
                [
                    self.Ax_pos,
                    self.Ax_sum,
                    self.Ax_rate,
                    self.Bx_pos,
                    self.Bx_sum,
                    self.Bx_rate,
                ],
                [
                    self.Ay_pos,
                    self.Ay_sum,
                    self.Ay_rate,
                    self.By_pos,
                    self.By_sum,
                    self.By_rate,
                ],
                [
                    self.A_pos,
                    self.A_sum,
                    self.A_rate,
                    self.B_pos,
                    self.B_sum,
                    self.B_rate,
                ],
            ],
            columns=["A_pos", "A_sum", "A_rate", "B_pos", "B_sum", "B_rate"],
            index=["x", "y", "all"],
        )
        return df

    def has_nice_ratios(self):
        return all(
            [
                rate in self.nice_ratios
                for rate in [
                    self.Ax_rate,
                    self.Ay_rate,
                    self.Bx_rate,
                    self.By_rate,
                    self.A_rate,
                    self.B_rate,
                ]
            ]
        )

    def has_ballanced_events(self):
        return A_sum == B_sum


def values_paradoxical(Ax_pos, Ax_neg, Ay_pos, Ay_neg, Bx_pos, Bx_neg, By_pos, By_neg):
    return all(
        [
            (0 if not (Ax_pos + Ax_neg) else Ax_pos / (Ax_pos + Ax_neg))
            > (0 if not (Bx_pos + Bx_neg) else Bx_pos / (Bx_pos + Bx_neg)),
            (0 if not (Ay_pos + Ay_neg) else Ay_pos / (Ay_pos + Ay_neg))
            > (0 if not (By_pos + By_neg) else By_pos / (By_pos + By_neg)),
            (
                0
                if not ((Ax_pos + Ax_neg) + (Ay_pos + Ay_neg))
                else (Ax_pos + Ay_pos) / ((Ax_pos + Ax_neg) + (Ay_pos + Ay_neg))
            )
            < (
                0
                if not ((Bx_pos + Bx_neg) + (By_pos + By_neg))
                else (Bx_pos + By_pos) / ((Bx_pos + Bx_neg) + (By_pos + By_neg))
            ),
        ]
    ) or all(
        [
            (0 if not (Ax_pos + Ax_neg) else Ax_pos / (Ax_pos + Ax_neg))
            < (0 if not (Bx_pos + Bx_neg) else Bx_pos / (Bx_pos + Bx_neg)),
            (0 if not (Ay_pos + Ay_neg) else Ay_pos / (Ay_pos + Ay_neg))
            < (0 if not (By_pos + By_neg) else By_pos / (By_pos + By_neg)),
            (
                0
                if not ((Ax_pos + Ax_neg) + (Ay_pos + Ay_neg))
                else (Ax_pos + Ay_pos) / ((Ax_pos + Ax_neg) + (Ay_pos + Ay_neg))
            )
            > (
                0
                if not ((Bx_pos + Bx_neg) + (By_pos + By_neg))
                else (Bx_pos + By_pos) / ((Bx_pos + Bx_neg) + (By_pos + By_neg))
            ),
        ]
    )


class SimpsonFinder:
    """
    Find pretty Simpson's paradox examples
    """

    def find_dc(self, rng, only_nice_ratios=False):
        """
        Use dataclass.
        """
        progress_bar = tqdm(total=rng**8)

        for values in product(range(rng), repeat=8):
            progress_bar.update()
            t = Table(*values)

            if (
                t.is_paradoxical()
                and not t.has_null()
                and (not only_nice_ratios or t.has_nice_ratios())
            ):
                return t.as_df()

    def find_rdc(self, rng, only_nice_ratios=False):
        """
        Recycle dataclass.
        """
        progress_bar = tqdm(total=rng**8)
        t = Table(*range(8))  # init class for recycling

        for values in product(range(rng), repeat=8):
            progress_bar.update()
            # recycle Table
            for val, atr in zip(values, t.__annotations__.keys()):
                setattr(t, atr, val)

            if (
                t.is_paradoxical()
                and not t.has_null()
                and (not only_nice_ratios or t.has_nice_ratios())
            ):
                return t.as_df()

    def find_f(self, rng, only_nice_ratios=False):
        """
        Check in func.
        """
        progress_bar = tqdm(total=rng**8)

        for values in product(range(rng), repeat=8):
            progress_bar.update()

            if not values_paradoxical(*values):
                continue

            t = Table(*values)

            if (
                t.is_paradoxical()
                and not t.has_null()
                and (not only_nice_ratios or t.has_nice_ratios())
            ):
                return t.as_df()

    def find_m(self, rng, only_nice_ratios=False):
        """
        Use only multiples.
        """
        progress_bar = tqdm(total=rng**8)
        self.last = None

        m_gen = multiples_gen(limit=rng**8)
        for values in m_gen:
            progress_bar.update()

            t = Table(*values)
            self.last = t

            if (
                t.is_paradoxical()
                and not t.has_null()
                and (not only_nice_ratios or t.has_nice_ratios())
            ):
                return t.as_df()


if __name__ == "__main__":
    sf = SimpsonFinder()
    solution = sf.find_m(rng=12, only_nice_ratios=False)
    print(solution)

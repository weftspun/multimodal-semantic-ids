"""MovieLens session dataset for Phase 1 parity.

Builds per-user chronological interaction **sessions**, drops sessions with < 3 items, applies a
leave-one-out temporal split (last item = test target, second-to-last = val target), and exposes
item **text** metadata (``"title | genres"``) for the ModernBERT encoder. Mirrors the eval framing
in decisions/session_recommendation_01.md.

Only the fast, dependency-light data layer lives here (pandas + stdlib download); the FSQ tokenizer
and Transformer that consume it land in the following Phase 1 actions.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ML_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
}

DEFAULT_DATA_DIR = Path("data")
MIN_SESSION_LEN = 3  # query-item + ground-truth needs >= 3 (decisions/session_recommendation_01.md)


@dataclass
class MovieLens:
    """A leave-one-out session split plus item text metadata.

    ``sessions_train`` are full training item-id sequences (autoregressive next-item targets are the
    positions within each). ``*_val`` / ``*_test`` are ``(history, target)`` pairs.
    """

    sessions_train: list[list[int]]
    sessions_val: list[tuple[list[int], int]]
    sessions_test: list[tuple[list[int], int]]
    item_text: dict[int, str]
    n_items: int
    n_users: int

    def stats(self) -> str:
        return (
            f"MovieLens: {self.n_users} users, {self.n_items} items, "
            f"{len(self.sessions_train)} train / {len(self.sessions_val)} val / "
            f"{len(self.sessions_test)} test sessions"
        )


def download(name: str = "ml-1m", data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Download+extract a MovieLens release if absent; return its extracted directory."""
    if name not in ML_URLS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(ML_URLS)}")
    data_dir = Path(data_dir)
    extracted = data_dir / name
    if extracted.exists():
        return extracted
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(ML_URLS[name], zip_path)  # noqa: S310 (trusted grouplens host)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return extracted


def _load_frames(name: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ratings[user,item,time], items[item,title,genres]) for either ML layout."""
    if name == "ml-1m":
        ratings = pd.read_csv(
            root / "ratings.dat", sep="::", engine="python",
            names=["user", "item", "rating", "time"], encoding="latin-1",
        )
        movies = pd.read_csv(
            root / "movies.dat", sep="::", engine="python",
            names=["item", "title", "genres"], encoding="latin-1",
        )
    else:  # ml-latest-small (CSV)
        ratings = pd.read_csv(root / "ratings.csv").rename(
            columns={"userId": "user", "movieId": "item", "timestamp": "time"}
        )
        movies = pd.read_csv(root / "movies.csv").rename(columns={"movieId": "item"})
    return ratings[["user", "item", "time"]], movies[["item", "title", "genres"]]


def load(name: str = "ml-1m", data_dir: Path = DEFAULT_DATA_DIR) -> MovieLens:
    """Load MovieLens into a leave-one-out session split with item text metadata."""
    root = download(name, data_dir)
    ratings, movies = _load_frames(name, root)

    # Chronological session per user; drop short sessions.
    ratings = ratings.sort_values(["user", "time"], kind="stable")
    grouped = ratings.groupby("user")["item"].apply(list)
    sessions = [seq for seq in grouped if len(seq) >= MIN_SESSION_LEN]

    train, val, test = [], [], []
    for seq in sessions:
        train.append(seq[:-2])              # autoregressive training portion
        val.append((seq[:-2], seq[-2]))     # predict second-to-last from its prefix
        test.append((seq[:-1], seq[-1]))    # predict last from its prefix

    item_text = {
        int(r.item): f"{r.title} | {str(r.genres).replace('|', ', ')}"
        for r in movies.itertuples(index=False)
    }
    n_items = int(ratings["item"].nunique())
    return MovieLens(train, val, test, item_text, n_items=n_items, n_users=len(sessions))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Load and summarize a MovieLens session split.")
    ap.add_argument("--dataset", default="ml-1m", choices=sorted(ML_URLS))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()

    ml = load(args.dataset, Path(args.data_dir))
    print(ml.stats())
    hist, target = ml.sessions_test[0]
    print(f"example test session: history[-5:]={hist[-5:]} -> target={target}")
    print(f"example item_text: {next(iter(ml.item_text.items()))}")


if __name__ == "__main__":
    main()

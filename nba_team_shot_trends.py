from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, shotchartdetail
from nba_api.stats.static import teams


ZONE_ORDER = [
    "Rim (0-3)",
    "Short mid (3-10)",
    "Long mid (10-22)",
    "Corner 3",
    "Above-break 3",
]


@dataclass(frozen=True)
class PullResult:
    shots_long: pd.DataFrame
    shots_wide: pd.DataFrame
    team_records: pd.DataFrame
    merged: pd.DataFrame
    failures: list[dict[str, str]]


def season_labels(start_year: int = 2010, end_year: int = 2024) -> list[str]:
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year + 1)]


def _team_lookup() -> pd.DataFrame:
    return (
        pd.DataFrame(teams.get_teams())[
            ["id", "full_name", "abbreviation", "nickname", "city", "state", "year_founded"]
        ]
        .rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME", "abbreviation": "TEAM_ABBREVIATION"})
        .sort_values(["TEAM_NAME"])
        .reset_index(drop=True)
    )


def _classify_shot_zones(shots: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
    shots = shots[shots["SHOT_ZONE_BASIC"] != "Backcourt"].copy()

    is_three = shots["SHOT_TYPE"].str.contains("3PT", na=False)
    shots["zone"] = np.select(
        [
            shots["SHOT_ZONE_BASIC"].isin(["Left Corner 3", "Right Corner 3"]),
            shots["SHOT_ZONE_BASIC"].eq("Above the Break 3"),
            (~is_three) & (shots["SHOT_DISTANCE"] <= 3),
            (~is_three) & (shots["SHOT_DISTANCE"] > 3) & (shots["SHOT_DISTANCE"] <= 10),
            (~is_three) & (shots["SHOT_DISTANCE"] > 10),
        ],
        [
            "Corner 3",
            "Above-break 3",
            "Rim (0-3)",
            "Short mid (3-10)",
            "Long mid (10-22)",
        ],
        default=pd.NA,
    )

    shots = shots.dropna(subset=["zone"]).copy()
    return shots


def _safe_pick(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return frame[column]
    raise KeyError(f"None of these columns were found: {candidates}")


def summarize_team_shots(raw_shots: pd.DataFrame, season: str) -> pd.DataFrame:
    shots = _classify_shot_zones(raw_shots)

    summary = (
        shots.groupby(["TEAM_ID", "TEAM_NAME", "zone"], as_index=False)
        .agg(
            FGA=("SHOT_ATTEMPTED_FLAG", "sum"),
            FGM=("SHOT_MADE_FLAG", "sum"),
        )
    )

    summary["FG_PCT"] = summary["FGM"] / summary["FGA"]
    summary["SEASON"] = season
    summary["TOTAL_FGA"] = summary.groupby(["TEAM_ID", "SEASON"])["FGA"].transform("sum")
    summary["FREQ"] = summary["FGA"] / summary["TOTAL_FGA"]
    summary["zone"] = pd.Categorical(summary["zone"], categories=ZONE_ORDER, ordered=True)

    return summary.sort_values(["SEASON", "TEAM_NAME", "zone"]).reset_index(drop=True)


def fetch_team_shot_data(
    seasons: Iterable[str] | None = None,
    pause_seconds: float = 0.7,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    seasons = list(seasons or season_labels())
    team_frame = _team_lookup()

    rows: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for season in seasons:
        print(f"Pulling shot chart data for {season}")
        for team in team_frame.itertuples(index=False):
            try:
                raw = shotchartdetail.ShotChartDetail(
                    team_id=int(team.TEAM_ID),
                    player_id=0,
                    season_nullable=season,
                    season_type_all_star="Regular Season",
                    context_measure_simple="FGA",
                ).get_data_frames()[0]
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "SEASON": season,
                        "TEAM_ID": str(team.TEAM_ID),
                        "TEAM_NAME": team.TEAM_NAME,
                        "ERROR": str(exc),
                    }
                )
                continue

            team_summary = summarize_team_shots(raw, season)
            rows.append(team_summary)
            time.sleep(pause_seconds)

    if not rows:
        raise RuntimeError("No shot chart data was returned.")

    combined = pd.concat(rows, ignore_index=True)
    return combined, failures


def fetch_team_records(seasons: Iterable[str] | None = None, pause_seconds: float = 0.7) -> pd.DataFrame:
    seasons = list(seasons or season_labels())
    team_meta = _team_lookup()[["TEAM_ID", "TEAM_ABBREVIATION"]]
    outputs: list[pd.DataFrame] = []

    for season in seasons:
        print(f"Pulling team record data for {season}")

        base = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
        ).get_data_frames()[0]

        advanced = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
        ).get_data_frames()[0]

        selected = pd.DataFrame(
            {
                "TEAM_ID": base["TEAM_ID"],
                "TEAM_NAME": base["TEAM_NAME"],
                "SEASON": season,
                "GP": _safe_pick(base, "GP"),
                "W": _safe_pick(base, "W"),
                "L": _safe_pick(base, "L"),
                "W_PCT": _safe_pick(base, "W_PCT"),
                "PTS": _safe_pick(base, "PTS"),
                "REB": _safe_pick(base, "REB"),
                "AST": _safe_pick(base, "AST"),
                "TOV": _safe_pick(base, "TOV"),
                "FG_PCT_TEAM": _safe_pick(base, "FG_PCT"),
                "FG3_PCT_TEAM": _safe_pick(base, "FG3_PCT"),
                "FGA_TEAM": _safe_pick(base, "FGA"),
                "FG3A_TEAM": _safe_pick(base, "FG3A"),
                "FTA_TEAM": _safe_pick(base, "FTA"),
                "OFF_RATING": _safe_pick(advanced, "OFF_RATING", "OFF_RTG"),
                "DEF_RATING": _safe_pick(advanced, "DEF_RATING", "DEF_RTG"),
                "NET_RATING": _safe_pick(advanced, "NET_RATING", "NET_RTG"),
                "PACE": _safe_pick(advanced, "PACE"),
                "TS_PCT": _safe_pick(advanced, "TS_PCT"),
                "PIE": _safe_pick(advanced, "PIE"),
            }
        )

        selected = selected.merge(team_meta, on="TEAM_ID", how="left")
        outputs.append(selected)
        time.sleep(pause_seconds)

    return pd.concat(outputs, ignore_index=True).sort_values(["SEASON", "TEAM_NAME"]).reset_index(drop=True)


def build_team_shot_features(shots_long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        shots_long.pivot_table(
            index=["SEASON", "TEAM_ID", "TEAM_NAME"],
            columns="zone",
            values=["FGA", "FGM", "FG_PCT", "FREQ"],
            fill_value=0,
        )
        .sort_index(axis=1)
        .reset_index()
    )

    wide.columns = [
        "_".join(str(part).strip().lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", ""))
        .strip("_")
        for part in wide.columns.to_flat_index()
    ]

    wide["three_share"] = wide["freq_above_break_3"] + wide["freq_corner_3"]
    wide["mid_share"] = wide["freq_long_mid_10_22"] + wide["freq_short_mid_3_10"]
    wide["rim_share"] = wide["freq_rim_0_3"]
    wide["three_to_rim_ratio"] = np.where(
        wide["rim_share"] > 0,
        wide["three_share"] / wide["rim_share"],
        np.nan,
    )

    return wide.sort_values(["SEASON", "TEAM_NAME"]).reset_index(drop=True)


def build_dataset(
    seasons: Iterable[str] | None = None,
    pause_seconds: float = 0.7,
) -> PullResult:
    shots_long, failures = fetch_team_shot_data(seasons=seasons, pause_seconds=pause_seconds)
    shots_wide = build_team_shot_features(shots_long)
    team_records = fetch_team_records(seasons=seasons, pause_seconds=pause_seconds)
    merged = shots_wide.merge(team_records, on=["SEASON", "TEAM_ID", "TEAM_NAME"], how="left")

    return PullResult(
        shots_long=shots_long,
        shots_wide=shots_wide,
        team_records=team_records,
        merged=merged,
        failures=failures,
    )


def save_dataset(
    output_dir: str | Path = ".",
    seasons: Iterable[str] | None = None,
    pause_seconds: float = 0.7,
) -> PullResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = build_dataset(seasons=seasons, pause_seconds=pause_seconds)

    result.shots_long.to_csv(output_path / "nba_team_zone_shots_long_2010_11_to_2024_25.csv", index=False)
    result.shots_wide.to_csv(output_path / "nba_team_zone_shots_wide_2010_11_to_2024_25.csv", index=False)
    result.team_records.to_csv(output_path / "nba_team_records_2010_11_to_2024_25.csv", index=False)
    result.merged.to_csv(output_path / "nba_team_shot_trends_2010_11_to_2024_25.csv", index=False)

    if result.failures:
        pd.DataFrame(result.failures).to_csv(output_path / "nba_team_shot_pull_failures.csv", index=False)

    return result


if __name__ == "__main__":
    result = save_dataset()
    print("Saved team shot trend datasets.")
    print(f"Shot-zone rows: {len(result.shots_long):,}")
    print(f"Team-season rows: {len(result.merged):,}")
    if result.failures:
        print(f"Failures logged: {len(result.failures):,}")

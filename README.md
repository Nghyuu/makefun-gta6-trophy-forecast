# makefun-gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nghyuu/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Nghyuu/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z, a week after launch, as five quantiles per trophy (5%, 25%, 50%, 75%, 95%). Built for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty.

## How it works

Exophase's page is a snapshot that only moves when Exophase refreshes it, every few hours. Between refreshes the tracked-player count is frozen and the tracked players slowly earn trophies; at a refresh a wave of newly tracked players arrives who hold far fewer trophies, so every rate drops. `main.ts` models the counts, not the rates:

- **Players tracked, N(t):** three growth curves (power law, saturating, straight line) fitted in log space to the case's history, median taken, stepped forward only at refresh times.
- **Holders, E_i(t):** between refreshes the holders grow at a per-trophy hazard lambda_i; at a refresh the new wave arrives holding trophy i at its cohort rate c_i. Lambda_i and c_i are fitted jointly from the refresh jumps the history shows, because the page is frozen between refreshes — a flat stretch says nothing about how fast players earn. With no refresh seen yet, the refresh is treated as overdue and the forecast dilutes toward the new wave's lower rates instead of repeating the last read.
- **The forecast** is the page's state at the last refresh expected before the target (a read shows the page as of its last refresh), with a spread from five scenarios varying the player-growth curve, the earning hazard and the refresh timing, plus a base term that grows with the horizon.

Validation: on the kit's practice cases the forecast scores about 0.18–0.20 pinball loss on the logit scale under plausible refreshed answers, against about 1.4 for the house persistence baseline; on a synthetic final-like case (two days of reads, 4.5-day horizon) it scores about 0.15 against 0.74 for the baseline. Node 22, standard library only, deterministic — two runs write the same bytes.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

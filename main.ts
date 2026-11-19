// gta6-trophy-rarity-forecast-01 entry.
//
// Forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the
// percent of the players Exophase tracks who will have earned it at the
// case's target, as five quantiles (5%, 25%, 50%, 75%, 95%).
//
// Model: Exophase's page is a snapshot that only moves when Exophase
// refreshes it, every few hours. Between refreshes the tracked-player count
// is frozen and the tracked players slowly earn trophies; at a refresh a
// wave of newly tracked players arrives who hold far fewer trophies, so
// every rate drops. The program therefore models the counts, not the
// rates:
//
//   - N(t), the players tracked, fitted from the case's history (three
//     growth curves in log space - power law, saturating, straight line -
//     and their median) and stepped forward only at refresh times.
//   - E_i(t), how many tracked players hold trophy i: between refreshes
//     the holders grow at a per-trophy hazard lambda_i fitted from the
//     drift of the history's flat stretches; at a refresh the new wave
//     arrives holding trophy i at its cohort rate c_i, fitted from the
//     jumps the history already shows.
//   - The forecast is the page's state at the last refresh expected
//     before the target (a read shows the page as of its last refresh,
//     not as of the read), with a spread from five scenarios that vary
//     the player-growth curve, the earning hazard and the refresh
//     timing, plus a base term that grows with the horizon.
//
// With a flat history (no refresh seen yet) the refresh is treated as
// overdue: the page has been unchanged since the first read, so the next
// refresh is due within the horizon and the forecast dilutes toward the
// new wave's lower rates instead of repeating the last read.
//
// Node 22, standard library only. Deterministic: no randomness, no clock,
// no network; two runs on the same case write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

/** Earning hazard (per day, among players who do not hold the trophy yet), prior by tier.
 *  Launch night documents how slow this is at first: between the last
 *  pre-refresh read and the refreshed page the 54 trophies other than
 *  the first-mission one gained only about a hundred holders between
 *  them, so the prior hazards are small; a history with refreshes
 *  replaces them with fitted values. */
const LAMBDA_PRIOR: Record<string, number> = { platinum: 0.004, gold: 0.02, silver: 0.03, bronze: 0.05 };
/** New-wave cohort rate as a fraction of the trophy's current rate, prior by tier.
 *  A wave of newly tracked players has barely played: on launch night the
 *  wave that arrived at the first refresh held almost nothing but the
 *  first-mission trophy (of 3,333 trophies the wave added, 3,230 were that
 *  one), so the prior is a small fraction of the current rate. Once the
 *  history shows real refreshes the fitted rates take over. */
const COHORT_FRACTION: Record<string, number> = { platinum: 0.02, gold: 0.03, silver: 0.04, bronze: 0.05 };
/** Shrinkage strength of the cohort-rate prior, in newly tracked players. */
const COHORT_PRIOR_WEIGHT = 300;
/** Refresh interval prior (hours) when the history shows no refresh yet. */
const GAP_PRIOR_HOURS = 4;
/** Base logit spread: 0.20 plus 0.10 per square-root day of horizon. */
const BASE_SPREAD = (horizonDays: number): number => 0.2 + 0.1 * Math.sqrt(horizonDays);

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (x: number): number => 100 / (1 + Math.exp(-x));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours since the Unix epoch, from an ISO 8601 timestamp. */
const hoursSinceEpoch = (iso: string): number => Date.parse(iso) / 3600000;
const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const std = (xs: number[]): number => {
  if (xs.length < 2) return 0;
  const mu = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - mu) * (b - mu), 0) / xs.length);
};
/** Least squares fit of y = a + b * x. */
const leastSquares = (xs: number[], ys: number[]): [number, number] => {
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return [my - b * mx, b];
};

interface FittedModel {
  /** Hazard per day for each trophy. */
  lambda: number[];
  /** Cohort rate (fraction of the new wave holding the trophy) for each trophy. */
  cohort: number[];
  /** Last read's rate for each trophy. */
  lastRate: number[];
  /** Expected hours from the last read to the next refresh. */
  firstRefreshIn: number;
  /** Expected hours between refreshes. */
  gap: number;
  /** Modelled players tracked at absolute time t (hours since the first read). */
  nAt: (t: number) => number;
  /** Log-space spread of the player-growth extrapolation at the target. */
  sigmaLogN: number;
  /** Last read's time and player count. */
  tLast: number;
  nLast: number;
}

function fitModel(c: Case): FittedModel {
  const nT = c.trophies.length;
  const firstAt = hoursSinceEpoch(c.history[0].at);
  const t = c.history.map((r) => hoursSinceEpoch(r.at) - firstAt);
  const n = c.history.map((r) => r.players_tracked);
  const k = c.history.length;
  const tLast = t[k - 1];
  const nLast = n[k - 1];
  // Earned counts per trophy and read (null where the read lacked the trophy).
  const e: Array<Array<number | null>> = c.history.map((r, ri) =>
    c.trophies.map((_, i) => (r.rates[i] === null ? null : (r.rates[i] as number) * n[ri] / 100)),
  );
  const lastRate: number[] = c.trophies.map((_, i) => {
    for (let ri = k - 1; ri >= 0; ri--) {
      if (e[ri][i] !== null) return c.history[ri].rates[i] as number;
    }
    return 0;
  });

  // Refresh jumps: the player count steps up between two reads. The page
  // is frozen between refreshes, so a flat stretch says nothing about how
  // fast players earn: every change in the history happens at a jump, and
  // each jump's earned-count rise mixes two things - the old cohort
  // earning during the gap since the last refresh, and the new wave
  // arriving already holding trophies. The hazard lambda_i and the cohort
  // rate c_i are therefore fitted JOINTLY from the jumps: for a candidate
  // lambda, the old cohort's earning in jump j is
  //   A_j = (N_j - E_ij) * (1 - exp(-lambda * gap_j / 24)),
  // the cohort rate is the least-squares c in dE_ij = A_j + c * dN_j,
  // and lambda is picked by the residual, ties broken toward the prior.
  // Times (hours since the first read) at which the player count stepped
  // up between two reads: the refreshes the history shows.
  const jumpTimes: number[] = [];
  for (let ri = 0; ri < k - 1; ri++) {
    if (n[ri + 1] > n[ri] * 1.005 + 1) jumpTimes.push(t[ri + 1]);
  }
  const LAM_GRID = [0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.25, 0.4, 0.7, 1.2, 2.0];

  const lambda: number[] = c.trophies.map((tr, i) => {
    const prior = LAMBDA_PRIOR[tr.tier] ?? 0.08;
    if (jumpTimes.length === 0) return prior;
    const rows: Array<{ gapH: number; nBefore: number; eBefore: number; dN: number; dE: number }> = [];
    let lt = t[0];
    for (let ri = 0; ri < k - 1; ri++) {
      if (n[ri + 1] > n[ri] * 1.005 + 1) {
        if (e[ri][i] !== null && e[ri + 1][i] !== null) {
          rows.push({
            gapH: t[ri + 1] - lt,
            nBefore: n[ri],
            eBefore: e[ri][i] as number,
            dN: n[ri + 1] - n[ri],
            dE: (e[ri + 1][i] as number) - (e[ri][i] as number),
          });
        }
        lt = t[ri + 1];
      }
    }
    if (rows.length === 0) return prior;
    let bestLam = prior;
    let bestRss = Infinity;
    for (const lam of LAM_GRID) {
      let num = 0;
      let den = 0;
      for (const r of rows) {
        const a = (r.nBefore - r.eBefore) * (1 - Math.exp(-lam * (r.gapH / 24)));
        num += r.dN * (r.dE - a);
        den += r.dN * r.dN;
      }
      const cc = den > 0 ? Math.max(0, num / den) : 0;
      let rss = 0;
      for (const r of rows) {
        const a = (r.nBefore - r.eBefore) * (1 - Math.exp(-lam * (r.gapH / 24)));
        rss += (r.dE - a - cc * r.dN) * (r.dE - a - cc * r.dN);
      }
      // Ties (within 5%) go to the lambda closest to the prior in log space.
      const better = rss < bestRss * 0.95;
      const tie =
        rss <= bestRss * 1.05 &&
        Math.abs(Math.log((lam + 0.002) / (prior + 0.002))) <
          Math.abs(Math.log((bestLam + 0.002) / (prior + 0.002)));
      if (better || tie) {
        bestRss = rss;
        bestLam = lam;
      }
    }
    return Math.min(2, Math.max(0, bestLam));
  });

  // Cohort rate c_i: what fraction of a newly tracked wave holds the trophy,
  // from the jumps after removing the old cohort's earning at lambda_i.
  const cohort: number[] = c.trophies.map((tr, i) => {
    const sticky = lastRate[i] >= 90;
    const prior = sticky ? 0.98 : (COHORT_FRACTION[tr.tier] ?? 0.04) * (lastRate[i] / 100);
    let num = 0;
    let den = 0;
    let lt = t[0];
    for (let ri = 0; ri < k - 1; ri++) {
      if (n[ri + 1] > n[ri] * 1.005 + 1) {
        if (e[ri][i] !== null && e[ri + 1][i] !== null) {
          const a =
            (n[ri] - (e[ri][i] as number)) * (1 - Math.exp(-lambda[i] * ((t[ri + 1] - lt) / 24)));
          num += (n[ri + 1] - n[ri]) * ((e[ri + 1][i] as number) - (e[ri][i] as number) - a);
          den += (n[ri + 1] - n[ri]) * (n[ri + 1] - n[ri]);
        }
        lt = t[ri + 1];
      }
    }
    const c = den > 0 ? num / den : prior;
    const shrunk = (den * Math.max(0, c) + COHORT_PRIOR_WEIGHT * prior) / (den + COHORT_PRIOR_WEIGHT);
    return Math.min(0.995, Math.max(0, shrunk));
  });

  // Player growth: three curves fitted in log space - a power law, a
  // saturating curve, and a straight line - and their median, so no
  // single shape owns the extrapolation. The spread is half the range
  // of the three at the target: model disagreement is the honest
  // uncertainty 4.5 days out.
  const nMin = Math.min(...n);
  const nMax = Math.max(...n);
  const tTarget = hoursSinceEpoch(c.target) - firstAt;
  let nAt: (t: number) => number;
  let sigmaLogN: number;
  if (nMax <= nMin * 1.02) {
    // No refresh seen yet: the page is stale since the first read. The
    // one refresh launch night documents multiplied the tracked base by
    // about 6 over 4.5 hours, so the prior catch-up curve is steep:
    // N(t) = N_first * (1 + t)^1.06 with t in hours since the first read,
    // capped at forty times the first read. It only ever fires on a flat
    // history; a history with any refresh uses the fitted curves.
    nAt = (tt: number) => Math.min(n[0] * 40, n[0] * Math.pow(1 + Math.max(0, tt), 1.06));
    sigmaLogN = 0.45;
  } else {
    const logN = n.map((v) => Math.log(v));
    const [aA, bA] = leastSquares(t.map((tt) => Math.log1p(tt / 24)), logN);
    const [aB, bB] = leastSquares(t.map((tt) => 1 - Math.exp(-tt / 72)), logN);
    const [aC, bC] = leastSquares(t, n);
    const logA = (tt: number): number => aA + bA * Math.log1p(tt / 24);
    const logB = (tt: number): number => aB + bB * (1 - Math.exp(-tt / 72));
    const logC = (tt: number): number => Math.log(Math.max(1, aC + bC * tt));
    nAt = (tt: number) => Math.exp(median([logA(tt), logB(tt), logC(tt)]));
    const logs = [logA(tTarget), logB(tTarget), logC(tTarget)];
    sigmaLogN = Math.min(0.8, (Math.max(...logs) - Math.min(...logs)) / 2 + 0.1);
  }

  // Refresh timing: the gap between observed refreshes, or the prior.
  // The next refresh is due after (gap - staleness): the page has been
  // unchanged since the last refresh, so a long flat stretch means one
  // is imminent.
  let gap = GAP_PRIOR_HOURS;
  if (jumpTimes.length >= 2) {
    const gaps: number[] = [];
    for (let j = 1; j < jumpTimes.length; j++) gaps.push(jumpTimes[j] - jumpTimes[j - 1]);
    gap = Math.min(12, Math.max(1, median(gaps)));
  }
  const lastJumpT = jumpTimes.length > 0 ? jumpTimes[jumpTimes.length - 1] : t[0];
  const staleness = Math.max(0, tLast - lastJumpT);
  const firstRefreshIn = Math.min(gap, Math.max(0.5, gap - staleness));

  return { lambda, cohort, lastRate, firstRefreshIn, gap, nAt, sigmaLogN, tLast, nLast };
}

interface Scenario {
  /** Multiplier on the modelled player count (log-space). */
  nScale: number;
  /** Multiplier on every earning hazard. */
  lambdaScale: number;
  /** Hours added to the first refresh time. */
  shift: number;
}

/**
 * Simulate the page forward to the target and return the rates the page
 * shows at its last refresh at or before the target (a read shows the
 * page as of its last refresh). Earning between refreshes follows the
 * closed form E' = N - (N - E) * exp(-lambda * dt).
 */
function simulatePage(c: Case, m: FittedModel, s: Scenario): number[] {
  const tTarget = hoursSinceEpoch(c.target) - hoursSinceEpoch(c.history[0].at);
  const nT = c.trophies.length;
  const eLast: number[] = c.trophies.map((_, i) => {
    for (let ri = c.history.length - 1; ri >= 0; ri--) {
      const v = c.history[ri].rates[i];
      if (v !== null) return (v as number) * c.history[ri].players_tracked / 100;
    }
    return 0;
  });
  let nCur = m.nLast;
  let eCur = [...eLast];
  let pageN = nCur;
  let pageE = [...eCur];

  // Refresh times: first due after firstRefreshIn, then every gap.
  const refreshes: number[] = [];
  let r = m.tLast + m.firstRefreshIn + s.shift;
  while (r <= tTarget) {
    refreshes.push(r);
    r += m.gap;
  }

  // Walk the segments between refreshes (and to the target).
  let tCur = m.tLast;
  const bounds = [...refreshes, tTarget];
  for (const b of bounds) {
    const dtDays = Math.max(0, b - tCur) / 24;
    if (dtDays > 0) {
      for (let i = 0; i < nT; i++) {
        const lam = m.lambda[i] * s.lambdaScale;
        eCur[i] = nCur - (nCur - eCur[i]) * Math.exp(-lam * dtDays);
      }
    }
    tCur = b;
    if (b < tTarget) {
      // A refresh: the tracked base jumps to the growth curve's value
      // and the new wave arrives holding each trophy at its cohort rate.
      const nNew = Math.max(nCur, m.nAt(b) * s.nScale);
      const dN = nNew - nCur;
      if (dN > 0) {
        for (let i = 0; i < nT; i++) eCur[i] += m.cohort[i] * dN;
        nCur = nNew;
      }
      pageN = nCur;
      pageE = [...eCur];
    }
  }
  return pageE.map((v, i) => (100 * v) / pageN);
}

function forecast(c: Case): Array<{ id: number; q: number[] }> {
  const m = fitModel(c);
  if (process.env.MAKEFUN_DEBUG === '1') {
    // Diagnostics to stderr only; the output files are unaffected.
    console.error(`case ${c.id}: reads=${c.history.length} jumps=${m.firstRefreshIn.toFixed(2)}h-first gap=${m.gap.toFixed(2)}h sigmaLogN=${m.sigmaLogN.toFixed(2)} Nlast=${m.nLast} Ntarget=${Math.round(m.nAt(hoursSinceEpoch(c.target) - hoursSinceEpoch(c.history[0].at)))}`);
    for (const i of [0, 2, 6, 13, 19, 34]) {
      console.error(`  trophy ${c.trophies[i].id} tier=${c.trophies[i].tier} last=${m.lastRate[i]} lambda=${m.lambda[i]} cohort=${m.cohort[i].toFixed(4)}`);
    }
  }
  const horizonDays = c.horizon_minutes / 1440;
  const base = BASE_SPREAD(horizonDays);
  const scenarios: Scenario[] = [
    { nScale: 1, lambdaScale: 1, shift: 0 },
    { nScale: Math.exp(m.sigmaLogN), lambdaScale: 1, shift: 0 },
    { nScale: Math.exp(-m.sigmaLogN), lambdaScale: 1, shift: 0 },
    { nScale: 1, lambdaScale: 1.8, shift: 0 },
    { nScale: 1, lambdaScale: 0.55, shift: m.gap / 2 },
  ];
  const runs = scenarios.map((s) => simulatePage(c, m, s));
  return c.trophies.map((tr, i) => {
    let med = runs[0][i];
    let spread = base;
    if (m.lastRate[i] >= 99.5) {
      // Held by every tracked player so far: a first-mission trophy the
      // new waves earn as fast as they arrive. It stays near the top,
      // with a spread wide enough to cover a dip if the tracked base
      // ever widens to players who have not played yet.
      med = 99.95;
      spread = 0.5;
    } else {
      const logits = runs.map((r) => logit(r[i]));
      spread = Math.min(2, Math.max(0.25, Math.sqrt(std(logits) * std(logits) + base * base)));
      med = clampRate(med);
    }
    const l = logit(med);
    const q = Z.map((z) => clampRate(round4(invLogitPct(l + z * spread))));
    // Enforce non-decreasing after rounding and clamping.
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: tr.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((lv, j) => lv !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const rows = forecast(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
}

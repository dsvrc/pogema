"""Per-step instrumentation for the contested-corridor NS.

Answers, from logged state rather than inference:
  Q1  Is K pinned at the clip floor, and from which sigma onward?
  Q2  Where do agents actually spend the episode -- home, queued, in corridor, done?
  Q3  Does demand collapse at high sigma (the auto-serialization escape)?
  Q4  At truncation, are unfinished agents STUCK AT THE MOUTH (capacity-bound) or
      still crossing (clock-bound)? This is the §3.1 question.

Usage:
    python trace_ns.py --sigma 1.0 --tier 3            # per-step table, one episode
    python trace_ns.py --diagnose                      # aggregate across sigma x tier
"""
import argparse

import numpy as np
from pogema import BatchAStarAgent

from contested_ns import make_contested_env


def rollout(sigma, tier, n=8, horizon=80, seed=0, trace=True):
    env = make_contested_env(num_agents=n, sigma=sigma, tier=tier,
                             horizon=horizon, trace=trace)
    agent = BatchAStarAgent()
    agent.reset_states()
    obs, info = env.reset(seed=seed)
    while True:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        if all(terminated) or all(truncated):
            break
    return env, info[0].get('metrics', {})


def show_steps(args):
    env, metrics = rollout(args.sigma, args.tier, args.n, args.horizon, args.seed)
    log = env.trace_log
    print(f'sigma={args.sigma} tier={args.tier} N={args.n} horizon={args.horizon} seed={args.seed}')
    print(f'ISR={metrics.get("ISR", 0):.3f}\n')
    print(f'{"t":>4} {"A":>5} {"K0":>5} {"K1":>5} {"K2":>5} {"dem":>10} '
          f'{"in":>3} {"que":>4} {"idle":>5} {"done":>5} {"thr":>4} '
          f'{"u_mn":>6} {"u_mx":>6} {"p_min":>6}')
    for r in log:
        dem = ','.join(f'{d:.0f}' for d in r['demand'])
        print(f'{r["t"]:>4} {r["A"]:>5.2f} {r["K"][0]:>5.2f} {r["K"][1]:>5.2f} '
              f'{r["K"][2]:>5.2f} {dem:>10} {r["n_inside"]:>3} {r["n_queued"]:>4} '
              f'{r["n_idle"]:>5} {r["n_done"]:>5} {r["throttled"]:>4} '
              f'{r["u_mean"]:>6.2f} {r["u_max"]:>6.2f} {r["p_min"]:>6.3f}')

    last = log[-1]
    print(f'\nAt episode end: done={last["n_done"]}/{args.n}  '
          f'inside={last["n_inside"]}  queued={last["n_queued"]}  idle={last["n_idle"]}')
    print(f'K at floor on {sum(r["k_at_floor"] for r in log)}/{len(log)} steps')


def diagnose(args):
    print(f'aggregate diagnosis  N={args.n}  horizon={args.horizon}  '
          f'episodes={args.episodes}\n')
    print(f'{"tier":>4} {"sigma":>6} {"ISR":>6} {"K_floor%":>9} {"K_mean":>7} '
          f'{"dem_mn":>7} {"dem_mx":>7} {"%queued":>8} {"%inside":>8} '
          f'{"end_que":>8} {"end_in":>7}')
    for tier in (2, 3):
        for sigma in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
            isr, kfloor, kmean, dmean, dmax = [], [], [], [], []
            pq, pi, endq, endi = [], [], [], []
            for seed in range(args.episodes):
                env, m = rollout(sigma, tier, args.n, args.horizon, seed)
                log = env.trace_log
                if not log:
                    continue
                isr.append(m.get('ISR', 0.0))
                kfloor.append(np.mean([r['k_at_floor'] for r in log]))
                kmean.append(np.mean([r['K'].mean() for r in log]))
                dmean.append(np.mean([r['demand'].mean() for r in log]))
                dmax.append(np.max([r['demand'].max() for r in log]))
                # where agents spend time, as a share of agent-steps
                tot = sum(r['n_inside'] + r['n_queued'] + r['n_idle'] for r in log) or 1
                pq.append(sum(r['n_queued'] for r in log) / tot)
                pi.append(sum(r['n_inside'] for r in log) / tot)
                endq.append(log[-1]['n_queued'])
                endi.append(log[-1]['n_inside'])
            print(f'{tier:>4} {sigma:>6.1f} {np.mean(isr):>6.3f} '
                  f'{np.mean(kfloor):>8.1%} {np.mean(kmean):>7.2f} '
                  f'{np.mean(dmean):>7.2f} {np.mean(dmax):>7.1f} '
                  f'{np.mean(pq):>7.1%} {np.mean(pi):>7.1%} '
                  f'{np.mean(endq):>8.2f} {np.mean(endi):>7.2f}')
    print()
    print('  Q1 K_floor%  -- if ~100% from the lowest sigma on, the dial is saturated')
    print('     and every severity is the same environment.')
    print('  Q3 dem_mn    -- if demand FALLS as sigma rises, agents are being pushed')
    print('     out of the medium and it is scheduling them (auto-serialization).')
    print('  Q4 end_que vs end_in -- unfinished agents piled at the MOUTH means')
    print('     capacity-bound (good, §3.1 holds); still INSIDE means clock-bound.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--tier', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--horizon', type=int, default=80)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--diagnose', action='store_true')
    args = p.parse_args()
    diagnose(args) if args.diagnose else show_steps(args)


if __name__ == '__main__':
    main()

"""Severity sweep + the certificates that do not need a trained learner.

Covers NS_design_guide.md §6.1 (N=1 byte-identity), §6.2 (frozen-partner
persistence), §3.4 (guaranteed collapse), §8.3 (the throttle must actually hurt)
and §8.10 (the metric must express a frontier, not a cliff).

Still owed and NOT covered here -- both need learners or a planner:
  §6.3 unilateral-fix, §6.4 oracle-driver gate, §6.5 scheduler-recovers-B0,
  §6.7 centralized ceiling, §6.8 sign test.

Usage:
    python run_ns_sweep.py                     # severity sweep, N=8, tier 3
    python run_ns_sweep.py --certificates      # N=1 and sigma=0 identity checks
    python run_ns_sweep.py --tier 2 --n 16
"""
import argparse
import statistics

from pogema import BatchAStarAgent

from contested_ns import make_contested_env

SEVERITIES = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def run(num_agents, sigma, tier, horizon, episodes):
    env = make_contested_env(num_agents=num_agents, sigma=sigma, tier=tier, horizon=horizon)
    agent = BatchAStarAgent()
    isr, csr, thr, umean, kmin, slack, eplen, solved, n_trunc = (
        [], [], [], [], [], [], [], [], 0)
    for seed in range(episodes):
        agent.reset_states()
        obs, info = env.reset(seed=seed)
        steps = 0
        while True:
            obs, reward, terminated, truncated, info = env.step(agent.act(obs))
            steps += 1
            if all(terminated) or all(truncated):
                break
        m = info[0].get('metrics', {})
        isr.append(m.get('ISR', 0.0))
        csr.append(m.get('CSR', 0.0))
        thr.append(m.get('ns_throttle_rate', 0.0))
        umean.append(m.get('ns_u_mean', 0.0))
        kmin.append(m.get('ns_K_min', float('nan')))
        slack.append(m.get('ns_slack', float('nan')))
        # POGEMA's own time metric (metrics.py:95), computed all along and ignored:
        # mean per-agent ARRIVAL TIME. Unlike ISR it sees delay directly, so the
        # horizon no longer has to bind for the NS to register.
        eplen.append(m.get('ep_length', float('nan')))
        if all(truncated):
            n_trunc += 1
        else:
            solved.append(steps)
    return dict(
        isr=statistics.mean(isr), csr=statistics.mean(csr),
        throttle=statistics.mean(thr), u=statistics.mean(umean),
        kmin=statistics.mean(kmin), slack=statistics.mean(slack),
        eplen=statistics.mean(eplen), trunc=n_trunc / episodes,
        makespan=statistics.mean(solved) if solved else float('nan'),
    )


def sweep(args):
    print(f'severity sweep  N={args.n}  tier={args.tier}  horizon={args.horizon}  '
          f'episodes={args.episodes}  policy=BatchAStarAgent')
    print(f'{"sigma":>6} {"TIME":>7} {"vs B0":>7} {"ISR":>7} {"CSR":>7} '
          f'{"throttle":>9} {"u_mean":>8} {"K_min":>7} {"slack":>7} {"trunc":>7}')
    b0, b0_time, isrs, usable = None, None, [], []
    for sigma in SEVERITIES:
        r = run(args.n, sigma, args.tier, args.horizon, args.episodes)
        if b0 is None:
            b0, b0_time = r['isr'], r['eplen']
        isrs.append(r['isr'])
        if r['slack'] >= 1.2:
            usable.append(sigma)
        flag = '' if r['slack'] >= 1.2 else '  <-- §3.1'
        ratio = r['eplen'] / b0_time if b0_time else float('nan')
        print(f'{sigma:>6.1f} {r["eplen"]:>7.1f} {ratio:>7.2f}x {r["isr"]:>6.3f} '
              f'{r["csr"]:>7.3f} {r["throttle"]:>9.3f} {r["u"]:>8.2f} '
              f'{r["kmin"]:>7.2f} {r["slack"]:>7.2f} {r["trunc"]:>7.1%}{flag}')
    monotone = all(a >= b - 0.02 for a, b in zip(isrs, isrs[1:]))
    print()
    print(f'  B0 (sigma=0) = {b0:.3f}   monotone: {"YES" if monotone else "NO -- severity is not well-ordered"}')
    print(f'  §6.6 slack >= 1.2 holds for sigma in {usable}')
    print('  Severities where slack < 1.2 are capacity-bound: no scheduler can reach')
    print('  B0 there, so they are OUTSIDE the class and must not be swept.')
    print('  §3.4 collapse: ISR must fall substantially and monotonically')
    print('  §8.10 frontier: it must DEGRADE GRACEFULLY, not cliff 1.0 -> 0.0')
    print('  §8.3 hurts: throttle rate > 0 while ISR still 1.0 means the NS is a no-op')


def certificates(args):
    print('§6.1 N=1 byte-identity -- l_i == 0 at N=1 for every sigma, so ISR must')
    print('     be identical to stationary POGEMA and throttle must be exactly 0.\n')
    print(f'{"sigma":>6} {"ISR":>7} {"throttle":>9} {"u_mean":>8}')
    ok_n1 = True
    for sigma in SEVERITIES:
        r = run(1, sigma, args.tier, args.horizon, 32)
        print(f'{sigma:>6.1f} {r["isr"]:>7.3f} {r["throttle"]:>9.3f} {r["u"]:>8.2f}')
        if r['throttle'] != 0.0 or r['isr'] < 0.999:
            ok_n1 = False
    print(f'  => {"PASS" if ok_n1 else "FAIL"}\n')

    print('sigma=0 identity -- the NS must be a no-op at severity 0 for every N.\n')
    print(f'{"N":>6} {"ISR":>7} {"throttle":>9} {"K_min":>8}')
    ok_s0 = True
    for n in (1, 4, 8, 16):
        r = run(n, 0.0, args.tier, args.horizon, 32)
        print(f'{n:>6} {r["isr"]:>7.3f} {r["throttle"]:>9.3f} {"-":>8}')
        if r['throttle'] != 0.0 or r['isr'] < 0.999:
            ok_s0 = False
    print(f'  => {"PASS" if ok_s0 else "FAIL"}')


def sweep_horizon(args):
    """POGEMA's reward is 1.0 on-goal and nothing else, so ISR is blind to delay
    until an agent misses the horizon. The NS can therefore only register once the
    horizon binds. Slack (§3.1/§6.6) is defined against the COORDINATED schedule --
    the horizon must fit a staggered crossing but not an unstaggered throttled one.
    This scan finds that window; sigma=0 is the coordinated-ish reference."""
    print(f'horizon scan  N={args.n}  tier={args.tier}  episodes={args.episodes}')
    print(f'{"horizon":>8} {"ISR@s=0":>9} {"ISR@s=3":>9} {"gap":>7} '
          f'{"mk@s=0":>8} {"mk@s=3":>8} {"slack":>7}')
    for horizon in (64, 80, 96, 112, 128):
        r0 = run(args.n, 0.0, args.tier, horizon, args.episodes)
        r3 = run(args.n, 3.0, args.tier, horizon, args.episodes)
        slack = horizon / r0['makespan'] if r0['makespan'] == r0['makespan'] else float('nan')
        print(f'{horizon:>8} {r0["isr"]:>9.3f} {r3["isr"]:>9.3f} '
              f'{r0["isr"] - r3["isr"]:>7.3f} {r0["makespan"]:>8.1f} '
              f'{r3["makespan"]:>8.1f} {slack:>7.2f}')
    print()
    print('  want: ISR@s=0 == 1.000 (B0 intact), ISR@s=3 well below it (§3.4),')
    print('        and slack >= 1.2 measured at sigma=0 (§6.6).')


def sweep_n(args):
    """Is the medium actually contested, or just under-subscribed?

    Measured at N=8: dem_mn 0.69 (mean demand per corridor BELOW one agent),
    %queued 9.5% at sigma=6. u = sigma*A*L_{-i}/K is exactly 0 whenever L_{-i}=0,
    so severity cannot bite on an empty corridor. If ISR collapses as N rises at
    fixed sigma, the medium was simply under-subscribed. If it does not, the load
    definition is wrong (it counts cell occupancy, not §7's REGIONAL occupancy --
    agents backed up in the home region heading for corridor c contribute nothing)."""
    print(f'team-size sweep  tier={args.tier}  horizon={args.horizon}  '
          f'episodes={args.episodes}')
    print(f'{"N":>4} {"sigma":>6} {"ISR":>7} {"throttle":>9} {"u_mean":>8} '
          f'{"trunc":>7} {"makespan":>9}')
    for n in (8, 16, 24, 32):
        for sigma in (0.0, 1.0, 3.0, 6.0):
            r = run(n, sigma, args.tier, args.horizon, args.episodes)
            print(f'{n:>4} {sigma:>6.1f} {r["isr"]:>7.3f} {r["throttle"]:>9.3f} '
                  f'{r["u"]:>8.2f} {r["trunc"]:>7.1%} {r["makespan"]:>9.1f}')
        print()
    print('  B0 must stay 1.000 at sigma=0 for EVERY N, or the map is capacity-bound')
    print('  and the METHOD §8 scaling experiment is confounded.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--tier', type=int, default=3, choices=[1, 2, 3])
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--episodes', type=int, default=160)
    p.add_argument('--certificates', action='store_true')
    p.add_argument('--sweep-horizon', action='store_true')
    p.add_argument('--sweep-n', action='store_true')
    args = p.parse_args()
    if args.certificates:
        certificates(args)
    elif args.sweep_horizon:
        sweep_horizon(args)
    elif args.sweep_n:
        sweep_n(args)
    else:
        sweep(args)


if __name__ == '__main__':
    main()

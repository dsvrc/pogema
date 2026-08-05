"""
POGEMA substrate for the contested-medium NS class (NS_design_guide.md §7, POGEMA row).

Setting: on_target='finish', corridor map, soft collisions, horizon slack.
NO non-stationarity is applied anywhere in this file. This is the severity-0
stationary reference: it measures B0 and runs the N=1 byte-identity certificate (§6.1).

Freeze the config only once B0 >= ~0.99 AND slack >= 1.2 (§6.6, §6.9).

Usage:
    python run_pogema_contested.py --n 8 --episodes 160          # B0
    python run_pogema_contested.py --n 1 --episodes 32           # N=1 certificate
    python run_pogema_contested.py --n 8 --corridor-width 2      # widen the medium
    python run_pogema_contested.py --n 8 --collision block_both  # reproduce the stall
"""
import argparse
import statistics

from pogema import BatchAStarAgent, GridConfig, pogema_v0


def build_map(left_w=6, mid_w=12, right_w=6, height=13, corridor_rows=(2, 6, 10), corridor_width=1):
    """Left home region -> N corridors -> right delivery region.

    Every agent must cross, so load Phi = corridor occupancy is task commitment,
    not a trigger an agent can quietly stop pulling (NS guide §8.9).
    @ = possible agent start, $ = possible target, # = obstacle.
    """
    open_rows = {r0 + k for r0 in corridor_rows for k in range(corridor_width)}
    return '\n'.join(
        '@' * left_w + ('.' if r in open_rows else '#') * mid_w + '$' * right_w
        for r in range(height)
    )


BASE_CONFIG = dict(
    on_target='finish',       # fixed total work -> scheduling-limited (§3.1)
    collision_system='soft',  # vertex+edge conflict free, but permits corridor flow
    observation_type='POMAPF',
    obs_radius=3,             # 7x7 view: K(t) stays genuinely latent
    integration=None,
)


def run_episode(env, agent, seed):
    agent.reset_states()
    obs, info = env.reset(seed=seed)
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        steps += 1
        if all(terminated):
            return info[0].get('metrics', {}), steps, False
        if all(truncated):
            return info[0].get('metrics', {}), steps, True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=8, help='num_agents (1 for the byte-identity certificate)')
    p.add_argument('--episodes', type=int, default=160)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--corridor-width', type=int, default=1)
    p.add_argument('--collision', default=BASE_CONFIG['collision_system'],
                   choices=['soft', 'block_both', 'priority'])
    args = p.parse_args()

    grid = build_map(corridor_width=args.corridor_width)
    cfg = GridConfig(**{**BASE_CONFIG,
                        'map': grid,
                        'num_agents': args.n,
                        'collision_system': args.collision,
                        'max_episode_steps': args.horizon})
    env = pogema_v0(grid_config=cfg)
    agent = BatchAStarAgent()

    isr, csr, solved_len, n_trunc = [], [], [], 0
    for seed in range(args.episodes):
        metrics, steps, truncated = run_episode(env, agent, seed)
        isr.append(metrics.get('ISR', 0.0))
        csr.append(metrics.get('CSR', 0.0))
        if truncated:
            n_trunc += 1
        else:
            solved_len.append(steps)

    # Slack proxy: horizon vs the worst makespan among episodes that actually finished.
    # Truncated episodes have no makespan -- they are the failure, not a long success.
    worst = max(solved_len) if solved_len else float('inf')

    print(f'N={args.n}  horizon={args.horizon}  corridor_width={args.corridor_width}  '
          f'collision={args.collision}  episodes={args.episodes}')
    print(f'  ISR (= B0 at severity 0) : {statistics.mean(isr):.3f}')
    print(f'  CSR                      : {statistics.mean(csr):.3f}')
    print(f'  truncated episodes       : {n_trunc}/{args.episodes} ({n_trunc / args.episodes:.1%})')
    if solved_len:
        print(f'  makespan solved (mean/max): {statistics.mean(solved_len):.1f} / {worst}')
        print(f'  slack ratio horizon/max  : {args.horizon / worst:.2f}   (want >= 1.2)')
    else:
        print('  makespan solved          : n/a -- every episode truncated')
    print()
    print('  FREEZE ONLY IF: ISR >= 0.99, truncated == 0, slack >= 1.2')


if __name__ == '__main__':
    main()

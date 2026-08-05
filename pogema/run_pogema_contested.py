"""
POGEMA substrate for the contested-medium NS class (NS_design_guide.md §7, POGEMA row).

Setting: on_target='finish', custom corridor map, block_both collisions, horizon slack.
This script establishes B0 (the stationary reference) and runs the N=1 byte-identity
certificate (§6.1). No NS is applied yet -- this is the frozen base config.

Usage:
    python run_pogema_contested.py            # B0 with A* baseline, 8 agents
    python run_pogema_contested.py --n 1      # N=1 certificate
    python run_pogema_contested.py --episodes 160
"""
import argparse
import statistics

from pogema import BatchAStarAgent, GridConfig, pogema_v0

# The medium: 3 unit-width corridors of length 12 joining a left home region to a
# right delivery region. Every agent must cross -> load Phi = corridor occupancy is
# task commitment, not an abandonable trigger (NS guide §8.9).
#   @ = possible agent start   $ = possible target   # = obstacle
CORRIDOR_MAP = """
@@@@@@############$$$$$$
@@@@@@############$$$$$$
@@@@@@............$$$$$$
@@@@@@############$$$$$$
@@@@@@############$$$$$$
@@@@@@############$$$$$$
@@@@@@............$$$$$$
@@@@@@############$$$$$$
@@@@@@############$$$$$$
@@@@@@############$$$$$$
@@@@@@............$$$$$$
@@@@@@############$$$$$$
@@@@@@############$$$$$$
"""

BASE_CONFIG = dict(
    map=CORRIDOR_MAP,
    on_target='finish',          # fixed total work -> scheduling-limited (§3.1)
    collision_system='block_both',  # no built-in priority ordering to pre-solve roles
    observation_type='POMAPF',
    obs_radius=3,                # 7x7 view: K(t) is genuinely latent
    max_episode_steps=128,       # the slack dial -- audit ratio must stay >= 1.2 (§6.6)
    integration=None,
)


def run_episode(env, agent, seed):
    agent.reset_states()
    obs, info = env.reset(seed=seed)
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(agent.act(obs))
        steps += 1
        if all(terminated) or all(truncated):
            break
    return info[0].get('metrics', {}), steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=8, help='num_agents (use 1 for the byte-identity certificate)')
    p.add_argument('--episodes', type=int, default=32)
    p.add_argument('--horizon', type=int, default=BASE_CONFIG['max_episode_steps'])
    args = p.parse_args()

    cfg = GridConfig(**{**BASE_CONFIG, 'num_agents': args.n, 'max_episode_steps': args.horizon})
    env = pogema_v0(grid_config=cfg)
    agent = BatchAStarAgent()

    isr, csr, makespan = [], [], []
    for seed in range(args.episodes):
        metrics, steps = run_episode(env, agent, seed)
        isr.append(metrics.get('ISR', 0.0))
        csr.append(metrics.get('CSR', 0.0))
        makespan.append(steps)

    print(f'N={args.n}  horizon={args.horizon}  episodes={args.episodes}')
    print(f'  ISR (= B0 at severity 0) : {statistics.mean(isr):.3f}')
    print(f'  CSR                      : {statistics.mean(csr):.3f}')
    print(f'  makespan (mean / max)    : {statistics.mean(makespan):.1f} / {max(makespan)}')
    print(f'  slack ratio horizon/makespan_max : {args.horizon / max(makespan):.2f}  (want >= 1.2)')


if __name__ == '__main__':
    main()

"""TOLL-lambda on the frozen contested-corridor NS (METHOD_design.md §4), plus the
AIMD baseline that §9.4 says must be beaten before the method claims anything.

WHAT IS AND IS NOT HERE
  TOLL-lambda (§4, Component 2)  -- implemented, runnable, no learner needed.
  AIMD proportional backoff (§9.4) -- implemented. The bar TOLL must clear.
  TOLL-G (§3, Component 1)      -- NOT here. It is an advantage correction and needs
                                   an actor-critic host (MAPPO/HAPPO), which this repo
                                   does not have. POGEMA is discrete, so it also needs
                                   the SCORE-FUNCTION form, not d(Phi)/da (§9.6).
                                   See externality_score_function() below: the term is
                                   implemented and finite-difference checkable, but it
                                   is not wired to a learner.

WHY ADMISSION IS THE DECISION VARIABLE
Under admission-control throttling, attempting entry costs an agent NOTHING privately
-- denied means wait, which is what it would have done anyway. But the attempt adds 1
to L_c, raising u for everyone else and damaging K_c. Zero private cost, positive
social cost: that IS the externality (§2.2), and it is what lambda prices. An agent
that voluntarily waits contributes claim=None and leaves the demand pool entirely, so
the lever is real rather than nominal.

DECENTRALIZATION (what each agent may use)
  allowed: static map geometry, own position/target, own felt blocking history
  FORBIDDEN: L, K, A, sigma, anyone else's state or price. No messages.
The medium itself is the only channel (§4).

FELT LIABILITY
l_hat_{i,c} = EWMA of "I commanded entry at corridor c and did not move". The agent
cannot tell a throttle from a collision -- that ambiguity is realistic and is logged
as block_ambiguity so it can be audited.
"""
import argparse
import statistics
from collections import defaultdict

import numpy as np
from pogema import BatchAStarAgent

from contested_ns import CORRIDOR_ROWS, LEFT_W, MID_W, corridor_of, make_contested_env

WAIT, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4


# ---------------------------------------------------------------- navigation

class Navigator:
    """Shared by every arm. Routes left region -> chosen corridor -> target.

    All arms use THIS navigator so the only difference between them is the
    admission decision. Otherwise a routing difference would be confounded with
    the pricing mechanism.
    """

    @staticmethod
    def region(r, c):
        if c < LEFT_W:
            return 'home'
        if c < LEFT_W + MID_W:
            return 'corridor'
        return 'delivery'

    @staticmethod
    def at_mouth(r, c, corridor_idx):
        return c == LEFT_W - 1 and r == CORRIDOR_ROWS[corridor_idx]

    @staticmethod
    def detour(r, corridor_idx):
        return abs(r - CORRIDOR_ROWS[corridor_idx])

    @staticmethod
    def step_toward_mouth(r, c, corridor_idx):
        target_row = CORRIDOR_ROWS[corridor_idx]
        if r != target_row:
            return DOWN if r < target_row else UP
        if c < LEFT_W - 1:
            return RIGHT
        return None  # at the mouth: caller decides admission

    @staticmethod
    def step_in_delivery(r, c, tr, tc):
        if c < tc:
            return RIGHT
        if r < tr:
            return DOWN
        if r > tr:
            return UP
        return WAIT


# ---------------------------------------------------------------- arms

class BaseArm:
    name = 'base'

    def __init__(self, n_agents, n_corridors=len(CORRIDOR_ROWS), seed=0):
        self.n = n_agents
        self.nc = n_corridors
        self.rng = np.random.default_rng(seed)
        # PERSISTENT learned state -- lambda is a learned quantity like a policy
        # parameter and must survive across episodes. Resetting it each episode gave
        # only ~2.4 dual updates per agent per episode, so the price never converged
        # (measured: l_hat 0.053 against a block rate of 0.73).
        self.l_hat = np.zeros((self.n, self.nc))
        self.lam = np.zeros((self.n, self.nc))
        self.reset_episode()

    def reset_episode(self):
        """Episode-local bookkeeping ONLY. Never touches l_hat / lam."""
        self.nav = BatchAStarAgent()
        self.attempts = np.zeros((self.n, self.nc))
        self.blocks = np.zeros((self.n, self.nc))
        self.admits = np.zeros((self.n, self.nc))
        self.voluntary_waits = np.zeros(self.n)
        self.chosen = np.full(self.n, -1)
        self.last_xy = [None] * self.n
        self.tried_entry = [None] * self.n

    # -- per-arm hooks ------------------------------------------------

    def pick_corridor(self, i, r):
        return int(np.argmin([Navigator.detour(r, k) for k in range(self.nc)]))

    def admit(self, i, k):
        return True

    def on_block(self, i, k):
        pass

    def on_admit(self, i, k):
        pass

    def on_idle(self, i, k):
        """Agent is at the mouth but chose not to attempt. Arms that price on felt
        degradation MUST decay here, or refusing becomes absorbing: refuse -> no
        observation -> l_hat never falls -> refuse forever."""
        pass

    # -- driver -------------------------------------------------------

    def observe(self, i, xy):
        """Detect whether last step's entry attempt was served. Purely local."""
        k = self.tried_entry[i]
        if k is None:
            return
        moved = self.last_xy[i] is not None and tuple(xy) != tuple(self.last_xy[i])
        if moved:
            self.admits[i, k] += 1
            self.on_admit(i, k)
        else:
            self.blocks[i, k] += 1
            self.on_block(i, k)
        self.tried_entry[i] = None

    def act_all(self, obs, xys, active, moves):
        """Navigation is delegated to stock BatchAStarAgent; the arm overrides ONLY
        the admission decision.

        Hand-rolled routing failed the gate at ISR 0.684 where BatchAStarAgent scores
        1.000 -- agents converging on a corridor row deadlocked under 'soft'
        collisions, and blind ISR then ROSE with severity because the throttle was
        relieving that deadlock. Delegating navigation removes the confound entirely
        and isolates the method to the one decision it is about."""
        actions = list(self.nav.act(obs))

        for i in range(self.n):
            if not active[i]:
                continue
            self.observe(i, xys[i])
            r, c = xys[i]
            if corridor_of(r, c) is not None:
                self.last_xy[i] = xys[i]
                continue  # in service: never interfere with draining

            dr, dc = moves[actions[i]]
            k = corridor_of(r + dr, c + dc)
            self.last_xy[i] = xys[i]
            if k is None:
                continue  # not an entry attempt; pass A* straight through

            self.chosen[i] = k
            if self.admit(i, k):
                self.attempts[i, k] += 1
                self.tried_entry[i] = k
            else:
                self.voluntary_waits[i] += 1
                self.on_idle(i, k)
                actions[i] = WAIT
                # A* returns a RANDOM action when it sees itself not move
                # (a_star_policy.py:112). Clearing its saved position stops our
                # deliberate hold from being mistaken for being stuck.
                self.nav.astar_agents[i]._saved_xy = None
        return actions


class BlindArm(BaseArm):
    """Vanilla: nearest corridor, always attempt. The behaviour §2.2 predicts a
    gradient follower converges to -- individually rational, collectively fatal."""
    name = 'blind'


class AIMDArm(BaseArm):
    """§9.4 -- congestion control in four lines, no learning. If TOLL cannot clearly
    beat this, the method contribution collapses regardless of the class."""
    name = 'aimd'
    ALPHA = 0.08     # additive increase on success
    BETA = 0.55      # multiplicative decrease on block
    P_MIN = 0.02

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.p = np.ones((self.n, self.nc))  # persistent, like lambda

    def admit(self, i, k):
        return self.rng.random() < self.p[i, k]

    def on_admit(self, i, k):
        self.p[i, k] = min(1.0, self.p[i, k] + self.ALPHA)

    def on_block(self, i, k):
        self.p[i, k] = max(self.P_MIN, self.p[i, k] * self.BETA)


class TollLambdaArm(BaseArm):
    """METHOD_design §4. Per-corridor local price from felt degradation.

      l_hat_{i,c} <- (1-rho) l_hat + rho * blocked          (measured, local)
      lam_{i,c}   <- max(0, lam + alpha*(l_hat - l_target))  (dual ascent, §4)
      admit iff   (1 - l_hat_{i,c})  -  lam_{i,c}  >  0      (max return - lam*D)
      route to    argmin_c [ detour_c + w * lam_{i,c} ]      (price-aware allocation)

    The admit rule is §4's `max_a [immediate return] - lambda(t) * D(a)` with
    D=1 for an entry attempt and the immediate return estimated by the agent's own
    served-rate (1 - l_hat). Waiting has return 0 and demand 0, so the comparison
    reduces to the inequality above.

    Per-corridor lambda (rather than one scalar) is the POGEMA adaptation: the medium
    here is three corridors, so the allocation question is WHICH as well as WHEN.
    A single scalar price can only answer WHEN, and measured on this NS the throttle
    already answers WHEN by itself.
    """
    name = 'toll'
    RHO = 0.15        # EWMA rate for felt liability
    ALPHA = 0.10      # dual ascent step
    L_TARGET = 0.25   # target felt-degradation level
    W_ROUTE = 3.0     # price weight in corridor choice
    LAM_MAX = 3.0

    def _update_price(self, i, k, blocked):
        self.l_hat[i, k] = (1 - self.RHO) * self.l_hat[i, k] + self.RHO * float(blocked)
        self.lam[i, k] = float(np.clip(
            self.lam[i, k] + self.ALPHA * (self.l_hat[i, k] - self.L_TARGET),
            0.0, self.LAM_MAX))

    EPS_PROBE = 0.10  # persistent excitation: never stop sampling the medium

    def on_block(self, i, k):
        self._update_price(i, k, True)

    def on_admit(self, i, k):
        self._update_price(i, k, False)

    def on_idle(self, i, k):
        # Demand fell because I yielded, so my estimate of congestion must decay --
        # otherwise refusing is absorbing and the agent excludes itself permanently.
        self._update_price(i, k, False)

    # NOTE: price-aware ROUTING (argmin_c [detour_c + w*lam_c]) is DEFERRED.
    # Navigation is now A*, which picks the shortest path and cannot be steered by a
    # price without a custom router. What remains is admission-only TOLL: the price
    # decides WHEN to enter, not WHICH corridor. That is faithful to §4's decision
    # rule but exercises only half the allocation problem this NS poses.

    def admit(self, i, k):
        if self.rng.random() < self.EPS_PROBE:
            return True
        return (1.0 - self.l_hat[i, k]) - self.lam[i, k] > 0.0


ARMS = {a.name: a for a in (BlindArm, AIMDArm, TollLambdaArm)}


# ---------------------------------------------------------------- TOLL-G stub

def externality_score_function(log_probs, coupling_term):
    """TOLL-G in SCORE-FUNCTION form -- the discrete-action version §9.6 demands.

    d(Phi)/da does not exist for a Discrete(5) action space, so the pathwise form in
    §3 is unavailable. The score-function identity replaces it:

        grad_i E[ sum_j V_j ] = E[ (sum_{j!=i} dV_j/dl_j * dl_j/dPhi_i) * Phi_i
                                   * grad log pi_i(a_i) ]

    i.e. the coupling term multiplies the score instead of an action derivative. The
    correction enters the advantage as  A_hat_i = A_i^GAE - eta * coupling_term.

    NOT WIRED TO A LEARNER. Verify T1 by finite differences against a host's true
    gradient before trusting it (§9.6).
    """
    return log_probs * coupling_term


# ---------------------------------------------------------------- rollout

def run_arm(arm_cls, sigma, n=16, tier=3, horizon=80, episodes=160, verbose=False):
    env = make_contested_env(num_agents=n, sigma=sigma, tier=tier, horizon=horizon)
    arm = arm_cls(n)
    stats = defaultdict(list)

    for seed in range(episodes):
        arm.reset_episode()  # lambda / l_hat PERSIST across episodes
        obs, info = env.reset(seed=seed)
        steps = 0
        moves = env.grid_config.MOVES
        while True:
            xys = env.get_agents_xy(ignore_borders=True)
            active = [env.grid.is_active[i] for i in range(n)]
            act = arm.act_all(obs, xys, active, moves)
            obs, reward, terminated, truncated, info = env.step(act)
            steps += 1
            if verbose and seed == 0:
                print(f'  t={steps:>3} lam_mean={arm.lam.mean():.3f} '
                      f'l_hat_mean={arm.l_hat.mean():.3f} '
                      f'vol_wait={arm.voluntary_waits.sum():.0f} '
                      f'attempts={arm.attempts.sum():.0f} blocks={arm.blocks.sum():.0f}')
            if all(terminated) or all(truncated):
                break

        m = info[0].get('metrics', {})
        att, blk = arm.attempts.sum(), arm.blocks.sum()
        usage = arm.attempts.sum(axis=0)
        share = usage / max(usage.sum(), 1)
        stats['isr'].append(m.get('ISR', 0.0))
        stats['csr'].append(m.get('CSR', 0.0))
        stats['slack'].append(m.get('ns_slack', float('nan')))
        stats['throttle'].append(m.get('ns_throttle_rate', 0.0))
        stats['lam'].append(float(arm.lam.mean()))
        stats['lam_max'].append(float(arm.lam.max()))
        stats['l_hat'].append(float(arm.l_hat.mean()))
        stats['attempts'].append(float(att))
        stats['block_rate'].append(float(blk / max(att, 1)))
        stats['vol_wait'].append(float(arm.voluntary_waits.sum()))
        # corridor balance: 1.0 = perfectly spread, 0.0 = all on one corridor
        ent = -sum(p * np.log(p) for p in share if p > 0)
        stats['balance'].append(float(ent / np.log(len(share))))
    return {k: statistics.mean(v) for k, v in stats.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--tier', type=int, default=3)
    p.add_argument('--horizon', type=int, default=80)
    p.add_argument('--episodes', type=int, default=160)
    p.add_argument('--sigmas', type=float, nargs='+', default=[0.0, 1.0, 2.0, 3.0])
    p.add_argument('--arms', nargs='+', default=list(ARMS))
    p.add_argument('--verbose', action='store_true', help='per-step log, first episode')
    args = p.parse_args()

    # GATE: at sigma=0 there is no throttle, so admission policy is irrelevant and
    # every arm collapses to the shared navigator. If it cannot match the
    # BatchAStarAgent reference (ISR 1.000 on this frozen config), the navigator is
    # broken and every downstream arm comparison measures navigation, not method.
    gate = run_arm(BlindArm, 0.0, args.n, args.tier, args.horizon, 64)
    print(f'NAVIGATOR GATE  ISR@sigma=0 = {gate["isr"]:.3f}  (must be >= 0.99; '
          f'BatchAStarAgent reference = 1.000)')
    if gate['isr'] < 0.99:
        print('  FAIL -- navigator loses agents with no NS present. Arm comparison')
        print('  below is meaningless; fix navigation before reading any of it.\n')
    else:
        print('  PASS\n')

    print(f'TOLL comparison  N={args.n} tier={args.tier} horizon={args.horizon} '
          f'episodes={args.episodes}')
    print('sigma range restricted to slack>=1.2 (§6.6); sigma>3 is capacity-bound\n')
    print(f'{"arm":>6} {"sigma":>6} {"ISR":>7} {"CSR":>7} {"lam":>6} {"lam_mx":>7} '
          f'{"l_hat":>6} {"blk_rt":>7} {"attempt":>8} {"vol_wait":>9} '
          f'{"balance":>8} {"slack":>7}')
    results = {}
    for sigma in args.sigmas:
        for name in args.arms:
            if args.verbose:
                print(f'\n-- {name} sigma={sigma} episode 0 --')
            r = run_arm(ARMS[name], sigma, args.n, args.tier, args.horizon,
                        args.episodes, args.verbose)
            results[(name, sigma)] = r
            print(f'{name:>6} {sigma:>6.1f} {r["isr"]:>7.3f} {r["csr"]:>7.3f} '
                  f'{r["lam"]:>6.3f} {r["lam_max"]:>7.3f} {r["l_hat"]:>6.3f} '
                  f'{r["block_rate"]:>7.3f} {r["attempts"]:>8.0f} '
                  f'{r["vol_wait"]:>9.0f} {r["balance"]:>8.3f} {r["slack"]:>7.2f}')
        print()

    print('DIAGNOSIS GUIDE')
    print('  lam ~ 0 everywhere      -> felt signal never reaches the price; check')
    print('                             block detection (collisions vs throttle).')
    print('  lam high, vol_wait ~ 0  -> admit rule never binds; (1-l_hat) too large.')
    print('  vol_wait high, ISR down -> synchronized backoff; everyone yields at once.')
    print('  balance flat vs blind   -> price-aware ROUTING is doing nothing.')
    print('  toll <= aimd            -> §9.4: the method contribution collapses.')
    for sigma in args.sigmas:
        if ('toll', sigma) in results and ('aimd', sigma) in results:
            d = results[('toll', sigma)]['isr'] - results[('aimd', sigma)]['isr']
            b = results[('toll', sigma)]['isr'] - results[('blind', sigma)]['isr']
            print(f'  sigma={sigma}: toll-aimd={d:+.3f}  toll-blind={b:+.3f}')


if __name__ == '__main__':
    main()

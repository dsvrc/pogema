"""Contested-corridor non-stationarity for POGEMA (NS_design_guide.md §4, §7 POGEMA row).

    medium      one contested medium PER CORRIDOR (3 corridors -> 3 independent K_c)
    load        L_c(t)  = # active agents currently occupying corridor c
    capacity    K_c(t+1) = K_c + (K_max - K_c)/tau_K - d*sigma*A(t)*max(0, L_c - K_c)
    liability   l_i(t)  = s_i(t) * h( sigma*A(t) * L_{-i}(t) / K_c(t) )     (excludes i)
    delivery    served at a degraded rate: with prob 1 - 1/(1+l_i) the agent's move is
                converted to WAIT -- contested-cell service DELAY, never destruction
    reward      untouched, byte for byte

h(u) = u / max(1-u, floor) -- the standard queueing curve (§4: "the domain's own curve"),
free while u << 1, blows up as u -> 1.

CATEGORY C IS STRUCTURAL (§2.1):
  N=1  -> L_{-i} == 0 always -> l_i == 0 -> no throttle -> byte-identical to stock POGEMA.
  sigma=0 -> u == 0 -> l_i == 0 AND K_c == K_max forever -> byte-identical.
Both hold by construction, not by tuning.

TWO DELIBERATE DEVIATIONS FROM THE §4 TEMPLATE -- see the module docstring notes:
  1. driver enters as sigma*A(t), not (1 + sigma*A(t)), so sigma=0 is exactly the
     stationary reference METHOD_design §8 requires as its upper row.
  2. background bots are VIRTUAL -- they gain capacity rather than occupying cells.
     §7's row says "regional occupancy incl. background bots" but §4 forbids additive
     background load (nonzero u at N=1 = category B in disguise). §4 wins.
"""
import numpy as np

from pogema import GridConfig, PogemaWrapper, pogema_v0

# ---------------------------------------------------------------- map geometry

LEFT_W, MID_W, RIGHT_W, HEIGHT = 6, 12, 6, 13
CORRIDOR_ROWS = (2, 6, 10)


def build_map(corridor_width=1):
    """@ = agent-start pool, $ = target pool, # = obstacle,
    ! = free but in NEITHER pool (grid_config.py:309 fall-through).

    Corridors MUST be '!'. A '.' joins both pools, so agents spawn mid-corridor and
    targets land inside corridors -> counter-flow and head-on deadlock in a one-wide
    corridor, and the guarantee that every agent crosses the medium once is lost.
    """
    open_rows = {r0 + k for r0 in CORRIDOR_ROWS for k in range(corridor_width)}
    return '\n'.join(
        '@' * LEFT_W + ('!' if r in open_rows else '#') * MID_W + '$' * RIGHT_W
        for r in range(HEIGHT)
    )


def corridor_of(row, col, corridor_width=1):
    """Corridor index for a map cell, or None if the cell is not part of the medium."""
    if not (LEFT_W <= col < LEFT_W + MID_W):
        return None
    for idx, r0 in enumerate(CORRIDOR_ROWS):
        if r0 <= row < r0 + corridor_width:
            return idx
    return None


# ---------------------------------------------------------------- driver A(t)

A_MIN = 0.35  # baseline bot traffic -- the driver NEVER reaches zero


def shift_driver(t, period=64, ramp_frac=0.15, on_frac=0.35, a_min=A_MIN):
    """Warehouse bot-shift schedule: ramp up, hold, ramp down, back to baseline
    (§8.6 -- shape should match the domain; a shift is a trapezoid, not a sinusoid).
    Periodic, so the episode shows collapse AND recovery (§8.7).

    A(t) in [a_min, 1], never 0. With a_min=0 the medium goes completely free during
    the off phase, and agents simply wait out every congested window -- measured:
    throttle 0.30 and makespan 30.8 -> 77.5, yet ISR held at 0.995, i.e. the NS was
    fully absorbed. The template's (1+sigma*A) supplies this floor structurally; we
    gate by sigma instead (so sigma=0 stays stationary) and put the floor in A."""
    ph = (t % period) / period
    if ph < ramp_frac:
        peak = ph / ramp_frac
    elif ph < ramp_frac + on_frac:
        peak = 1.0
    elif ph < 2 * ramp_frac + on_frac:
        peak = 1.0 - (ph - ramp_frac - on_frac) / ramp_frac
    else:
        peak = 0.0
    return a_min + (1.0 - a_min) * peak


# ---------------------------------------------------------------- the wrapper

class ContestedCorridorNS(PogemaWrapper):
    """sigma is the ONLY exposed dial (§9). Everything else is fixed and published."""

    # published constants (§9: fixed per environment)
    # tier 1 = "fixed and ample" -> no contention. Tiers 2/3 must sit low enough that
    # max(0, L - K) actually fires: at K_MAX=3 with 8 agents over 3 corridors the
    # damage term never triggered, so tier 3's hysteresis was inert and tier 3 was
    # silently identical to tier 2.
    K_MAX = {1: 64.0, 2: 2.0, 3: 2.0}
    TAU_K = 12.0        # capacity regeneration time constant
    DAMAGE = 0.25       # d
    H_FLOOR = 0.05      # h(u) = u/max(1-u, H_FLOOR)
    DRIVER_PERIOD = 64
    S_SPREAD = 0.6      # susceptibility spread, tier 3 only
    S_PERIOD = 96       # susceptibility rotation period (§4: "who is fragile drifts")

    def __init__(self, env, sigma=0.0, tier=3, corridor_width=1, expose_oracle=False):
        super().__init__(env)
        self.sigma = float(sigma)
        self.tier = int(tier)
        self.corridor_width = corridor_width
        self.expose_oracle = expose_oracle
        self.k_max = self.K_MAX[self.tier]
        self._n_corr = len(CORRIDOR_ROWS)
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._K = np.full(self._n_corr, self.k_max)
        self._spawn_row = None
        self._ep = None

    # -- susceptibility ---------------------------------------------------

    def _susceptibility(self, agent_idx):
        """s_i multiplies the cross-agent term (§2.1), so it is invisible at N=1.

        Tier 3 only: rotates slowly, phase tied to the agent's home row, which is
        §7's 'home-region overlap with flows' -- agents whose home sits opposite the
        busy corridor feel the load more."""
        if self.tier < 3:
            return 1.0
        phase = 2 * np.pi * self._spawn_row[agent_idx] / HEIGHT
        return 1.0 + self.S_SPREAD * np.cos(2 * np.pi * self._t / self.S_PERIOD + phase)

    def _h(self, u):
        return u / max(1.0 - u, self.H_FLOOR)

    # -- gym API ----------------------------------------------------------

    def reset(self, seed=None, **kwargs):
        obs, infos = self.env.reset(seed=seed, **kwargs)
        self._rng = np.random.default_rng(0 if seed is None else seed)
        self._t = 0
        self._K = np.full(self._n_corr, self.k_max)
        self._spawn_row = [xy[0] for xy in self.get_agents_xy(ignore_borders=True)]
        self._ep = {'A': [], 'u': [], 'l': [], 'throttled': 0, 'medium_steps': 0,
                    'K_min': self.k_max}
        return obs, infos

    def step(self, actions):
        actions = list(actions)
        A = shift_driver(self._t, self.DRIVER_PERIOD)
        gain = self.sigma * A  # multiplicative on the cross-agent term ONLY

        xy = self.get_agents_xy(ignore_borders=True)
        active = self.grid.is_active
        moves = self.grid_config.MOVES

        # Load Phi = attempted DEMAND on the corridor, not realized occupancy:
        # an agent inside c, or queued at its mouth trying to enter, is contending
        # for c either way (a backlogged CSMA/CA station is still load).
        #
        # Occupancy-based load makes the NS self-correcting: throttling empties the
        # corridor, which drops L_{-i} to 0, which grants the next arrival free
        # passage -- the medium ends up scheduling the team one-at-a-time and ISR
        # RECOVERS as sigma grows. Measured with occupancy: ISR 0.52 at sigma=3 but
        # 0.76 at sigma=6, non-monotonic. Demand keeps retries on the books.
        claim = []
        for i, (x, y) in enumerate(xy):
            if not active[i]:
                claim.append(None)
                continue
            c = corridor_of(x, y, self.corridor_width)
            if c is None:
                dx, dy = moves[actions[i]]
                c = corridor_of(x + dx, y + dy, self.corridor_width)
            claim.append(c)

        L = np.zeros(self._n_corr)
        for c in claim:
            if c is not None:
                L[c] += 1

        n_throttled = 0
        for i, (x, y) in enumerate(xy):
            if not active[i]:
                continue
            c = claim[i]
            if c is None:
                continue  # not using the medium this step; home/delivery regions are free

            L_minus_i = L[c] - 1.0  # exclude i's own claim
            u = gain * L_minus_i / max(self._K[c], 1e-6)
            liability = self._susceptibility(i) * self._h(u)
            p_serve = 1.0 / (1.0 + liability)

            self._ep['u'].append(u)
            self._ep['l'].append(liability)
            self._ep['medium_steps'] += 1

            # service DELAY: the move is not served this step. Never destroyed,
            # and not unilaterally invertible -- you cannot move faster than the
            # medium serves you (§4, delivery harm).
            if actions[i] != 0 and self._rng.random() > p_serve:
                actions[i] = 0
                n_throttled += 1

        # capacity: regenerate toward K_max, damaged only by overload (§4).
        # sigma gates the damage so sigma=0 leaves K_c == K_max exactly.
        if self.tier >= 3:
            overload = np.maximum(0.0, L - self._K)
            self._K += (self.k_max - self._K) / self.TAU_K - self.DAMAGE * gain * overload
            self._K = np.clip(self._K, 0.25, self.k_max)

        self._ep['A'].append(A)
        self._ep['throttled'] += n_throttled
        self._ep['K_min'] = min(self._ep['K_min'], float(self._K.min()))
        self._t += 1

        obs, reward, terminated, truncated, infos = self.env.step(actions)

        if self.expose_oracle:  # §6.4 oracle-driver gate: true A, K, l_i in info
            for i in range(len(infos)):
                infos[i]['oracle'] = {'A': A, 'K': self._K.copy(), 'sigma': self.sigma}

        if all(terminated) or all(truncated):
            infos[0].setdefault('metrics', {}).update(
                ns_A_mean=float(np.mean(self._ep['A'])),
                ns_u_mean=float(np.mean(self._ep['u'])) if self._ep['u'] else 0.0,
                ns_l_mean=float(np.mean(self._ep['l'])) if self._ep['l'] else 0.0,
                ns_throttle_rate=(self._ep['throttled'] / max(self._ep['medium_steps'], 1)),
                ns_K_min=self._ep['K_min'],
            )
        return obs, reward, terminated, truncated, infos


def make_contested_env(num_agents=8, sigma=0.0, tier=3, horizon=128,
                       corridor_width=1, expose_oracle=False, seed=None):
    cfg = GridConfig(
        map=build_map(corridor_width),
        num_agents=num_agents,
        on_target='finish',          # fixed total work -> scheduling-limited (§3.1)
        collision_system='soft',
        observation_type='POMAPF',
        obs_radius=3,                # K(t) stays latent -- never placed in the obs
        max_episode_steps=horizon,
        integration=None,
        seed=seed,
    )
    return ContestedCorridorNS(pogema_v0(grid_config=cfg), sigma=sigma, tier=tier,
                               corridor_width=corridor_width, expose_oracle=expose_oracle)

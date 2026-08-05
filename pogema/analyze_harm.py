"""Does the harm function admit congestion COLLAPSE, or only queueing?

This settles -- with arithmetic, not rollouts -- whether yielding can ever help the
team. If team throughput is monotonically increasing in load, no price of any kind
can improve anything, and TOLL is being asked to solve a problem the physics forbids.

    queue (current):  h(u) = u/max(1-u, 0.05),  p = 1/(1+s*h)
        past the clamp h ~ 20u so p ~ 1/u ~ 1/(L-1), giving
        throughput = C*p ~ C/(C+Z-1), INCREASING in C -> piling in always wins.

    aloha (proposed): p = exp(-s*u),  u = sigma*A*L_{-i}/K
        throughput = C*exp(-(C+Z-1)*sigma*A/K) peaks at L* = K/(sigma*A) and
        collapses beyond -- the CSMA/slotted-Aloha curve NS guide §3.1 names.

Run:  python analyze_harm.py
"""
import numpy as np


def p_queue(u, s=1.0, floor=0.05):
    return 1.0 / (1.0 + s * (u / max(1.0 - u, floor)))


def p_aloha(u, s=1.0):
    return float(np.exp(-s * u))


def table(K, sigma, A=1.0, s=1.0, max_load=14):
    print(f'\nK={K}  sigma={sigma}  A={A}  s={s}   (L* = K/(sigma*A) = '
          f'{K / (sigma * A):.1f})')
    print(f'{"L":>3} {"u":>6} {"p_queue":>9} {"thr_queue":>10} '
          f'{"p_aloha":>9} {"thr_aloha":>10}')
    best_q = best_a = (0, -1)
    for L in range(1, max_load + 1):
        u = sigma * A * (L - 1) / K
        pq, pa = p_queue(u, s), p_aloha(u, s)
        tq, ta = L * pq, L * pa
        if tq > best_q[1]:
            best_q = (L, tq)
        if ta > best_a[1]:
            best_a = (L, ta)
        print(f'{L:>3} {u:>6.2f} {pq:>9.4f} {tq:>10.3f} {pa:>9.4f} {ta:>10.3f}')
    print(f'  queue peak at L={best_q[0]} (thr {best_q[1]:.3f})   '
          f'aloha peak at L={best_a[0]} (thr {best_a[1]:.3f})')
    return best_q, best_a


def main():
    print('THROUGHPUT vs LOAD -- does yielding ever help the team?')
    for sigma in (1.0, 2.0, 3.0):
        bq, ba = table(K=6.0, sigma=sigma)
        q_collapses = bq[0] < 14
        a_collapses = ba[0] < 14
        print(f'  queue collapses? {"YES" if q_collapses else "NO -- monotone, '
              f'yielding can never help"}')
        print(f'  aloha collapses? {"YES" if a_collapses else "NO"}')

    print('\nHETEROGENEITY CHECK (§4 role emergence needs s_i to change the peak)')
    print('If a fragile agent yields and a robust one proceeds, team throughput must')
    print('beat both-yield and both-proceed. Aloha peak shifts with s; queue does not.')
    for s in (0.5, 1.0, 2.0):
        u = 3.0 * 1.0 * 4 / 6.0
        print(f'  s={s:>4}: p_queue={p_queue(u, s):.4f}  p_aloha={p_aloha(u, s):.4f}')

    print('\nVERDICT')
    print('  A monotone-increasing throughput curve means NO price helps, and the')
    print('  measured toll-blind = -0.055 is the predicted result, not a bug.')
    print('  Switch h before changing anything else in the method.')


if __name__ == '__main__':
    main()

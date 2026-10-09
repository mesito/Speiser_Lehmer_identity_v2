"""Numerical constants and the conventions of the paper."""
import numpy as np

# pi to 40 digits; the 80-bit value is what the phase reductions use.  With the
# double-precision pi the Riemann-Siegel instrument loses three orders of
# magnitude at t ~ 8.4e9 (Section 2.3 of the paper).
PI_STR = "3.141592653589793238462643383279502884197"
PI_LD = np.longdouble(PI_STR)
TWO_PI_LD = 2 * PI_LD
GAMMA_E = 0.5772156649015328606        # Euler's constant

# conventions
S_TIGHT = 0.15                         # tight pair: s = g * dens(t0) < 0.15
W = 250                                # window (ordinates on each side) for far, lambda_far
XS = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]   # Euler cutoffs
DELTAS = (100.0, 1000.0)               # half-widths of the kernel-sum windows S_Delta
N_CTRL = 300                           # control mid-gaps
CTRL_SEED = 11                         # rng seed of the control draw
CTRL_BAND = (0.95, 1.05)               # normalised spacing band of the controls
T0_PRIMARY = 8436146000.0              # start of the LMFDB block
N_PRIMARY = 772719                     # zeros in the LMFDB block
N_PAIRS_PRIMARY = 2736
N_PAIRS_2M = 6088
GRID_FLOOR = 33                        # interior grid for max|Z| over a gap (plus parabolic refinement)


def dens(t):
    """local mean density of ordinates at height t."""
    return np.log(np.asarray(t, dtype=float) / (2 * np.pi)) / (2 * np.pi)

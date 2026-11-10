"""RHAN-IM — Ignorance-Map perception (redesign prototype).

Perception is treated as *search with a stopping rule*. The recurrent state is
an exact Bayesian posterior over "what is in the scene and where" together with
an accumulated-precision map that records how well each region has been
resolved. Every look is a measurement with a *measured* reliability (d'), the
next look is the one that is expected to reduce error the most, evidence enters
the state through a bounded channel, and the stopping threshold is calibrated
with a finite-sample risk-control procedure.

Nothing in this package modifies the Gen-1 / Gen-2 code or any checkpoint.
See docs/research/RHAN_IM_00_README.md for the specification, the literature notes and the
experiment plan.
"""

from rhan_im.sdt import (  # noqa: F401
    bayes_error_equal_prior,
    dprime_from_samples,
    gaussian_llr,
    normal_cdf,
    zroc_slope,
)
from rhan_im.belief import SearchPosterior  # noqa: F401
from rhan_im.policy import (  # noqa: F401
    evoi_fixation,
    greedy_mode_fixation,
    koopman_fixation,
    koopman_gain,
    mc_expected_max_posterior,
    random_fixation,
)
from rhan_im.stopping import (  # noqa: F401
    binom_cdf,
    curtail_certificate,
    curtail_certificate_simple,
    future_swing_bound,
    ltt_fixed_sequence,
    max_step_gain,
    max_step_swing,
    min_fixations_to_decide,
    min_fixations_to_reach,
    naive_plugin_choice,
    patch_certificate,
    stopped_stats,
    stopped_stats_selective,
)

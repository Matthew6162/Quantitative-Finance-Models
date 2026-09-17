"""Plotting helpers for the Black-Scholes Greeks numerical-validation section.

Kept separate from src/validation.py and src/validation_jax.py on purpose:
tests/test_greeks.py imports those two for the numerical checks and shouldn't
need to import matplotlib to run them. This module is imported by the
notebook only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

Array = npt.NDArray[np.float64]
Series = Array | tuple[Array, Array]  # one curve, or a (call, put) pair

# Categorical colours (from a colour-vision-deficiency-validated palette),
# assigned to whichever method names show up in `results`, in the order
# they're first seen -- not hardcoded to specific method names, so this still
# works if a third validation method is ever added. Capped at four: beyond
# that the palette's own adjacent-pair guarantees stop holding (see
# src/plotting.py's design discussion in chat), so a fifth method should mean
# splitting into more than one figure rather than adding a fifth colour here.
_METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow


def plot_error_sweep(x: Array, x_label: str, results: dict[str, dict[str, Series]],
                     ncols: int = 3,
                     axis_size: tuple[float, float] = (4, 3)) -> tuple[Figure, npt.NDArray]:
    """Grid of relative-error-vs-parameter subplots, one per Greek, comparing
    one or more validation methods (e.g. finite differences vs JAX autodiff)
    against each other on the same axes.

    x : array of the swept parameter values (e.g. np.linspace over S, T, or sigma).
    x_label : label for the swept parameter, e.g. "Time till expiration".
    results : dict mapping a Greek's display name to a dict of
        {method_name: values}, where values is either
        - a 1D array of relative errors (no call/put split), or
        - a (call_errors, put_errors) tuple.

      e.g.:
        {
            "Theta": {"FD": (fd_call_err, fd_put_err), "JAX": (jax_call_err, jax_put_err)},
            "Gamma": {"FD": fd_gamma_err, "JAX": jax_gamma_err},
            "Vanna": {"JAX": jax_vanna_err},
        }

      Compute every relative-error array yourself before calling this --
      this function only draws what it's given.

    Method and call/put are two independent things that can vary at once, so
    they're mapped to two different visual channels instead of stacking both
    into colour (which would need up to four colours per subplot and clutter
    fast): colour encodes the *method* (fixed per name, consistent across
    every subplot in the grid), line style encodes call (solid) vs put
    (dashed) wherever that distinction applies. A Greek with one method and
    no call/put split just draws one solid line; a Greek with two methods and
    a call/put split draws up to four lines using only two colours and two
    dash patterns.

    Each subplot uses a log y-axis: relative error spans several orders of
    magnitude, especially comparing a finite-difference method against exact
    autodiff, so a linear axis would flatten the more accurate method's curve
    to indistinguishable-from-zero.

    Returns (fig, axes).
    """
    names = list(results.keys())
    nrows = -(-len(names) // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(axis_size[0] * ncols, axis_size[1] * nrows),
        squeeze=False,
    )

    # Assign each method name a fixed colour, in first-seen order, shared
    # across every subplot -- so e.g. "JAX" is the same colour in the Gamma
    # panel as it is in the Vanna panel, even though Vanna has no "FD" entry.
    method_order = []
    has_call_put = False
    for methods in results.values():
        for method_name, value in methods.items():
            if method_name not in method_order:
                method_order.append(method_name)
            if isinstance(value, tuple):
                has_call_put = True

    if len(method_order) > len(_METHOD_COLORS):
        raise ValueError(
            f"plot_error_sweep only has {len(_METHOD_COLORS)} method colours "
            f"defined; got {len(method_order)} methods: {method_order}. "
            "Split into more than one figure rather than adding a colour."
        )
    method_colors = dict(zip(method_order, _METHOD_COLORS))

    for ax, name in zip(axes.flat, names):
        for method_name, value in results[name].items():
            color = method_colors[method_name]
            if isinstance(value, tuple):
                call_err, put_err = value
                ax.semilogy(x, call_err, color=color, linewidth=2, linestyle="-")
                ax.semilogy(x, put_err, color=color, linewidth=2, linestyle="--")
            else:
                ax.semilogy(x, value, color=color, linewidth=2, linestyle="-")

        ax.set_title(name)
        ax.set_xlabel(x_label)
        ax.set_ylabel("relative error")
        ax.grid(True, which="both", linewidth=0.5, alpha=0.4)

    # blank out any unused grid cells
    for ax in axes.flat[len(names):]:
        ax.axis("off")

    # Legend has two parts, since colour and line style each carry their own
    # meaning: one colour swatch per method (in its real colour), plus, only
    # if any Greek used a call/put pair, two neutral-coloured swatches
    # explaining solid = call / dashed = put once for the whole figure.
    legend_handles = [
        Line2D([0], [0], color=method_colors[m], linewidth=2, label=m)
        for m in method_order
    ]
    if has_call_put:
        legend_handles += [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="call"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="put"),
        ]

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), bbox_to_anchor=(0.5, 1.0))

    return fig, axes

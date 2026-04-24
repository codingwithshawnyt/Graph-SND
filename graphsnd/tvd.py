"""Total Variation Distance for categorical distributions.

Used by the MPE measurement panel to compute Graph-SND with discrete
action spaces. Not integrated into ``het_control/snd.py`` — this is a
standalone measurement utility.
"""

from __future__ import annotations

import torch
from torch import Tensor


def tvd(p: Tensor, q: Tensor) -> Tensor:
    """Total Variation Distance: ``0.5 * sum(|p_i - q_i|)`` over the last dim.

    Parameters
    ----------
    p, q : Tensor
        Probability vectors of shape ``(..., K)`` where ``K`` is the
        number of categories. Entries should be non-negative and sum to
        ~1 along the last dimension.

    Returns
    -------
    Tensor
        Shape ``(...)`` — the TVD for each pair of distributions in the
        batch. Values lie in ``[0, 1]``.
    """
    return 0.5 * (p - q).abs().sum(dim=-1)


def tvd_pairwise(probs: Tensor) -> Tensor:
    """Pairwise TVD matrix for ``n`` categorical distributions.

    Parameters
    ----------
    probs : Tensor
        Shape ``(n, K)`` — one probability vector per agent.

    Returns
    -------
    Tensor
        Shape ``(n, n)`` — symmetric matrix where entry ``(i, j)`` is
        ``tvd(probs[i], probs[j])``.
    """
    n = probs.shape[0]
    # Broadcast: (n, 1, K) - (1, n, K) -> (n, n, K)
    diff = probs.unsqueeze(1) - probs.unsqueeze(0)
    return 0.5 * diff.abs().sum(dim=-1)

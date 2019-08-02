"""
SLIM implementation(s).
"""

import logging

import numpy as np
import pandas as pd

from numba import njit

_log = logging.getLogger(__name__)


@njit
def _ind_add(ys, iis, istart, iend, x):
    for j in range(istart, iend):
        ys[iis[j]] += x


@njit
def _ind_add_sum_old(ys, iis, istart, iend, x):
    ret = 0.0
    for j in range(istart, iend):
        i = iis[j]
        ret += y[i]
        ys[i] += x
    return ret


@njit
def _cd_round(item, matrix, tmat, nbrs, weights, l2reg, l1reg):
    """
    One round of solving an item's weights with coordinate descent.

    Args:
        item(int): the item we are scoring
        matrix(CSR): the user-item rating matrix
        tmat(CSR): the item-user rating matrix
        nbrs(ndarray): the candidate neighbors (co-rated items)
        weights(ndarray): the current or initial coefficients for nbrs
        l2reg(float): the L2 regularization term
        l1reg(float): the L1 regularization term

    Returns:
        ndarray: the new coefficient weights
    """
    # compute residuals for this item for users rating neighbors
    resid = np.zeros(matrix.nrows)
    # start by computing -pred
    for ji, j in enumerate(nbrs):
        s, e = tmat.row_extent(j)
        _ind_add(resid, tmat.colinds, s, e, -weights[ji])
    # then add 1 to everything rated by us!
    resid[tmat.row_cs(item)] += 1.0

    # now we do a round of coordinate descent
    for ji, j in enumerate(nbrs):
        grad = 0
        lb, ub = tmat.row_extent(j)
        # extract the gradient and take out current weights for update
        grad = _ind_add_sum_old(resid, tmat.colinds, lb, ub, weights[ji])
        grad /= matrix.nrows
        # update weights
        if np.abs(grad) <= l1reg:
            weights[ji] = 0.0
        elif grad > 0:
            weights[ji] = (grad - l1reg) / (1 + l2reg)
        else:
            weights[ji] = (grad + l1reg) / (1 + l2reg)
        # force nonnegative
        if weights[ji] < 0:
            weights[ji] = 0
        # update residuals with new weights
        if weights[ji] > 0:
            _ind_add(resid, tmat.colinds, lb, ub, -weights[ji])

import logging
from collections import namedtuple

import numpy as np
import numba as n

from .mf_common import MFPredictor
from ..matrix import sparse_ratings
from ..util import no_progress, Stopwatch

_log = logging.getLogger(__name__)

BPRParams = namedtuple('BPRParams', ['learn_rate', 'u_reg', 'i_reg', 'j_reg'])


@n.njit
def _sample_pair(rmat):
    u = np.random.randint(rmat.nrows)
    rated = rmat.row_cs(u)
    i = np.random.choice(rated)
    j = np.random.randint(rmat.ncols)
    while np.searchsorted(rated, j) == j:
        # this is found
        j = np.random.randint(rmat.ncols)

    return u, i, j


@n.njit
def _bpr_iter(rmat, umat, imat, params):
    nf = umat.shape[1]
    for k in range(rmat.nnz):
        u, i, j = _sample_pair(rmat)
        uv = umat[u, :]
        iv = imat[i, :]
        jv = imat[j, :]
        xuij = np.dot(uv, iv) - np.dot(uv, jv)
        exuij = np.exp(-xuij)
        mult = exuij / (1 + exuij)
        for f in range(nf):
            umat[u, f] += params.learn_rate * (mult * (iv[f] - jv[f]) + params.u_reg * uv[f])
            imat[i, f] += params.learn_rate * (mult * uv[f] + params.i_reg * iv[f])
            imat[j, f] += params.learn_rate * (-mult * uv[f] + params.j_reg * jv[f])


class BPR(MFPredictor):
    """
    Custom implementation of BPR with no extra dependencies.

    .. note:: :cls:`lenskit.algorithms.implicit.BPR` will often be faster.
    """
    def __init__(self, features, *, iterations=100, learning_rate=0.05, regularization=0.0025,
                 progress=no_progress):
        self.features = features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.progress = progress

    def fit(self, ratings):
        timer = Stopwatch()
        rmat, uidx, iidx = sparse_ratings(ratings)
        rmat.sort_keys()

        umat = np.random.randn(len(uidx), self.features) * 0.1
        imat = np.random.randn(len(iidx), self.features) * 0.1

        if isinstance(self.regularization, tuple):
            u_reg, i_reg, j_reg = self.regularization
        else:
            u_reg = i_reg = j_reg = self.regularization

        params = BPRParams(self.learning_rate, u_reg, i_reg, j_reg)

        _log.info('training %d features on %d ratings for %d iterations',
                  self.features, rmat.nnz, self.iterations)
        for i in self.progress(range(self.iterations)):
            it = Stopwatch()
            _bpr_iter(rmat.N, umat, imat, params)
            _log.debug('iteration %d took %s', i+1, it)

        _log.info('trained %d features for %d iterations in %s',
                  self.features, self.iterations, timer)

        self.user_index_ = uidx
        self.item_index_ = iidx
        self.user_features_ = umat
        self.item_features_ = imat

        return self

    def predict_for_user(self, user, items, ratings=None):
        return self.score_by_ids(user, items)

    def __str__(self):
        return 'BPR({}, α={}, λ={})'.format(self.features, self.learning_rate, self.regularization)

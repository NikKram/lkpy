import logging

import numpy as np

from lenskit.math.elastic import fit_net

from pytest import approx, mark

try:
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr
except ImportError:
    r = None

_log = logging.getLogger(__name__)
need_rpy = mark.skipif(r is None, reason='RPY2 not available')


@need_rpy
def test_fit_qs():
    glm = importr('glmnet')
    r('data(QuickStartExample)')
    x = np.array(r['x'])
    y = np.array(r['y'])

    _log.info('fitting R glmnet model')
    fit = glm.glmnet(r.x, r.y, alpha=0.5)

    assert False

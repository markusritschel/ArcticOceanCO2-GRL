# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-08-02
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging
import matplotlib.colors as mcolors
import numpy as np

log = logging.getLogger(__name__)


class MidpointNorm(mcolors.Normalize):
    """TODO: add docstring"""
    def __init__(self, vmin=None, vmax=None, center=None, clip=False):
        self.midpoint = center
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max( [ np.abs(self.vmin), np.abs(self.vmax) ] )
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


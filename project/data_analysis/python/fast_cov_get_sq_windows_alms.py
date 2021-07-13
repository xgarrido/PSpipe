# This script compute all alms squared windows, it's a necessary step of covariance computation.

import logging
import os
import sys
from itertools import combinations_with_replacement as cwr
from itertools import product

import yaml
from pspy import pspy_utils, so_map, so_mpi, sph_tools

d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)

product_dir = d["product_dir"]
windows_dir = os.path.join(product_dir, "windows")
sq_win_alms_dir = os.path.join(product_dir, "sq_win_alms")
os.makedirs(sq_win_alms_dir, exist_ok=True)

surveys = d.get("surveys")
svxar = [(k, ar) for k, v in surveys.items() for ar in v.get("arrays")]
comb = [c1 + c2 for c1, c2 in cwr(svxar, 2)]

logging.info(f"Number of sq win alms to compute : {len(comb)}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(comb) - 1)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = comb[task]

    win_T1 = so_map.read_map(d["window_T_%s_%s" % (sv1, ar1)])
    win_T2 = so_map.read_map(d["window_T_%s_%s" % (sv2, ar2)])

    sq_win = win_T1.copy()
    sq_win.data[:] *= win_T2.data[:]
    sqwin_alm = sph_tools.map2alm(sq_win, niter=niter, lmax=lmax)

    np.save("%s/alms_%s_%sx%s_%s.npy" % (sq_win_alms_dir, sv1, ar1, sv2, ar2), sqwin_alm)

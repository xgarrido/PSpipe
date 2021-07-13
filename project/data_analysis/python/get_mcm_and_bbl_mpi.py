# This script computes the mode coupling matrices and the binning matrices Bbl
# for the different surveys and arrays.

import logging
import os
import sys
from itertools import combinations_with_replacement as cwr
from itertools import product

import numpy as np
import yaml
from pspy import pspy_utils, so_map, so_mcm, so_mpi, sph_tools

d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)

data_dir = d["data_dir"]
product_dir = d["product_dir"]
windows_dir = os.path.join(product_dir, "windows")
mcms_dir = os.path.join(product_dir, "mcms")
os.makedirs(mcms_dir, exist_ok=True)
sq_win_alms_dir = os.path.join(product_dir, "sq_win_alms")
os.makedirs(sq_win_alms_dir, exist_ok=True)

l_exact, l_band, l_toep = None, None, None
if d.get("use_toeplitz_mcm", False):
    l_exact, l_band, l_toep = d.get("l_exact", 800), d.get("l_band", 2000), d.get("l_toep", 2750)
    logging.info(
        "Use the toeplitz approximation with the following parameters: "
        f"l_exact={l_exact}, l_band={l_band}, l_toep={l_toep}"
    )

surveys = d.get("surveys")
svxar = [(sv, ar) for sv in d.get("selected_surveys") for ar in surveys.get(sv).get("arrays")]
comb = [c1 + c2 for c1, c2 in cwr(svxar, 2)]

logging.info(f"Number of mcm matrices to compute : {len(comb)}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(comb) - 1)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = comb[task]
    logging.debug(f"[{task:>3}] Compute mcm for {sv1}_{ar1}x{sv2}_{ar2}...")

    survey1, survey2 = surveys.get(sv1), surveys.get(sv2)
    ell, bl1 = pspy_utils.read_beam_file(os.path.join(data_dir, survey1.get("beams").get(ar1)))
    win1 = so_map.read_map(os.path.join(windows_dir, f"window_{sv1}_{ar1}.fits"))
    ell, bl2 = pspy_utils.read_beam_file(os.path.join(data_dir, survey2.get("beams").get(ar2)))
    win2 = so_map.read_map(os.path.join(windows_dir, f"window_{sv2}_{ar2}.fits"))

    # Make no difference between temperature/polarization windows
    mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(
        win1=(win1, win1),
        win2=(win2, win2),
        bl1=(bl1, bl1),
        bl2=(bl2, bl2),
        binning_file=os.path.join(data_dir, d.get("binning_file")),
        niter=d.get("niter", 0),
        lmax=d.get("lmax", 7930),
        type=d.get("type", "Dl"),
        l_exact=l_exact,
        l_band=l_band,
        l_toep=l_toep,
        save_file=os.path.join(mcms_dir, f"{sv1}_{ar1}x{sv2}_{ar2}"),
    )

    # This was initially done in fast_cov_get_sq_windows_alms
    sq_win = win1.copy()
    sq_win.data *= win2.data
    sqwin_alm = sph_tools.map2alm(sq_win, niter=d.get("niter", 0), lmax=d.get("lmax"))
    np.save(os.path.join(sq_win_alms_dir, f"alms_{sv1}_{ar1}x{sv2}_{ar2}.npy"), sqwin_alm)

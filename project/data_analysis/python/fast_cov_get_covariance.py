#
# This script compute the analytical covariance matrix elements.
#
import logging
import os
import sys
from itertools import combinations_with_replacement as cwr
from itertools import product

import data_analysis_utils
import numpy as np
import yaml
from pspy import pspy_utils, so_mcm, so_mpi, so_spectra

d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)

data_dir = d["data_dir"]
product_dir = d["product_dir"]
mcms_dir = os.path.join(product_dir, "mcms")
ps_model_dir = os.path.join(product_dir, "noise_model")
best_fits_dir = os.path.join(product_dir, "best_fits")
sq_win_alms_dir = os.path.join(product_dir, "sq_win_alms")
cov_dir = os.path.join(product_dir, "covariances")
os.makedirs(cov_dir, exist_ok=True)

lmax = d["lmax"]
data_dir = d["data_dir"]
binning_file = os.path.join(data_dir, d["binning_file"])
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)


surveys = d.get("surveys")
svxar = [(sv, ar) for sv in d.get("selected_surveys") for ar in surveys.get(sv).get("arrays")]
comb = [c1 + c2 for c1, c2 in cwr(svxar, 2)]

logging.info(f"Retrieving best fit & noise models ({len(comb)} combinations)")
nsplits, ps_all, nl_all = {}, {}, {}
for sv1, ar1, sv2, ar2 in comb:
    if sv1 == sv2 and ar1 == ar2:
        _, Nl = so_spectra.read_ps(
            os.path.join(ps_model_dir, f"mean_{ar1}x{ar2}_{sv1}_noise.dat"),
            spectra=d.get("spectra", ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]),
        )
    survey1, survey2 = surveys.get(sv1), surveys.get(sv2)
    ell, bl1 = pspy_utils.read_beam_file(os.path.join(data_dir, survey1.get("beams").get(ar1)))
    ell, bl2 = pspy_utils.read_beam_file(os.path.join(data_dir, survey2.get("beams").get(ar2)))
    bl1 = bl1[2 : lmax + 2]
    bl2 = bl2[2 : lmax + 2]

    freq1 = survey1.get("frequencies").get(ar1)
    freq2 = survey2.get("frequencies").get(ar2)
    for spec in ["TT", "TE", "ET", "EE"]:
        _, ps_th = np.loadtxt(
            os.path.join(best_fits_dir, f"best_fit_{freq1}x{freq2}_{spec}.dat"), unpack=True
        )
        key = (f"{sv1}&{ar1}", f"{sv2}&{ar2}", spec)
        ps_all[key] = bl1 * bl2 * ps_th[:lmax]

        if sv1 == sv2 and ar1 == ar2:
            nsplits[sv1] = len(survey1.get("maps").get(ar1))
            nl_all[key] = Nl[spec][:lmax] * nsplits[sv1]
        else:
            nl_all[key] = np.zeros(lmax)

        ikey = (f"{sv2}&{ar2}", f"{sv1}&{ar1}", spec)
        ps_all[ikey] = ps_all[key]
        nl_all[ikey] = nl_all[key]

l_exact, l_band, l_toep = None, None, None
if d.get("use_toeplitz_cov", True):
    l_exact, l_band, l_toep = d.get("l_exact", 800), d.get("l_band", 2000), d.get("l_toep", 2750)
    logging.info(
        "Use the toeplitz approximation with the following parameters: "
        f"l_exact={l_exact}, l_band={l_band}, l_toep={l_toep}"
    )

comb = [c1 + c2 for c1, c2 in cwr(comb, 2)]
logging.info(f"Number of covariance matrices to compute : {len(comb)}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(comb) - 1)
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2, sv3, ar3, sv4, ar4 = comb[task]

    def get_abcd(sep):
        return f"{sv1}{sep}{ar1}", f"{sv2}{sep}{ar2}", f"{sv3}{sep}{ar3}", f"{sv4}{sep}{ar4}"

    na, nb, nc, nd = get_abcd("&")
    na_r, nb_r, nc_r, nd_r = get_abcd("_")

    fn = os.path.join(cov_dir, f"analytic_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}.npy")
    if not d.get("force", False) and os.path.isfile(fn):
        continue

    logging.info(f"[{so_mpi.comm.rank:>2}] cov element ({na_r} x {nb_r}, {nc_r} x {nd_r})")

    coupling = data_analysis_utils.fast_cov_coupling(
        sq_win_alms_dir, na_r, nb_r, nc_r, nd_r, lmax, l_exact=l_exact, l_band=l_band, l_toep=l_toep
    )

    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
    try:
        mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(
            prefix=os.path.join(mcms_dir, f"{na_r}x{nb_r}"), spin_pairs=spin_pairs
        )
    except:
        mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(
            prefix=os.path.join(mcms_dir, f"{nb_r}x{na_r}"), spin_pairs=spin_pairs
        )

    try:
        mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(
            prefix=os.path.join(mcms_dir, f"{nc_r}x{nd_r}"), spin_pairs=spin_pairs
        )
    except:
        mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(
            prefix=os.path.join(mcms_dir, f"{nd_r}x{nc_r}"), spin_pairs=spin_pairs
        )

    analytic_cov = data_analysis_utils.covariance_element(
        coupling, [na, nb, nc, nd], nsplits, ps_all, nl_all, binning_file, mbb_inv_ab, mbb_inv_cd
    )

    # Some heuristic correction for the number of modes lost due to the transfer function
    # This should be tested against simulation and revisited
    tf = np.ones(len(lb))
    for sv in [sv1, sv2, sv3, sv4]:
        tf_name = surveys.get(sv).get("transfer_function")
        if tf_name is not None:
            # logging.debug(f"Using transfer function for {sv}")
            _, _, sv_tf, _ = np.loadtxt(os.path.join(data_dir, tf_name), unpack=True)
            tf *= sv_tf[: len(lb)] ** (1 / 4)

    cov_tf = np.tile(tf, 4)
    analytic_cov /= np.outer(np.sqrt(cov_tf), np.sqrt(cov_tf))

    np.save(fn, analytic_cov)

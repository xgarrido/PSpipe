# This script use the covariance matrix elements to form a multifrequency covariance matrix
# with block TT - TE - ET - EE
# Note that for the ET block, we do not include any same array, same survey spectra, since for
# these guys TE = ET
import logging
import os
import sys
from itertools import combinations_with_replacement as cwr
from itertools import product

import data_analysis_utils
import numpy as np
import yaml
from pspy import pspy_utils

d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)

product_dir = d["product_dir"]
cov_dir = os.path.join(product_dir, "covariances")

lmax = d["lmax"]
data_dir = d["data_dir"]
binning_file = os.path.join(data_dir, d["binning_file"])
bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nbins = len(bin_hi)

# We make a list of all spectra included in the analysis
surveys = d.get("surveys")
svxars = [(sv, ar) for sv in d.get("selected_surveys") for ar in surveys.get(sv).get("arrays")]
comb = [c1 + c2 for c1, c2 in cwr(svxars, 2)]
spec_names = ["{}_{}x{}_{}".format(*c) for c in comb]

# We fill the full covariance matrix with our elements
# The cov mat format is [block TT, block TE, block ET, block EE]
# The block contain all cross array and season spectra

spectra = ["TT", "TE", "ET", "EE"]

nspec = len(spec_names)
full_analytic_cov = np.zeros((4 * nspec * nbins, 4 * nspec * nbins))
for name1, name2 in cwr(spec_names, 2):
    logging.debug(f"Loading analytic cov for {name1}, {name2}")
    sid1, sid2 = spec_names.index(name1), spec_names.index(name2)
    analytic_cov = np.load(os.path.join(cov_dir, f"analytic_cov_{name1}_{name2}.npy"))
    for spec1, spec2 in product(spectra, spectra):
        s1, s2 = spectra.index(spec1), spectra.index(spec2)
        sub_cov = analytic_cov[s1 * nbins : (s1 + 1) * nbins, s2 * nbins : (s2 + 1) * nbins]
        id_start_1 = sid1 * nbins + s1 * nspec * nbins
        id_stop_1 = (sid1 + 1) * nbins + s1 * nspec * nbins
        id_start_2 = sid2 * nbins + s2 * nspec * nbins
        id_stop_2 = (sid2 + 1) * nbins + s2 * nspec * nbins
        full_analytic_cov[id_start_1:id_stop_1, id_start_2:id_stop_2] = sub_cov

# We make the matrix symmetric and save it

transpose = full_analytic_cov.copy().T
transpose[full_analytic_cov != 0] = 0
full_analytic_cov += transpose
logging.info(f"Storing {full_analytic_cov.shape} covariance")
np.save(os.path.join(cov_dir, "full_analytic_cov.npy"), full_analytic_cov)

# For spectra with the same survey and the same array (sv1 = sv2 and ar1 = ar2) TE = ET
# therefore we remove the ET block in order for the covariance not to be redondant

block_to_delete = []
for sid, name in enumerate(spec_names):
    na, nb = name.split("x")
    for s, spec in enumerate(spectra):
        id_start = sid * nbins + s * nspec * nbins
        id_stop = (sid + 1) * nbins + s * nspec * nbins
        if na == nb and spec == "ET":
            block_to_delete = np.append(block_to_delete, np.arange(id_start, id_stop))

block_to_delete = block_to_delete.astype(int)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=1)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete, axis=0)

logging.info(f"Storing {full_analytic_cov.shape} (truncated) covariance")
np.save(os.path.join(cov_dir, "truncated_analytic_cov.npy"), full_analytic_cov)

# logging.debug(f"Is matrix positive definite: {data_analysis_utils.is_pos_def(full_analytic_cov)}")
# logging.debug(f"Is matrix symmetric: {data_analysis_utils.is_symmetric(full_analytic_cov)}")

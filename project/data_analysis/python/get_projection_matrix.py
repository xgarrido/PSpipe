# This script create projection matrix that will be used to combine all arrays, all survey spectra into a set of multifrequency spectra

import logging
import os
import sys
from itertools import combinations_with_replacement as cwr
from itertools import product

import data_analysis_utils
import numpy as np
import yaml
from pspy import pspy_utils, so_cov

d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)

data_dir = d["data_dir"]
product_dir = d["product_dir"]
cov_dir = os.path.join(product_dir, "covariances")

lmax = d["lmax"]
data_dir = d["data_dir"]
binning_file = os.path.join(data_dir, d["binning_file"])
bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

# First let's get a list of all frequencies we plan to study
frequencies = sorted(
    {
        freq
        for sv in d.get("selected_surveys")
        for freq in d.get("surveys").get(sv).get("frequencies").values()
    }
)
logging.debug(f"Frequencies: {frequencies} GHz")

# We make a list of all spectra included in the analysis
# We also make one of all spectra with same survey and same array
# This matter for the TE projector since for these spectra TE = ET and
# therefore the ET block have been removed from the cov matrix in order to avoid
# redondancy

# We make a list of all spectra included in the analysis
surveys = d.get("surveys")
svxars = [(sv, ar) for sv in d.get("selected_surveys") for ar in surveys.get(sv).get("arrays")]
comb = [c1 + c2 for c1, c2 in cwr(svxars, 2)]

# Now we need to compute the projection matrix
# This bit is extra complicated due to the fact that the TE block part is treated differently
# This happen for two reasons: we don't use any same array, same season ET (since ET = TE)
# and we need to keep T_\nu_1 E_\nu_2 and T_\nu_2 E_\nu_1 separated


# We start with a projector for the TT and EE block

n_freq = len(frequencies)
n_bins = len(bin_hi)
n_ps = len(comb)
n_cross_freq = n_freq * (n_freq + 1) // 2
Pmat = np.zeros((n_cross_freq * n_bins, n_ps * n_bins))

for c_id, cross_freq in enumerate(cwr(frequencies, 2)):
    for ps_id, (sv1, ar1, sv2, ar2) in enumerate(comb):
        f1 = surveys.get(sv1).get("frequencies").get(ar1)
        f2 = surveys.get(sv2).get("frequencies").get(ar2)
        if cross_freq in [(f1, f2), (f2, f1)]:
            id_start_cf = n_bins * c_id
            id_stop_cf = n_bins * (c_id + 1)
            id_start_n = n_bins * ps_id
            id_stop_n = n_bins * (ps_id + 1)
            Pmat[id_start_cf:id_stop_cf, id_start_n:id_stop_n] = np.identity(n_bins)

# Now we write a projector for the TE block
comb_same = [c1 + c2 for c1, c2 in cwr(svxars, 2) if c1 == c2]
comb_diff = [c1 + c2 for c1, c2 in cwr(svxars, 2) if c1 != c2]

n_cross_freq_TE = n_freq ** 2
n_ps_same = len(comb_same)
n_ps_TE = 2 * n_ps - n_ps_same
Pmat_TE = np.zeros((n_cross_freq_TE * n_bins, n_ps_TE * n_bins))

for c_id, cross_freq in enumerate(product(frequencies, frequencies)):
    for ps_id, (sv1, ar1, sv2, ar2) in enumerate(comb + comb_diff):
        f1 = surveys.get(sv1).get("frequencies").get(ar1)
        f2 = surveys.get(sv2).get("frequencies").get(ar2)
        spec_cf_list = (f1, f2)
        if ps_id >= n_ps:
            # we are in the ET block
            spec_cf_list = (f2, f1)
        if cross_freq == spec_cf_list:
            id_start_cf = n_bins * c_id
            id_stop_cf = n_bins * (c_id + 1)
            id_start_n = n_bins * ps_id
            id_stop_n = n_bins * (ps_id + 1)
            Pmat_TE[id_start_cf:id_stop_cf, id_start_n:id_stop_n] = np.identity(n_bins)

# We then glue together the TT - TE - EE projectors

shape_x = Pmat.shape[0]
shape_y = Pmat.shape[1]
shape_TE_x = Pmat_TE.shape[0]
shape_TE_y = Pmat_TE.shape[1]

full_Pmat = np.zeros((2 * shape_x + shape_TE_x, 2 * shape_y + shape_TE_y))
full_Pmat[:shape_x, :shape_y] = Pmat
full_Pmat[shape_x : shape_x + shape_TE_x, shape_y : shape_y + shape_TE_y] = Pmat_TE
full_Pmat[shape_x + shape_TE_x :, shape_y + shape_TE_y :] = Pmat

so_cov.plot_cov_matrix(full_Pmat, file_name=os.path.join(cov_dir, "P_mat"))
np.save(os.path.join(cov_dir, "projection_matrix.npy"), full_Pmat)

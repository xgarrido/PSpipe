import logging
import os
import sys
from itertools import combinations_with_replacement as cwr
from itertools import product

import data_analysis_utils
import numpy as np
import sacc
import yaml
from pspy import pspy_utils, so_mcm, so_spectra

d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)

product_dir = d["product_dir"]
spectra_dir = os.path.join(product_dir, "spectra")
cov_dir = os.path.join(product_dir, "covariances")
like_product_dir = os.path.join(product_dir, "like_product")
os.makedirs(like_product_dir, exist_ok=True)

# Lets read the data vector corresponding to the covariance matrix

surveys = d.get("surveys")
svxars = [(sv, ar) for sv in d.get("selected_surveys") for ar in surveys.get(sv).get("arrays")]
comb = [c1 + c2 for c1, c2 in cwr(svxars, 2)]

logging.info("Appending all spectra")
data_vec = []
for spec in ["TT", "TE", "ET", "EE"]:
    for sv1, ar1, sv2, ar2 in comb:
        # remove same array, same season ET
        if spec == "ET" and sv1 == sv2 and ar1 == ar2:
            continue

        spectra = d.get("spectra", ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"])
        cl_type = d.get("type")
        lb, Db = so_spectra.read_ps(
            os.path.join(spectra_dir, f"{cl_type}_{sv1}_{ar1}x{sv2}_{ar2}_cross.dat"),
            spectra=spectra,
        )

        data_vec = np.append(data_vec, Db[spec])

# Lets combine the data (following the doc)
# we will need the covariance matrix and the projection matrix
cov_mat = np.load(os.path.join(cov_dir, "truncated_analytic_cov.npy"))
P_mat = np.load(os.path.join(cov_dir, "projection_matrix.npy"))

logging.info("Invert covariance matrix")
inv_cov_mat = np.linalg.inv(cov_mat)

logging.info("Project covariance matrix & data vector")
proj_cov_mat = np.linalg.inv(np.dot(np.dot(P_mat, inv_cov_mat), P_mat.T))
proj_data_vec = np.dot(proj_cov_mat, np.dot(P_mat, np.dot(inv_cov_mat, data_vec)))

logging.info(f"Is matrix positive definite: {data_analysis_utils.is_pos_def(proj_cov_mat)}")
logging.info(f"Is matrix symmetric: {data_analysis_utils.is_symmetric(proj_cov_mat)}")

np.save(os.path.join(like_product_dir, "combined_analytic_cov.npy"), proj_cov_mat)
np.savetxt(os.path.join(like_product_dir, "data_vec.dat"), proj_data_vec)

# Saving into sacc format
act_sacc = sacc.Sacc()
logging.info("Add tracers...")

lmax = d["lmax"]
binning_file = os.path.join(d.get("data_dir"), d["binning_file"])
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)


frequencies = sorted(
    {
        freq
        for sv in d.get("selected_surveys")
        for freq in d.get("surveys").get(sv).get("frequencies").values()
    }
)
logging.debug(f"Frequencies: {frequencies} GHz")
for freq in frequencies:
    for spin, quantity in zip([0, 2], ["temperature", "polarization"]):
        logging.debug(f"Add spin-{spin} tracer for {freq} GHz")
        # dummies file: not in used
        data_beams = {"l": np.arange(10000), "bl": np.ones(10000)}

        act_sacc.add_tracer(
            "NuMap",
            f"ACT_{freq}_s{spin}",
            quantity=f"cmb_{quantity}",
            spin=spin,
            nu=[freq],
            bandpass=[1.0],
            ell=data_beams.get("l"),
            beam=data_beams.get("bl"),
        )

count = 0
for spec in ["TT", "TE", "EE"]:
    spec_frequencies = cwr(frequencies, 2) if spec != "TE" else product(frequencies, frequencies)
    for f1, f2 in spec_frequencies:
        logging.debug(f"Adding {f1}x{f2} GHz - {spec} spectra")
        # Set sacc tracer type and names
        pa, pb = spec
        ta_name = f"ACT_{f1}_s0" if pa == "T" else f"ACT_{f1}_s2"
        tb_name = f"ACT_{f2}_s0" if pb == "T" else f"ACT_{f2}_s2"

        map_types = {"T": "0", "E": "e", "B": "b"}
        if pb == "T":
            cl_type = "cl_" + map_types[pb] + map_types[pa]
        else:
            cl_type = "cl_" + map_types[pa] + map_types[pb]

        # Get Bbl
        mbb_inv, Bbl = so_mcm.read_coupling(
            os.path.join(product_dir, "mcms", f"s17_pa4_f150xs17_pa4_f150"),
            spin_pairs=["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"],
        )
        Bbl_TT = Bbl["spin0xspin0"]
        Bbl_TE = Bbl["spin0xspin2"]
        Bbl_EE = Bbl["spin2xspin2"][: Bbl_TE.shape[0], : Bbl_TE.shape[1]]

        if spec in ["EE", "EB", "BE", "BB"]:
            Bbl = Bbl_EE
        elif spec in ["TE", "TB", "ET", "BT"]:
            Bbl = Bbl_TE
        else:
            Bbl = Bbl_TT
        ls_w = np.arange(2, Bbl.shape[-1] + 2)
        bp_window = sacc.BandpowerWindow(ls_w, Bbl.T)

        # Add ell/cl to sacc
        n_bins = len(lb)
        Db = proj_data_vec[count * n_bins : (count + 1) * n_bins]
        act_sacc.add_ell_cl(cl_type, ta_name, tb_name, lb, Db, window=bp_window)

        sigmab = np.sqrt(proj_cov_mat.diagonal()[count * n_bins : (count + 1) * n_bins])
        np.savetxt(
            os.path.join(like_product_dir, f"spectra_{spec}_{f1}x{f2}.dat"),
            np.transpose([lb, Db, sigmab]),
        )

        count += 1

logging.info("Adding covariance")
act_sacc.add_covariance(proj_cov_mat)
logging.info("Writing sacc file")
act_sacc.save_fits(
    os.path.join(like_product_dir, f"act_{''.join(d.get('selected_surveys'))}_sacc.fits"),
    overwrite=True,
)

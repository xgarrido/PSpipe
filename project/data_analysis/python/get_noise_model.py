import logging
import os
import sys
from itertools import combinations_with_replacement as cwr

import numpy as np
import yaml
from pspy import pspy_utils, so_spectra
from scipy.interpolate import interp1d

d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)
data_dir = d["data_dir"]
product_dir = d["product_dir"]
spectra_dir = os.path.join(product_dir, "spectra")
ps_model_dir = os.path.join(product_dir, "noise_model")
os.makedirs(ps_model_dir, exist_ok=True)

spectra = d.get("spectra", ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"])
surveys = d["surveys"]
lmax = d["lmax"]
cl_type = d["type"]
binning_file = os.path.join(data_dir, d.get("binning_file"))

lth = np.arange(2, lmax + 2)

for sv in d.get("selected_surveys"):
    survey = surveys.get(sv)
    for ar1, ar2 in cwr(survey.get("arrays"), 2):
        beams = survey.get("beams")
        ell, bl_ar1 = pspy_utils.read_beam_file(os.path.join(data_dir, beams.get(ar1)))
        ell, bl_ar2 = pspy_utils.read_beam_file(os.path.join(data_dir, beams.get(ar2)))

        lb, bb_ar1 = pspy_utils.naive_binning(ell, bl_ar1, binning_file, lmax)
        lb, bb_ar2 = pspy_utils.naive_binning(ell, bl_ar2, binning_file, lmax)

        spec_name_noise = f"{cl_type}_{sv}_{ar1}x{sv}_{ar2}_noise"
        logging.debug(f"Reading {spec_name_noise} spectra")
        lb, nbs = so_spectra.read_ps(
            os.path.join(spectra_dir, f"{spec_name_noise}.dat"), spectra=spectra
        )

        nl_dict = {}
        for spec in spectra:
            nbs_mean = nbs[spec] * bb_ar1 * bb_ar2

            if spec in ["TT", "EE", "BB"] and ar1 == ar2:
                nl = interp1d(lb, nbs_mean, fill_value="extrapolate")
                nl_dict[spec] = np.array([nl(i) for i in lth])
                idx = np.where(lth <= np.min(lb))
                nl_dict[spec][idx] = nbs_mean[0]
                nl_dict[spec] = np.abs(nl_dict[spec])
            else:
                nl_dict[spec] = np.zeros(len(lth))

        so_spectra.write_ps(
            os.path.join(ps_model_dir, f"mean_{ar1}x{ar2}_{sv}_noise.dat"),
            lth,
            nl_dict,
            cl_type,
            spectra=spectra,
        )

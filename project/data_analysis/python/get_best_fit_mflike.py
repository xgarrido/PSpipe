#
# This script compute best fit from theory and fg power spectra.
# It uses camb and the foreground model of mflike based on fgspectra
#
import logging
import os
import sys
from itertools import product

import numpy as np
import yaml

d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)

data_dir = d["data_dir"]
product_dir = d["product_dir"]
best_fits_dir = os.path.join(product_dir, "best_fits")
os.makedirs(best_fits_dir, exist_ok=True)

frequencies = sorted(
    {
        freq
        for sv in d.get("selected_surveys")
        for freq in d.get("surveys").get(sv).get("frequencies").values()
    }
)
logging.debug(f"Frequencies: {frequencies} GHz")

# Now we use camb to produce best fit power spectrum, we will use CAMB to do so with standard LCDM
# params
import camb

ell_min, ell_max = 2, d.get("lmax") + 500
bf_parameters = d.get("best_fit_parameters")
cosmo_params = bf_parameters.get("cosmo_params")
camb_cosmo = {k: v for k, v in cosmo_params.items() if k not in ["logA", "As"]}
camb_cosmo.update(
    {"As": 1e-10 * np.exp(cosmo_params["logA"]), "lmax": ell_max, "lens_potential_accuracy": 1}
)
logging.debug(f"Generating CAMB simulations with the following parameters: {camb_cosmo}")
pars = camb.set_params(**camb_cosmo)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

clth = {
    spec: powers["total"][ell_min:ell_max][:, i] for i, spec in enumerate(["TT", "EE", "BB", "TE"])
}
clth["ET"] = clth["TE"]
clth.update({spec: np.zeros_like(clth["TT"]) for spec in ["TB", "BT", "EB", "BE"]})
logging.debug(f"Simulated power spectra {clth}")

ell = np.arange(ell_min, ell_max)
np.savetxt(
    os.path.join(best_fits_dir, "lcdm.dat"),
    np.transpose([ell, clth["TT"], clth["EE"], clth["BB"], clth["TE"]]),
)

# We will now use mflike (and in particular the fg module) to get the best fit foreground model
# We will only include foreground in tt, note that for now only extragalactic foreground are present
import mflike

fg_config = bf_parameters.get("foregrounds")
fg_dict = mflike.get_foreground_model(
    fg_params=fg_config.get("params"),
    fg_model=dict(
        normalisation=fg_config.get("norm"),
        components=dict(tt=fg_config.get("components"), ee=[], te=[]),
    ),
    frequencies=frequencies,
    ell=ell,
)

spectra = d.get("spectra", ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"])
for spec, freq1, freq2 in product(spectra, frequencies, frequencies):
    name = f"{freq1}x{freq2}_{spec}"
    cl_th_and_fg = clth[spec]

    if spec == "TT":
        cl_th_and_fg += fg_dict["tt", "all", freq1, freq2]
        np.savetxt(
            os.path.join(best_fits_dir, f"fg_{name}.dat"),
            np.transpose([ell, fg_dict["tt", "all", freq1, freq2]]),
        )

    np.savetxt(
        os.path.join(best_fits_dir, f"best_fit_{name}.dat"), np.transpose([ell, cl_th_and_fg])
    )

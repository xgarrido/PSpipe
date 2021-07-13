# This script compute all power spectra and write them to disk.
# It uses the window function provided in the dictionnary file.
# Optionally, it applies a calibration to the maps and a kspace filter.
# The spectra are then combined in mean auto, cross and noise power spectrum and written to disk.
# If write_all_spectra=True, each individual spectrum is written to disk.

import logging
import os
import sys
from itertools import combinations_with_replacement as cwr
from itertools import product

import data_analysis_utils
import numpy as np
import yaml
from pixell import enmap
from pspy import pspy_utils, so_map, so_mcm, so_mpi, so_spectra, sph_tools

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
alms_dir = os.path.join(product_dir, "alms")
os.makedirs(alms_dir, exist_ok=True)
spectra_dir = os.path.join(product_dir, "spectra")
os.makedirs(spectra_dir, exist_ok=True)

lmax = d.get("lmax")
niter = d.get("niter", 0)
cl_type = d.get("type", "Dl")
spectra = d.get("spectra", ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"])
binning_file = os.path.join(data_dir, d.get("binning_file"))
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

so_mpi.init(True)

surveys = d.get("surveys")
svxars = [(sv, ar) for sv in d.get("selected_surveys") for ar in surveys.get(sv).get("arrays")]
if so_mpi.rank == 0:
    logging.info(f"Number of cross to compute : {len(svxars)}")

subtasks = so_mpi.taskrange(imin=0, imax=len(svxars) - 1)
for task in subtasks:
    task = int(task)
    sv, ar = svxars[task]

    logging.info(f"[{task:>2}] Compute alms for survey {sv}, array {ar}...")
    win = so_map.read_map(os.path.join(windows_dir, f"window_{sv}_{ar}.fits"))
    window_tuple = (win, win)

    survey = surveys.get(sv)
    maps = survey.get("maps").get(ar)

    inv_pixwin_lxly = None
    if survey.get("deconvolve_pixwin", True):
        wy, wx = enmap.calc_window(win.data.shape)
        inv_pixwin_lxly = (wy[:, None] * wx[None, :]) ** (-1)

    logging.debug(f"{len(maps)} splits for survey {sv}, array {ar}")

    alms = []
    for map_name in maps:
        split = so_map.read_map(os.path.join(data_dir, map_name))

        if survey.get("src_free_maps", True):
            logging.debug(f"Coadd point sources to {os.path.basename(map_name)}")
            model_map_name = map_name.replace("srcfree.fits", "model.fits")
            if model_map_name == map_name:
                raise ValueError("No model map is provided! Check map names!")
            point_source_map = so_map.read_map(os.path.join(data_dir, model_map_name))
            point_source_mask = so_map.read_map(
                os.path.join(data_dir, d.get("window_parameters").get("ps_mask"))
            )
            split = data_analysis_utils.get_coadded_map(split, point_source_map, point_source_mask)

        if survey.get("use_kspace_filter", True):
            logging.debug(f"Apply kspace filter on {os.path.basename(map_name)}")
            binary = so_map.read_map(os.path.join(windows_dir, f"binary_{sv}_{ar}.fits"))
            split = data_analysis_utils.get_filtered_map(
                split,
                binary,
                vk_mask=survey["vk_mask"],
                hk_mask=survey["hk_mask"],
                normalize=False,
                inv_pixwin_lxly=inv_pixwin_lxly,
            )

        split.data *= survey.get("calibrations").get(ar, 1.0)

        if d.get("remove_mean", False):
            split = data_analysis_utils.remove_mean(split, window_tuple)

        logging.debug(f"Compute alms for split {maps.index(map_name)}")
        alms += [sph_tools.get_alms(split, window_tuple, niter, lmax)]
        if survey.get("use_kspace_filter", True):
            # There is an extra normalisation for the FFT/IFFT bit note that we apply it here rather
            # than at the FFT level because correcting the alm is faster than correcting the
            # maps. Make sure it works even for only temperature map (so the use of np.product)
            alms[-1] /= np.product(split.data.shape[-2:])

    np.save(os.path.join(alms_dir, f"alms_{sv}_{ar}"), alms)

# Wait for all subtasks to be done
so_mpi.comm.Barrier()

if so_mpi.rank == 0:
    logging.debug("alms computation done. Now gathering all the alms...")
    master_alms = {
        (sv, ar): np.load(os.path.join(alms_dir, f"alms_{sv}_{ar}.npy")) for sv, ar in svxars
    }

    comb = [c1 + c2 for c1, c2 in cwr(svxars, 2)]
    logging.info(f"Number of cross-spectra to compute : {len(comb)}")

    for sv1, ar1, sv2, ar2 in comb:
        logging.info(
            f"Compute spectra for {sv1}_{ar1}x{sv2}_{ar2} "
            f"({comb.index((sv1, ar1, sv2, ar2))}/{len(comb)})..."
        )

        # Get transfer function
        def get_transfer_function(sv):
            tf = np.ones(len(lb))
            tf_file = surveys.get(sv).get("transfer_function")
            if tf_file is not None:
                logging.debug(f"Using transfer function for {sv}")
                _, _, tf, _ = np.loadtxt(os.path.join(data_dir, tf_file), unpack=True)
            if surveys.get(sv).get("deconvolve_pixwin", True):
                logging.debug(f"Deconvolve pixel window function for {sv}")
                pixwin = hp.pixwin(2048) if sv.lower() == "planck" else np.ones(2 * lmax)
                _, pw = pspy_utils.naive_binning(np.arange(len(pixwin)), pixwin, binning_file, lmax)
                tf *= pw
            return tf

        tf1 = get_transfer_function(sv1)
        tf2 = get_transfer_function(sv2)

        ps_dict = {(spec, stype): [] for spec, stype in product(spectra, ["auto", "cross"])}
        nsplit1 = len(master_alms[sv1, ar1])
        nsplit2 = len(master_alms[sv2, ar2])
        for s1, s2 in product(range(nsplit1), range(nsplit2)):
            if sv1 == sv2 and ar1 == ar2 and s1 > s2:
                continue

            mbb_inv, Bbl = so_mcm.read_coupling(
                prefix=os.path.join(mcms_dir, f"{sv1}_{ar1}x{sv2}_{ar2}"),
                spin_pairs=["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"],
            )

            logging.debug(f"Compute spectra for {sv1}_{ar1}x{sv2}_{ar2}, splits {s1}x{s2}")
            ell, ps_master = so_spectra.get_spectra_pixell(
                master_alms[sv1, ar1][s1], master_alms[sv2, ar2][s2], spectra=spectra
            )
            lb, ps = so_spectra.bin_spectra(
                ell,
                ps_master,
                binning_file=binning_file,
                lmax=lmax,
                type=cl_type,
                mbb_inv=mbb_inv,
                spectra=spectra,
            )

            # Deconvolve TF
            data_analysis_utils.deconvolve_tf(lb, ps, tf1, tf2, lmax)

            if d.get("write_splits_spectra", False):
                spec_name = os.path.join(
                    spectra_dir, f"{cl_type}_{sv1}_{ar1}x{sv2}_{ar2}_{s1}{s2}.dat"
                )
                so_spectra.write_ps(spec_name, lb, ps, cl_type, spectra=spectra)

            for spec in spectra:
                stype = "auto" if s1 == s2 and sv1 == sv2 else "cross"
                ps_dict[spec, stype] += [ps[spec]]

        ps_dict_cross_mean = {}
        ps_dict_auto_mean = {}
        ps_dict_noise_mean = {}
        for spec in spectra:
            ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)

            if sv1 == sv2:
                if ar1 == ar2:
                    # Average TE / ET so that for same array same season TE = ET
                    ps_dict_cross_mean[spec] = (
                        np.mean(ps_dict[spec, "cross"], axis=0)
                        + np.mean(ps_dict[spec[::-1], "cross"], axis=0)
                    ) / 2.0
                ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                ps_dict_noise_mean[spec] = (
                    ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]
                ) / nsplit1

        spec_name = f"{cl_type}_{sv1}_{ar1}x{sv2}_{ar2}"
        so_spectra.write_ps(
            os.path.join(spectra_dir, f"{spec_name}_cross.dat"),
            lb,
            ps_dict_cross_mean,
            cl_type,
            spectra=spectra,
        )

        if sv1 == sv2:
            so_spectra.write_ps(
                os.path.join(spectra_dir, f"{spec_name}_auto.dat"),
                lb,
                ps_dict_auto_mean,
                cl_type,
                spectra=spectra,
            )
            so_spectra.write_ps(
                os.path.join(spectra_dir, f"{spec_name}_noise.dat"),
                lb,
                ps_dict_noise_mean,
                cl_type,
                spectra=spectra,
            )

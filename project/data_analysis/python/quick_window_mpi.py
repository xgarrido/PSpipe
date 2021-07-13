# This script create the window functions used in the PS computation They consist of a point source
# mask, a galactic mask and a mask based on the amount of cross linking in the data The different
# masks are apodized.  We also produce a binary mask that will later be used for the kspace
# filtering operation, in order to remove the edges and avoid nasty pixels before this not so well
# defined Fourier operation.

import logging
import os
import sys
from itertools import product

import numpy as np
import yaml
from pspy import pspy_utils, so_map, so_mpi, so_window


def create_crosslink_mask(xlink_map, cross_link_threshold):
    # remove pixels with very little amount of cross linking
    xlink = so_map.read_map(xlink_map)
    xlink_lowres = xlink.downgrade(32)
    with np.errstate(invalid="ignore"):
        x_mask = (
            np.sqrt(xlink_lowres.data[1] ** 2 + xlink_lowres.data[2] ** 2) / xlink_lowres.data[0]
        )
    x_mask[np.isnan(x_mask)] = 1
    x_mask[x_mask >= cross_link_threshold] = 1
    x_mask[x_mask < cross_link_threshold] = 0
    x_mask = 1 - x_mask
    xlink_lowres.data[0] = x_mask
    xlink = so_map.car2car(xlink_lowres, xlink)
    x_mask = xlink.data[0].copy()
    idx = np.where(x_mask > 0.9)
    x_mask[:] = 0
    x_mask[idx] = 1
    return x_mask


d = yaml.safe_load(open(sys.argv[1]))

logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG if d.get("debug", False) else logging.INFO,
)

data_dir = d["data_dir"]
windows_dir = os.path.join(d["product_dir"], "windows")
os.makedirs(windows_dir, exist_ok=True)

# Window parameters
window_parameters = d["window_parameters"]
# the apodisation lenght of the point source mask in degree
apod_pts_source_degree = window_parameters.get("apod_pts_source_degree", 0.3)
# the apodisation lenght of the survey x gal x cross linking mask
apod_survey_degree = window_parameters.get("apod_survey_degree", 2)
# we will skip the edges of the survey where the noise is very difficult to model
skip_from_edges_degree = window_parameters.get("skip_from_edges_degree", 1)
# the threshold on the amount of cross linking to keep the data in
use_cross_link = window_parameters.get("use_cross_link", False)
cross_link_threshold = window_parameters.get("cross_link_threshold", 0.97)

ps_mask = so_map.read_map(os.path.join(data_dir, window_parameters["ps_mask"]))
gal_mask = so_map.read_map(os.path.join(data_dir, window_parameters["gal_mask"]))

patch = None
if "patch" in window_parameters:
    logging.info(f"Using '{window_parameters['patch']}' as patch file")
    patch = so_map.read_map(os.path.join(data_dir, window_parameters["patch"]))


# here we list the different windows that need to be computed,
# we will then do a MPI loops over this list
surveys = d.get("surveys")
comb = [
    comb
    for sv in d.get("selected_surveys")
    for comb in product([sv], surveys.get(sv).get("arrays"))
]

logging.info(f"Number of windows to compute : {len(comb)}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=len(comb) - 1)

for task in subtasks:
    task = int(task)
    sv, ar = comb[task]
    logging.debug(f"[{task}] Processing '{sv}' survey, array '{ar}'...")

    survey = surveys.get(sv)
    survey_mask_file = survey.get("survey_mask")
    if survey_mask_file is not None:
        survey_mask = so_map.read_map(os.path.join(data_dir, survey_mask_file))
    else:
        survey_mask = gal_mask.copy()
        survey_mask.data[:] = 1

        maps = survey.get("maps").get(ar)
        for map_name in maps:
            logging.debug(f"Loading {map_name} map...")
            m = so_map.read_map(os.path.join(data_dir, map_name))
            survey_mask.data[m.data[0] == 0.0] = 0

            if use_cross_link:
                logging.debug("Creating xlink mask...")
                if survey.get("src_free_maps", True):
                    xlink_map_name = map_name.replace("map_srcfree.fits", "xlink.fits")
                else:
                    xlink_map_name = map_name.replace("map.fits", "xlink.fits")
                if xlink_map_name == map_name:
                    raise ValueError("No xlink map is provided! Check map names!")
                x_mask = create_crosslink_mask(
                    os.path.join(data_dir, xlink_map_name), cross_link_threshold
                )
                survey_mask.data *= x_mask

    survey_mask.data *= gal_mask.data
    if patch is not None:
        survey_mask.data *= patch.data

    # so here we create a binary mask this will only be used in order to skip the edges before
    # applying the kspace filter this step is a bit arbitrary and preliminary, more work to be done
    # here
    binary = survey_mask.copy()

    # Note that we don't skip the edges as much for the binary mask compared to what we will do with
    # the final window, this should prevent some aliasing from the kspace filter to enter the data
    dist = so_window.get_distance(survey_mask, rmax=np.deg2rad(apod_survey_degree))
    binary.data[dist.data < skip_from_edges_degree / 2] = 0
    binary.write_map(os.path.join(windows_dir, f"binary_{sv}_{ar}.fits"))
    binary = binary.downgrade(4)
    binary.plot(file_name=os.path.join(windows_dir, f"binary_{sv}_{ar}"))

    # Now we create the final window function that will be used in the analysis
    survey_mask.data[dist.data < skip_from_edges_degree] = 0
    survey_mask = so_window.create_apodization(survey_mask, "C1", apod_survey_degree, use_rmax=True)
    ps_mask = so_window.create_apodization(ps_mask, "C1", apod_pts_source_degree, use_rmax=True)
    survey_mask.data *= ps_mask.data

    logging.debug("Writing window...")
    survey_mask.write_map(os.path.join(windows_dir, f"window_{sv}_{ar}.fits"))
    survey_mask = survey_mask.downgrade(4)
    survey_mask.plot(file_name=os.path.join(windows_dir, f"window_{sv}_{ar}"))

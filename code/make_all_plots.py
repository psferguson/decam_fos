import fitsio
import yaml 
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import sys
sys.path.append("../code/")
import fos_preparation 
import fos_plotting
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]




if __name__ == "__main__":
    # this should be run from the code folder
    rgb_decals = np.load("../data/hpx_maps/decals_dr10_corrected_rgb_nside_512_smooth.npy")
    rgb_des= np.load("../data/hpx_maps/des_y6_gold_rgb_stack_nside_512.npy")
    gaia_data = fos_preparation.load_gaia_density_data("../data/hpx_maps/gaia_dr3_density_nside_512_nest.fits")
    mask_arr = fitsio.read("../data/hpx_maps/decam_mask_nside_512.fits")
    with open("../data/annotations/stream_arrow_annotations.yaml", "r") as f:
        annotations_dict = yaml.safe_load(f)

    annotations_to_plot = [
        '300S',
        'AAU (Aliqa Uma)',
        'AAU (Atlas)',
        'Elqui',
        'Indus',
        #'Jhelum',
        'Jet',
        'OC (Chenab)',
        'OC (Orphan)',
        'Pal 13',
        'Pal 5',
        'Palca',
        'Phoenix',
        'Sagittarius',
        'Tri And',
        'Tuc III',
        'Turranburra',
        'Willka\nYaku'
    ]
    annotations_dict = {stream_name : annotations_dict[stream_name] for stream_name in annotations_to_plot}


    for white_background in [True, False]:
        print(f"white_background = {white_background}")
        for proj_string in ["ortho", "moll"]:
            print(f"proj = {proj_string}")
            if proj_string == "ortho":
                proj = hp.projector.OrthographicProj(rot=[30, -20, 0])
                annotations_bool = [False]
            if proj_string == "moll":
                proj = hp.projector.MollweideProj()
                annotations_bool = [True, False]
            
            image, proj = fos_preparation.assemble_image(rgb_des, rgb_decals, 
                                                         mask_arr, gaia_data, do_diameter_closing=True, 
                                                         white_background=white_background, 
                                                         proj=proj)
            for annotations in annotations_bool:
                print(f"annotations = {annotations}")
                fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=175)
        
                fos_plotting.plot_fos(
                    ax, 
                    image, 
                    proj,
                    DES=True, 
                    LSST=True,
                    annotations=annotations,
                    oval=True,
                    white_background=white_background,
                    annotations_dict=annotations_dict)
                fig.tight_layout()

                if white_background:
                    facecolor = "white"
                else:
                    facecolor = "k"
                plt.savefig(f"../plots/fos_{proj_string}_{annotations}_{white_background}.png", 
                            bbox_inches="tight", facecolor=facecolor)
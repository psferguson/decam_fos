import numpy as np
import fitsio
import healpy as hp
from skimage.morphology import diameter_closing


def load_gaia_density_data(
        filepath="../data/hpx_maps/gaia_dr3_density_nside_512_nest.fits"
):
    """Load Gaia density data from a FITS file and normalize it."""
    gaia_data = fitsio.read(filepath)
    gaia_data = hp.reorder(gaia_data, n2r=True)
    gaia_m = np.ma.masked_invalid(gaia_data).astype(float)
    gaia_m /= np.percentile(gaia_m.filled(), 99)
    gaia_m = np.clip(gaia_m, 0, 1)
    return gaia_m


def clip_norm(x, min=None, max=None):
    """
    Normalize an array to the range [0, 1] with optional clipping."
    """
    if min is None:
        min = np.nanmin(x[~np.isinf(x)])
    if max is None:
        max = np.nanmax(x[~np.isinf(x)])
    return np.clip((x - min) / (max - min), 0, 1)


def renorm_rgb(rgb, pmin=1, pmax=99):
    """"
    Renormalize RGB channels using percentiles for clipping."
    """
    for i in range(3):
        X = rgb[i]

        amin = np.nanpercentile(np.ma.filled(X, np.nan), pmin)
        amax = np.nanpercentile(np.ma.filled(X, np.nan), pmax)

        rgb[i] = clip_norm(X, min=amin, max=amax)
        rgb[i][X == 0] = 0.

    return rgb   


def get_image(proj, rgb, do_mask_replace=True, mask_arr=None, 
              mask_replace_arr=None, pmin=0.2, pmax=93):
    rgb = np.ma.copy(rgb)
    rgb = renorm_rgb(rgb, pmin=pmin, pmax=pmax)
    if do_mask_replace:
        if mask_arr is None:
            raise ValueError(
                "mask_arr must be provided if do_mask_replace is True"
                )
        if mask_replace_arr is None:
            raise ValueError(
                "mask_replace must be provided if do_mask_replace is True"
                )
        for i in range(3):
            rgb[i][mask_arr == 1] = mask_replace_arr[mask_arr == 1]

    image = np.stack([proj.projmap(rgb[i].filled(np.nan), vec2pix) 
                     for i in range(3)], axis=-1)

    return image


def create_base_image(proj, white_background=True):
    array = np.arange(hp.nside2npix(512)) * 0
    vpix = proj.projmap(array, vec2pix)
    if white_background:
        bg_value = 1
    else:
        bg_value = 0
    
    bg_arr = np.ones_like(proj.projmap(array, vec2pix)).astype(float) 
    bg_arr *= bg_value
    bg_arr[~np.isinf(vpix)] = 0
    bg_image = np.stack([bg_arr for i in range(3)], axis=-1)
    return bg_image


def stack_image(image_list, proj, white_background=True):
    image = create_base_image(proj, white_background=white_background)
    for im in image_list:
        idx = np.isfinite(im) & ~np.isnan(im) & (im > 0)
        image[idx] = im[idx]
    return image


def func_vec2pix(nside):
    return lambda x, y, z: hp.vec2pix(nside, x, y, z)


vec2pix = func_vec2pix(nside=512)


def assemble_image(rgb_des, rgb_decals, mask_arr, gaia_data, proj=None, 
                   do_diameter_closing=True, white_background=True):
    if proj is None:
        proj = hp.projector.MollweideProj() 
    image_des = get_image(proj, rgb_des, mask_arr=mask_arr, 
                          mask_replace_arr=gaia_data)
    image_decals = get_image(proj, rgb_decals, mask_arr=mask_arr,
                             mask_replace_arr=gaia_data)
    image = stack_image([image_decals, image_des], proj,
                        white_background=white_background)

    if do_diameter_closing:
        print("Applying diameter closing to the image...")
        # Initialize output image
        closed_image = np.zeros_like(image)

        # Apply diameter closing to each channel
        for channel in range(3):  # R, G, B
            closed_image[..., channel] = diameter_closing(
                image[..., channel],
                diameter_threshold=10,  
                connectivity=2         
            )
        image = closed_image
    return image, proj
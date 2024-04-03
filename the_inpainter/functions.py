"""
Functions that I (Harjas Sandhu) have created in the process of this research project
"""

import numpy as np
import healpy as hp

def loc2data(map_array: np.ndarray, loc: list, circ_rad: int, cutout_rad:
             int, side_len: int = 0, range_max: int = 0, mollview: bool = False) -> np.ndarray:
    '''
    Plot and/or return data around given locations on a map.
    
    Args:
        map_array: 1d numpy array of pixels in healpy Ring format.
        loc: a longitude and latitude location on the map in radians.
        circ_rad: radius of the disk around the location to be considered.
        cutout_rad: radius of the pixels within the circle to be set to 0.
        side_len (optional): length of one side of the returned 2d array/image.
        range_max (optional): max range for healpy mollview/gnomview.
        mollview: if True, also show mollview projection.

    Returns:
        data_2d: 2d numpy array of pixels returned by healpy.gnomview.
    '''
    nside = hp.get_nside(map_array)

    vec_rad = loc
    vec_deg = np.rad2deg(vec_rad)
    vec_3d = hp.ang2vec(*vec_deg, lonlat=True)

    ipix_disc = hp.query_disc(nside=nside, vec=vec_3d, radius=circ_rad)
    subdisc = hp.query_disc(nside=nside, vec=vec_3d, radius=cutout_rad)

    submap = np.zeros(hp.nside2npix(nside=nside))
    disc_values = map_array[ipix_disc]
    subdisc_values = map_array[subdisc]
    submap[ipix_disc] = disc_values
    submap[subdisc] = 0
    if range_max == 0:
        range_max = max(subdisc_values)

    if mollview:
        hp.mollview(
            submap,
            title="Submap Mollweide Projection",
            unit="mK",
            max=range_max
        )

    if side_len == 0:
        side_len = circ_rad * 4750 # magic number found with brute force.
    data_2d = hp.gnomview(
        submap,
        rot=vec_deg,
        xsize=side_len,
        title="Submap Gnomonic Projection (zoomed in)",
        unit="mK",
        max=range_max,
        return_projected_map=True
    )

    return data_2d

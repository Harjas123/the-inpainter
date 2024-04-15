"""
Functions that I (Harjas Sandhu) have created in the process of this research project
"""

import numpy as np
import healpy as hp

def get_rand_locs(num_locs, lon_range: tuple = (-np.pi, np.pi),
                  lat_range: tuple = (-np.pi/2, np.pi/2)) -> list:
    '''
    Returns a list of random locations in the form [longitude, latitude].
    
    Args:
        num_locs: number of locations.
        lon_range (optional): min and max values of longitude. Default is from -pi/2 to pi/2.
        lat_range (optional): min and max values of latitude. Default is from -pi/2 to pi/2.

    Returns:
        loc_list: list containing locations in the form [longitude, latitude].
    '''

    lon_diff = lon_range[1] - lon_range[0]
    lat_diff = lat_range[1] - lat_range[0]
    loc_array = np.random.rand(2, num_locs)
    loc_array[0] = loc_array[0] * lon_diff - lon_diff / 2
    loc_array[1] = loc_array[1] * lat_diff - lat_diff / 2
    loc_list = loc_array.T.tolist()
    return loc_list

def loc2data(map_array: np.ndarray, loc: list[float], circ_rad: float, cutout_rad: float,
             side_len: int = 0, range_max: int = 0, show_mollview: bool = False, show_gnomview: bool = False,
             units: str = "mK") -> tuple[np.ndarray, float, float]:
    '''
    Plots and/or returns data around a location on a map.
    
    Args:
        map_array: 1d numpy array of pixels in healpy Ring format.
        loc: a longitude and latitude location on the map in radians.
        circ_rad: radius of the disk around the location to be considered.
        cutout_rad: radius of the pixels within the circle to be set to 0.
        side_len (optional): length of one side of the returned 2d array/image.
            If not given, will attempt to include entire circle.
        range_max (optional): max pixel value for healpy mollview/gnomview projection.
            If not given, will use 95th percentile pixel value of main disk.
        show_mollview (optional): if True, show mollweide projection. Default False.
        show_mollview (optional): if True, show gnomonic projection. Default False.
        units (optional): units for map projection. Default is "mK".

    Returns:
        data_2d: 2d numpy array of pixels returned by healpy.gnomview.
        annulus_average: average value of pixels in circle (excluding cutout)
        actual_average: average value of pixels in cutout

    '''
    nside = hp.get_nside(map_array)

    loc_deg = np.rad2deg(loc)
    loc_3d = hp.ang2vec(*loc_deg, lonlat=True)

    ipix_disc = hp.query_disc(nside=nside, vec=loc_3d, radius=circ_rad)
    subdisc = hp.query_disc(nside=nside, vec=loc_3d, radius=cutout_rad)

    submap = np.zeros(len(map_array))
    submap[ipix_disc] = map_array[ipix_disc]
    submap[subdisc] = 0

    if range_max == 0:
        range_max = np.percentile(map_array[ipix_disc], 95)
    if show_mollview:
        hp.mollview(
            submap,
            title="Submap Mollweide Projection",
            unit=units,
            max=range_max
        )

    if side_len == 0:
        side_len = int(circ_rad * 4750) # magic number found with brute force.
    data_2d = hp.gnomview(
        submap,
        rot=loc_deg,
        xsize=side_len,
        title="Submap Gnomonic Projection",
        unit=units,
        max=range_max,
        return_projected_map=True,
        no_plot=(not show_gnomview)
    )

    submap[subdisc] = np.nan
    annulus_average: float = np.nanmean(submap[ipix_disc])
    actual_average: float = np.nanmean(map_array[subdisc])
    return (data_2d, annulus_average, actual_average)

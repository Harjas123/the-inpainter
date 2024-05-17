"""
Functions that I (Harjas Sandhu) have created in the process of this research project
"""

import numpy as np
import healpy as hp

def get_rand_locs(num_locs, lon_range: tuple = (-np.pi, np.pi),
                  lat_range: tuple = (-np.pi/2, np.pi/2), sphere_distr: bool = True) -> list:
    '''
    Returns a list of random locations in the form [longitude, latitude].
    
    Args:
        num_locs: number of locations.
        lon_range (optional): min and max values of longitude. Default is from -pi to pi.
        lat_range (optional): min and max values of latitude. Default is from -pi/2 to pi/2.
        sphere_distr (optional): if True, latitude values will be random on a sphere. Default True.

    Returns:
        loc_list: list containing locations in the form [longitude, latitude].
    '''

    lon_diff = lon_range[1] - lon_range[0]
    lat_diff = lat_range[1] - lat_range[0]
    loc_array = np.random.rand(2, num_locs)
    loc_array[0] = loc_array[0] * lon_diff - lon_diff / 2
    if sphere_distr:
        loc_array[1] = (np.arcsin(loc_array[1] * 2 - 1) / np.pi + 0.5) * lat_diff - lat_diff / 2
    else:
        loc_array[1] = loc_array[1] * lat_diff - lat_diff / 2
    loc_list = loc_array.T.tolist()
    return loc_list

def loc2data(map_array: np.ndarray, loc: list[float], circ_rad: float, cutout_rad: float,
             side_len: int = 0, range_max: int = 0, show_mollview: bool = False, show_gnomview: bool = False,
             units: str = "mK") -> tuple[np.ndarray, np.ndarray]:
    '''
    Plots and/or returns data around a location on a map.
    
    Args:
        map_array: 1d numpy array of pixels in healpy Ring format.
        loc: a longitude and latitude location on the map in radians.
        circ_rad: radius of the disk around the location to be considered.
        cutout_rad: radius of the pixels within the circle to be set to 0.
                    If < 0, will only cut out a single pixel.
        side_len (optional): length of one side of the returned 2d array/image.
            If not given, will attempt to include entire circle.
        range_max (optional): max pixel value for healpy mollview/gnomview projection.
            If not given, will use 95th percentile pixel value of main disk.
        show_mollview (optional): if True, show mollweide projection. Default False.
        show_gnomivew (optional): if True, show gnomonic projection. Default False.
        units (optional): units for map projection. Default is "mK".

    Returns:
        data_2d: 2d numpy array of pixels returned by healpy.gnomview.
    '''
    nside = hp.get_nside(map_array)

    loc_deg = np.rad2deg(loc)
    loc_3d = hp.ang2vec(*loc_deg, lonlat=True)

    ipix_disc = hp.query_disc(nside=nside, vec=loc_3d, radius=circ_rad)
    if cutout_rad >= 0:
        subdisc = hp.query_disc(nside=nside, vec=loc_3d, radius=cutout_rad)
    else:
        subdisc = hp.ang2pix(nside, *loc_deg, lonlat=True)

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
    annulus_2d = hp.gnomview(
        submap,
        rot=loc_deg,
        xsize=side_len,
        title="Submap Gnomonic Projection",
        unit=units,
        max=range_max,
        return_projected_map=True,
        no_plot=(not show_gnomview)
    )

    submap[ipix_disc] = 0
    submap[subdisc] = map_array[subdisc]
    actual_2d = hp.gnomview(
        submap,
        rot=loc_deg,
        xsize=side_len,
        title="Submap Gnomonic Projection",
        unit=units,
        max=range_max,
        return_projected_map=True,
        no_plot=(not show_gnomview)
    )

    return (annulus_2d, actual_2d)

def loc2annulus(map_array: np.ndarray, loc: list[float], circ_rad: float,
                  cutout_rad: float) -> tuple[np.ndarray, np.ndarray]:
    '''
    Returns annulus average and actual average of pixel values in a location on a map.
    
    Args:
        map_array: 1d numpy array of pixels in healpy Ring format.
        loc: a longitude and latitude location on the map in radians.
        circ_rad: radius of the disk around the location to be considered.
        cutout_rad: radius of the pixels within the circle to be set to 0.
                    If < 0, will only cut out a single pixel.

    Returns:
        annulus: pixels in circle, with cutout pixels set to np.nan
        actual: pixels in cutout

    '''
    nside = hp.get_nside(map_array)

    loc_deg = np.rad2deg(loc)
    loc_3d = hp.ang2vec(*loc_deg, lonlat=True)

    ipix_disc = hp.query_disc(nside=nside, vec=loc_3d, radius=circ_rad)
    if cutout_rad >= 0:
        subdisc = hp.query_disc(nside=nside, vec=loc_3d, radius=cutout_rad)
        actual = map_array[subdisc]
    else:
        subdisc = hp.ang2pix(nside, *loc_deg, lonlat=True)
        actual = np.array(map_array[subdisc])

    submap = np.zeros(len(map_array))
    submap[ipix_disc] = map_array[ipix_disc]
    submap[subdisc] = np.nan

    annulus = submap[ipix_disc]

    return (annulus, actual)

# SAME AS loc2annulus EXCEPT IT RETURNS THE AVERAGES DIRECTLY INSTEAD OF THE LIST VALUES
def loc2amplitude(map_array: np.ndarray, loc: list[float], circ_rad: float,
                  cutout_rad: float) -> tuple[float, float]:
    '''
    Returns annulus average and actual average of pixel values in a location on a map.
    
    Args:
        map_array: 1d numpy array of pixels in healpy Ring format.
        loc: a longitude and latitude location on the map in radians.
        circ_rad: radius of the disk around the location to be considered.
        cutout_rad: radius of the pixels within the circle to be set to 0.
                    If < 0, will only cut out a single pixel.

    Returns:
        annulus_average: average value of pixels in circle (excluding cutout)
        actual_average: average value of pixels in cutout

    '''
    nside = hp.get_nside(map_array)

    loc_deg = np.rad2deg(loc)
    loc_3d = hp.ang2vec(*loc_deg, lonlat=True)

    ipix_disc = hp.query_disc(nside=nside, vec=loc_3d, radius=circ_rad)
    if cutout_rad >= 0:
        subdisc = hp.query_disc(nside=nside, vec=loc_3d, radius=cutout_rad)
    else:
        subdisc = hp.ang2pix(nside, *loc_deg, lonlat=True)

    submap = np.zeros(len(map_array))
    submap[ipix_disc] = map_array[ipix_disc]

    submap[subdisc] = np.nan
    annulus_average: float = np.nanmean(submap[ipix_disc])
    actual_average: float = np.nanmean(map_array[subdisc])
    return (annulus_average, actual_average)
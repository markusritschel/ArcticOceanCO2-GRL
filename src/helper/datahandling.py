# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-07-30
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import logging


log = logging.getLogger(__name__)


def mask_xarray(ds, shapefile=None, **kwargs):
    """Mask an :class:`xarray.Dataset` object with the polygons found in a shapefile.
    The result is a 3-dim :class:`xarray.Dataset` with regional dimensions (lon/lat) and the different domains as a 3rd dimension. Masks can then simply be selected via `.sel(domain='Arctic')`.
    """
    import geopandas as gpd
    import regionmask
    from pathlib import Path

    if shapefile:
        shapefile = Path(shapefile)
        gdf = gpd.read_file(shapefile)
        name = shapefile.stem
    elif 'gdf' in kwargs:
        gdf = kwargs.pop('gdf')
        name = 'regions'

    regions = regionmask.from_geopandas(gdf, names="name", abbrevs="_from_name", name=name, overlap=True)

    region_masks = regions.mask_3D(ds).rename({'names':'domain', 'region':'region_idx'})

    region_masks = region_masks.swap_dims({'region_idx':'domain'})
    
    return region_masks

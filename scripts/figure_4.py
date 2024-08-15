# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-08-02
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
from calendar import calendar
import logging
import textwrap
import cmocean
import matplotlib as mpl
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cmcrameri
import matplotlib.pyplot as plt
from my_code_base.plot.maps import *

from src import *
from src.helper.graphics import MidpointNorm

xr.set_options(keep_attrs=True)
log = logging.getLogger(__name__)
plt.style.use(BASE_DIR/"assets/mpl_styles/white_paper.mplstyle")


def main():
    data = load_data()
    
    annual_min_month = get_month_of_annual_min(data)
    annual_max_month = get_month_of_annual_max(data)
    seasonal_range = get_seasonal_range(data)

    # TODO: work with subfigures.?
    # fig = plt.figure(figsize=(12,10))
    # month_fig, ampl_fig = fig.subfigures(1, 2, width_ratios=[2,1])
    # plot_min_max_month_maps(month_fig)
    # plot_amplitude_maps(ampl_fig)
    for data, name in zip([annual_min_month, annual_max_month, seasonal_range], ['annual_min_month', 'annual_max_month', 'seasonal_range']):
        # first_5, last_5, raw_delta = build_pentads_and_delta(data)
        data_dict = build_pentads_and_delta(data)

        fig, axes = plt.subplots(1,3, subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
        
        images = []
        titles = iter(['2005–2009', '2017-2021', 'Delta'])
        # for j, (da, ax) in enumerate(zip([first_5, last_5, delta], axes)):
        for data_var, ax in zip(['first', 'last', 'delta'], axes):
            da = data_dict[data_var]
            long_line = f"{da[name].attrs['long_name']}"
            if ('units' in da[name].attrs):
                # if (data_var == 'delta'):
                #     long_line += f" trend [{da[name].attrs['units']}/year]"
                # else:
                long_line += f" [{da[name].attrs['units']}]"
            kwargs = dict(robust=False,
                          cbar_kwargs={'label':textwrap.fill(long_line, 20), 
                                       'shrink':.6, 
                                       'orientation': 'horizontal'}
                )
            if name == 'seasonal_range':
                kwargs['levels'] = None
                kwargs['cmap'] = cmcrameri.cm.lipari
            else:  # months 
                cmap_timing = cmcrameri.cm.brocO
                kwargs['cmap'] = cmap_timing
                kwargs['levels'] = np.arange(1, 13)
                # norm = mpl.colors.BoundaryNorm(np.arange(0, 13) + .5, cmap_timing.N)
                # cax1 = fig.add_axes([0.92, 0.49, 0.02, 0.3])
                # cbar1 = mpl.colorbar.ColorbarBase(cax1, cmap=cmap_timing, norm=norm, orientation='vertical', ticks=np.arange(1, 13))
                # # tick_locs = (np.arange(n_clusters) + 1.5)*(n_clusters - 1)/n_clusters
                # # cbar.set_ticks(tick_locs)
                # # # set tick labels (as before)
                # cbar1.set_ticklabels([calendar.month_abbr[i][:1] for i in np.arange(1,13)])



            if data_var == 'delta':
                kwargs['robust'] = True
                kwargs['cmap'] = cmocean.cm.balance
                if name.endswith('month'):
                    # Adjust for cyclic nature (months from 1 to 12)
                    da = ((da + 6) % 12) - 6
                    kwargs['robust'] = False
                    kwargs['levels'] = None
            # else:
                # kwargs['cbar_kwargs'] = None
                # kwargs['add_colorbar'] = False

            im = plot_map(da[name], ax=ax, **kwargs)
            if data_var != 'delta':
                images.append(im)
            ax.set_title(next(titles))

        if name == 'seasonal_range':
            norm = get_common_norm(images)
            for im in images:
                im.set_norm(norm)

        save(fig, PLOT_DIR/f"figure_4_{name}.png", add_hash=True, bbox_inches='tight', transparent=True)


def load_data():
    input_file = (DATA_DIR/f"raw/merged/observations/pco2/monthly/").glob(f"pco2_MPIM-SOM-FFN_*.nc")
    da = xr.open_mfdataset(input_file)['pco2'].sel(time=slice("2005", "2022")).load()
    da = da.sel(lat=slice(50, 90))
    return da


def get_month_of_annual_max(data):
    month_of_max = data.interpolate_na('time').groupby('time.year').reduce(np.argmax, 'time') + 1
    month_of_max = month_of_max % 12
    month_of_max.attrs['long_name'] = "Month of annual maximum"
    month_of_max.attrs.pop('units')
    month_of_max = month_of_max.to_dataset(name='annual_max_month')
    return month_of_max


def get_month_of_annual_min(data):
    month_of_min = data.interpolate_na('time').groupby('time.year').reduce(np.argmin, 'time') + 1
    month_of_min = month_of_min % 12
    month_of_min.attrs['long_name'] = "Month of annual minimum"
    month_of_min.attrs.pop('units')
    month_of_min = month_of_min.to_dataset(name='annual_min_month')
    return month_of_min


def get_seasonal_range(data):
    annual_min = data.groupby('time.year').reduce(np.nanmin, 'time')
    annual_max = data.groupby('time.year').reduce(np.nanmax, 'time')
    seas_range = annual_max - annual_min
    seas_range.attrs['long_name'] = "Seasonal range of pCO2"
    seas_range.attrs['units'] = "µatm"
    seas_range = seas_range.to_dataset(name='seasonal_range')
    return seas_range


def build_pentads_and_delta(data):
    first_5 = data.isel(year=slice(0, 5)).mean('year')
    last_5 = data.isel(year=slice(-5, None)).mean('year')
    raw_delta = get_delta(first_5, last_5)
    return {'first': first_5, 
            'last': last_5, 
            'delta': raw_delta}


def get_delta(ds1, ds2):
    delta = ds2 - ds1
    var = list(ds1.data_vars)[0]
    delta[var].attrs['long_name'] = 'Δ' + delta[var].attrs['long_name']
    return delta


def plot_map(data, ax=None, **kwargs):
    ax = ax or plt.gca()
    im = data.plot(ax=ax, transform=ccrs.PlateCarree(), **kwargs)
    ax.polar.add_features(labels=False, ruler=False)
    return im


def get_common_norm(images, **kwargs):
    """
    Find the min and max of all colors for use in setting the color scale.
    """
    midpoint = kwargs.pop('midpoint', None)
    ims = np.array([image.get_array().data for image in images])
    vmin = np.nanquantile((ims), .02)
    vmax = np.nanquantile((ims), .98)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    if (midpoint is not None) or (np.sign(vmin) != np.sign(vmax)):
        norm = MidpointNorm(vmin=vmin, vmax=vmax, center=midpoint or 0)
    return norm


if __name__ == "__main__":
    main()

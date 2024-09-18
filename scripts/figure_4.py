# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  git@markusritschel.de
# Date:   2024-08-02
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import calendar
import logging
import textwrap
import matplotlib as mpl
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cmcrameri
import matplotlib.pyplot as plt
from my_code_base.plot.maps import *

from src import *
from src.helper.graphics import MidpointNorm
from src import config as cfg

xr.set_options(keep_attrs=True)
log = logging.getLogger(__name__)
plt.style.use(BASE_DIR/"assets/mpl_styles/white_paper.mplstyle")


def main():
    data = load_data()
    
    annual_min_month = get_month_of_annual_min(data)
    annual_max_month = get_month_of_annual_max(data)
    seasonal_range = get_seasonal_range(data)

    plot_maps(annual_min_month, cmap=cmcrameri.cm.brocO)
    plot_maps(annual_max_month, cmap=cmcrameri.cm.brocO)
    plot_maps(seasonal_range, cmap=cmcrameri.cm.lipari)


def load_data():
    da = xr.open_mfdataset(BASE_DIR/cfg.pco2_file)[cfg.pco2_variable].sel(time=slice(*cfg.focus_period)).load()
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
    first_5.attrs['title'] = f"{data.year.values[0]}–{data.year.values[4]}"

    last_5 = data.isel(year=slice(-5, None)).mean('year')
    last_5.attrs['title'] = f"{data.year.values[-5]}–{data.year.values[-1]}"

    raw_delta = get_delta(first_5, last_5)
    raw_delta.attrs['title'] = "Delta"

    return {'first': first_5, 
            'last': last_5, 
            'delta': raw_delta}


def get_delta(da1, da2):
    delta = da2 - da1
    if delta.attrs['long_name'].startswith('Month'):
        # Adjust for cyclic nature (months from 1 to 12)
        delta = ((delta + 6) % 12) - 6
    delta.attrs['long_name'] = 'Δ' + delta.attrs['long_name']
    return delta


def plot_maps(ds, cmap=None, **kwargs):
    var = list(ds.data_vars)[0]
    da = ds[var]
    da_dict = build_pentads_and_delta(da)
    
    fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection=ccrs.NorthPolarStereo()), 
                             layout='constrained')

    fig.get_layout_engine().set(w_pad=.05, rect=[.05, 0, .95, 1])
    fig.text(0, .4, textwrap.fill(da.attrs['long_name'], 15), 
             ha='center', va='center', rotation=90, fontsize='large')

    da = da_dict['first']
    im1 = plot_map(da, ax=axes[0], cmap=cmap)
        
    da = da_dict['last']
    im2 = plot_map(da,  ax=axes[1], cmap=cmap)
        
    da = da_dict['delta']
    _im = plot_map(da, ax=axes[2], robust=True)
    

    if 'range' in var.lower():
        norm = get_common_norm([im1, im2])
        for im in [im1, im2]:
            im.set_norm(norm)
    # TODO: transform (maybe in a class) such that the norm is determined AND set

    save(fig, PLOT_DIR/f"figure_4_{var}.png", add_hash=True, 
         bbox_inches='tight', transparent=True)


def plot_map(data, ax=None, **kwargs):
    cbar_kwargs = {
        "shrink": 0.8,
        "orientation": "horizontal",
        "extend": "both",
    }
    
    long_name = data.long_name
    if long_name.lower().startswith('month'):
        cmap = kwargs.get('cmap', cmcrameri.cm.brocO)
        norm = mpl.colors.BoundaryNorm(np.arange(0, 13) + .5, cmap.N)
        kwargs['norm'] = norm

    var = data.name
    if var.lower().endswith('month'):
        kwargs['robust'] = False
        cbar_kwargs['extend'] = 'neither'

    ax = ax or plt.gca()
    im = data.plot(
        ax=ax, transform=ccrs.PlateCarree(), cbar_kwargs=cbar_kwargs, **kwargs
    )

    if long_name.lower().startswith('month'):
        im.colorbar.set_ticks(np.arange(1,13), 
                              labels=[calendar.month_abbr[i][:1] for i in np.arange(1,13)], 
                              minor=False)
        im.colorbar.minorticks_off()

    ax.polar.add_features(labels=False, ruler=False)
    ax.set_title(data.attrs["title"])
    return im


def get_common_norm(images, **kwargs):
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

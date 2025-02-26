import numpy as np
import xarray as xr
import os

# Constants
PACIFIC_Y_INDICES = (7550, -2)
ATLANTIC_Y_INDICES = (7350, -1)
PACIFIC_X_INDICES = (1500, 5900)
TARGET_PATH = '/mnt/localssd/Data_nemo/Meshes_domains/Coordinates/Regional'
TARGET_NAME = 'arct_cutorca36_coord.nc'

# Grid type lists
T_VARS = ['nav_lon', 'nav_lat', 'glamt', 'gphit', 'e1t', 'e2t']
U_VARS = ['glamu', 'gphiu', 'e1u', 'e2u']
V_VARS = ['glamv', 'gphiv', 'e1v', 'e2v']
F_VARS = ['glamf', 'gphif', 'e1f', 'e2f']

def grid_selector(pcf, var, extent, pac_patch=False):
    '''
    Function that selects and cuts 2D arrays from parent global ORCA coordinate file.

    Parameters
    ----------
    pcf : xarray Dataset
        Parent global ORCA Coordinate File.
    var : str
        Grid variable name from pcf.
    extent : list
        List with indices to cut [y_min, y_max, x_min, x_max].
    pac_patch : bool
        Pacific (True) and Atlantic (False) switch (default is False)

    Returns
    -------
    grid_array
        Ndarray to put in patch dataset.
    '''
    if var not in pcf:
        raise ValueError(f"Variable {var} not found in the dataset.")

    if pac_patch:  # Pacific patch selection
        if var in T_VARS:
            grid_array = np.flip(pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2], extent[3])).values)
        elif var in U_VARS:
            grid_array = np.flip(pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2]-1, extent[3]-1)).values)
        elif var in V_VARS:
            grid_array = np.flip(pcf[var].sel(y=slice(extent[0]-1, extent[1]-1), x=slice(extent[2], extent[3])).values)
        elif var in F_VARS:
            grid_array = np.flip(pcf[var].sel(y=slice(extent[0]-1, extent[1]-1), x=slice(extent[2]-1, extent[3]-1)).values)
    else:  # Atlantic patch selection
        grid_array = pcf[var].sel(y=slice(extent[0], extent[1]), x=slice(extent[2], extent[3])).values

    return grid_array

def create_dataset(pcf, extent, pac_patch=False):
    '''
    Create a dataset for a given extent and patch type.
    '''
    data_vars = {}
    for var in T_VARS + U_VARS + V_VARS + F_VARS:
        data_vars[var] = (["y", "x"], grid_selector(pcf, var, extent, pac_patch))
    return xr.Dataset(data_vars)

def main():
    # Load parent coordinate file
    pcf = xr.open_dataset('/mnt/localssd/Data_nemo/Meshes_domains/Coordinates/Global/ORCA_R36_coord_new.nc').squeeze()
    x_middle = int(pcf['x'].size / 2)

    # Calculate Atlantic x-indices
    atl_first_xind = PACIFIC_X_INDICES[1] + (x_middle - PACIFIC_X_INDICES[1]) * 2 + 1
    atl_last_xind = pcf['x'].size - PACIFIC_X_INDICES[0] + 1

    # Create Atlantic and Pacific datasets
    atl_extent = [ATLANTIC_Y_INDICES[0], ATLANTIC_Y_INDICES[1], atl_first_xind, atl_last_xind]
    pac_extent = [PACIFIC_Y_INDICES[0], PACIFIC_Y_INDICES[1], PACIFIC_X_INDICES[0], PACIFIC_X_INDICES[1]]

    atl_dataset = create_dataset(pcf, atl_extent)
    pac_dataset = create_dataset(pcf, pac_extent, pac_patch=True)

    # Concatenate and save datasets
    whole_dataset = xr.concat([atl_dataset, pac_dataset], dim='y')
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
    whole_dataset.to_netcdf(f'{TARGET_PATH}/{TARGET_NAME}')

if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from scipy import interpolate as intp
from scipy import optimize
import matplotlib.pyplot as plt
import gsw
from tqdm import tqdm
import multiprocessing
from joblib import Parallel,delayed

def find_sigma0_z(salinity, temperature, pressure, latitude, longitude, sigmas):
    """
    Find the depth of the isopycnal using T/S from a cast
    Inputs:
        salinity    (array) : Salinity in PSU
        temperature (array) : In-situ temperature (C)
        latitude    (array) : Latitude of the sample
        longitude   (array) : Longitude of the sample
        sigmas      (array) : Isopycnals for which to find the depth
    Outputs:
        sigma_z     (array) : Depth of the isopycnals
    Algorithm:
        1. Convert in-situ salinity and temperature to Absolute Salinity and Conservative Temperature
        2. Calculate sigma0
        3. Check to ensure the isopycnal surface spans the density range of the water column
        4. Build interpolating functions for both T and S
        5. Sweep down and find the first place where the specified density exists in between adjacent points
        6. Use the EOS to find the where the zero crossing is
    """

    def dens_diff(pressure, SA_f, CT_f, target_sigma0):
        return np.square(gsw.sigma0(SA_f(pressure),CT_f(pressure)) - target_sigma0)

    def return_early(is_scalar):
        if is_scalar:
            return np.nan
        else:
            return np.nan*np.ones(sigmas.shape)

    # Promote to single element array if only one level requested
    valid = (~np.isnan(salinity)) & (~np.isnan(temperature)) & (~np.isnan(pressure))
    salinity = salinity[valid]
    temperature = temperature[valid]
    pressure = pressure[valid]
    longitude = longitude[valid]
    latitude = latitude[valid]


    is_scalar = type(sigmas) == type(0.)
    if is_scalar:
        sigmas = np.array([sigmas])

    if valid.sum() < 3:
        return return_early(is_scalar)

    pressure = np.array(pressure)
    SA = gsw.SA_from_SP(salinity, pressure, longitude, latitude)
    CT = gsw.CT_from_t(salinity, temperature, pressure)

    P_sort = np.unique(pressure)
    if len(P_sort) < 3:
        return return_early(is_scalar)

    SA_sort = np.zeros(P_sort.shape)
    CT_sort = np.zeros(P_sort.shape)
    # Loop through all pressures that overlap, average the temperatures and salinity accordingly
    for idx, P in enumerate(P_sort):
        presidx = (pressure == P)
        SA_sort[idx] = SA[presidx].mean()
        CT_sort[idx] = CT[presidx].mean()

    SA_intp = intp.interp1d(P_sort, SA_sort, kind='quadratic')
    CT_intp = intp.interp1d(P_sort, CT_sort, kind='quadratic')

    sigma0 = gsw.sigma0(SA_sort,CT_sort)
    sigma0_max = sigma0.max()
    sigma0_min = sigma0.min()

    sigma0_z = np.zeros(len(sigmas))

    for sigidx, siglev in enumerate(sigmas):
        if (siglev < sigma0_min) or (siglev > sigma0_max):
            sigma0_z[sigidx] = np.nan
        else:
            for botidx in range(len(sigma0)-1):
                if (siglev >= sigma0[botidx]) & (siglev <= sigma0[botidx+1]):
                    start_idx = botidx
                    break

            out = optimize.minimize_scalar(dens_diff,
                                            bounds=[P_sort[botidx],P_sort[botidx+1]],
                                            method='Bounded',
                                            args=(SA_intp, CT_intp, siglev))

            sigma0_z[sigidx] = out.x

    if is_scalar:
        sigma0_z = np.squeeze(sigma0_z)

    return sigma0_z

def process_cruise(cruise_df,sigma0_vals):
    nsigma0 = len(sigma0_vals)
    stations = cruise_df.station.unique()
    cols = ['latitude','longitude'] + [ f'{sig0}_z' for sig0 in sigma0_vals ]
    df_out = pd.DataFrame(columns = cols)
    for station in stations:
        df = cruise_df[ cruise_df.station == station ]
        sigma0_z = find_sigma0_z(df.salinity,df.temperature,df.pressure,df.latitude,df.longitude,sigma0_vals)
        if (np.isnan(sigma0_z).sum() != len(sigma0_vals)):
            out = np.nan*np.ones(nsigma0+2)
            out[0] = df.latitude.mean()
            out[1] = df.longitude.mean()
            out[2:] = sigma0_z
            df_out = df_out.append( pd.DataFrame([out], columns = cols), ignore_index = True )

    return df_out

if __name__ == '__main__':
  num_cores = multiprocessing.cpu_count()
  path = '/home/ashao/data/glodap/'
  path = ''
  glodap = pd.read_csv(path+'GLODAPv2.2019_Merged_Master_File.csv')
  cruise_ids = glodap.cruise.unique()
  glodap_by_cruise = [ glodap[(glodap.cruise == cruise_id) & (glodap.salinityqc == 1)] for cruise_id in cruise_ids ]
  inputs = tqdm( glodap_by_cruise )
  siglevels = np.linspace(26,27,11)
  out = Parallel(n_jobs=num_cores)(delayed(process_cruise)(cruise_df.replace(-9999,np.nan),siglevels) for cruise_df in inputs)
  out = pd.concat(out,ignore_index=True)
  out.to_csv('sigma_levels_z.csv')


import pandas as pd
import numpy as np
import datetime
from ocean_gases import ocean_gases as og
import multiprocessing
from joblib import Parallel,delayed
import time
from tqdm import tqdm

num_cores = multiprocessing.cpu_count()

if __name__ == "__main__":
    data = pd.read_csv('/home/ashao/data/glodap/GLODAPv2.2019_Merged_Master_File.csv')
    data['datetime'] = [ datetime.datetime(int(pack[0]), int(pack[1]), int(pack[2])) 
            for pack in zip(data['year'],data['month'],data['day']) ]
    data['yearfrac'] = [ date.year + date.dayofyear/(datetime.datetime(date.year,12,31)-
        datetime.datetime(date.year,1,1)).days for date in data['datetime'] ]
    subdata = data[data.pcfc11 != -9999].iloc[range(100,1000)]
    start_time = time.perf_counter()

    inputs = tqdm( (row.pcfc11,row.datetime,row.latitude) for idx,row in subdata.iterrows() )

    subdata['cfc11_age'] = Parallel(n_jobs=num_cores)(delayed(og.ttdmatch)(pcfc,date,'cfc11',sat=0.85,lat=lat) 
            for pcfc,date,lat in inputs)
#    subdata['cfc11_age'] = [ og.ttdmatch(row.pcfc11,row.datetime,'cfc11',sat=0.85,lat=row.latitude,verbose=True) for idx, row in subdata.iterrows() ]
    end_time = time.perf_counter()
    subdata.to_csv('glodap_with_ttd_ages.h5')
    print(f'Processed {len(subdata)} values in {end_time-start_time} seconds')

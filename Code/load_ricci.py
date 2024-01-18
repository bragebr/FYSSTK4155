import time
import numpy as np
import pandas as pd
from matminer.datasets import load_dataset

from pymatgen.core.composition import Composition


from matminer.featurizers.conversions import StrToComposition

start_time = time.perf_counter()
data = load_dataset('ricci_boltztrap_mp_tabular')
end_time   = time.perf_counter()
print(f'Loaded data set in {end_time - start_time} seconds.')

pre = data.shape[0]
data = data.dropna()
post = data.shape[0] ; print(f'Dropped {pre - post} NaN rows.')

print(data['κₑ.n [W/K/m/s]'])

data_conduc = pd.DataFrame(data[[
'pretty_formula',
'structure',
'is_metal',
'κₑ.n [W/K/m/s]',
]].copy())

data = pd.DataFrame(data[[
'pretty_formula',
'structure',
'is_metal',
'S.n [µV/K]',
]].copy())



data_conduc = data_conduc.rename(columns={'κₑ.n [W/K/m/s]' : 'conductivity'})

masks = data['is_metal'] == 'Yes'

pre = data.shape[0]
data_conduc = data_conduc[~masks]
data = data[~masks]
post = data.shape[0]
print(f'Dropped {pre - post} rows containing metallic compositions.')
data = data.drop('is_metal', axis=1)
data_conduc = data_conduc.drop('is_metal', axis=1)
print(f'Current shape of training data is {data.shape}')

data.to_pickle('Project3/Datasets/ricci_formatted.pkl')
data_conduc.to_pickle('Project3/Datasets/ricci_conductivity.pkl')

superframe_conduc = [data_conduc[i:i+6800] for i in range(0,len(data_conduc),6800)]
superframe = [data[i:i+2000] for i in range(0,len(data),2000)]
superframe_conduc[0].to_pickle('Project3/Datasets/ricci_conductivity_partitioned.pkl')

print(superframe_conduc[0])
superframe[4].to_pickle('Project3/Datasets/ricci_partition.pkl')

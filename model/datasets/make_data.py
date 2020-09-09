import h5py
import os
h5_file_name = "2020_GRF_1"
f = h5py.File(h5_file_name, 'w')

for name in os.listdir():
    if "2020" in name:
        f.create_dataset(name+'/features', data=data_of_name)
        f.create_dateset(name+'/gtscore', data=data_of_name)

f.close()

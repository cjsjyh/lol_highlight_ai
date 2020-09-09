import h5py
filename = "eccv16_dataset_tvsum_google_pool5.h5"

with h5py.File(filename,"r") as f:
    for ally in f:
        print(ally)
        for key in list(f[ally].keys()):
            x = f[ally][key]
            print(key+' '+str(x.shape))
            print(x[()])
            if key is "user_summary":
                print(f[ally][key][...])
        
        break
filename="a.h5"
with h5py.File(filename,"r") as f:
    for ally in f:
        print(ally)
        for key in list(f[ally].keys()):
            x = f[ally][key]
            print(key+' '+str(x.shape))
            print(x[()])
            if key is "user_summary":
                print(f[ally][key][...])
        
        break


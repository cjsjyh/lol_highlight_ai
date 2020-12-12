import h5py

target = "lol_dataset.h5"
target = h5py.File(target,"a")
syncer = ['newframe_dataset.h5']
feature = 'features_efficientnet'
for sep_h5 in syncer:
    f = h5py.File(sep_h5,'r')
    for tg in f.keys():
        if feature not in target[tg].keys():
            target.copy(f[tg+'/'+feature],tg+'/'+feature)
            print(f'{tg} merged!')
            print(f'{feature} is added to {target}')




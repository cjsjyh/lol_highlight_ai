import h5py

target = "test.h5"
target = h5py.File(target,"a")
syncer = ['a.h5',"b.h5",'c.h5','d.h5','e.h5','f.h5']

for sep_h5 in syncer:
    f = h5py.File(sep_h5,'r')
    for tg in f.keys():
        if tg not in target.keys():
            target.copy(f[tg],tg)
            print(f'{tg} merged!')




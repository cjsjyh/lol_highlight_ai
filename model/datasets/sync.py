import h5py

target = "m_resnet.h5"
target = h5py.File(target,"a")
syncer = "m3_audio.h5"
syncer = h5py.File(syncer,"a")

sync_feature = "audio_features_128"

for tg in target:
    if sync_feature not in target[tg].keys():
        target.copy(syncer[tg+'/'+sync_feature],tg+'/'+sync_feature)
        print(tg+'/'+sync_feature)


import numpy as np
import zarr

def save_zarr_dict(path, dict):
    store = zarr.ZipStore(path, mode='w')
    for key in dict:
        root = zarr.group(store=store)

        data = root.zeros(key, shape=dict[key].shape, dtype=dict[key].dtype)
        data[:] = dict[key][:]
    store.close()

def load_zarr_dict(path):
    store = zarr.open(path, 'r')
    data = {}
    for k in sorted(store.keys()):
        data[k] = store[k][:]
    return data

if __name__=="__main__":

    # np.save('example.npy', np.zeros((3621, 128, 128, 3)).astype(np.uint8))

    store = zarr.ZipStore('example.zip', mode='w')
    root = zarr.group(store=store)
    imgs = root.zeros('imgs', shape=(3621, 128, 128, 3), chunks=(100, 100), dtype=np.uint8)
    obs = root.zeros('obs', shape=(3621, 128, 128, 3), chunks=(100, 100), dtype=np.uint8)

    store.close()

    store = zarr.ZipStore('example.zip', mode='r')
    root = zarr.group(store=store)
    z = root['bar']
    print(z[:])
    store.close()


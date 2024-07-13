import numpy as np
import os

path = './reduced'

if __name__ == '__main__':
    if not os.path.exists(path):
        os.mkdir(path)
    
    arr = np.random.uniform(0, 1, size=(80, 80, 35, 3599))

    np.save(path+'/test.npy', arr)

    loaded = np.load(path+'/test.npy')
    print(loaded.shape)

    print(np.abs(loaded - arr))

import numpy as np
import h5py

def dataLoader(data_file):
        file=h5py.File(str(data_file),'r')        
        ref=file['ref'][:].astype(np.float32)
        label=file['label'][:].astype(np.float32)
        ref=ref.transpose(0,3,2,1)
        label=np.transpose(label,(0,3,2,1))
        return ref/255.0, label/255.0

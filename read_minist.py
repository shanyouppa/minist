import numpy as np
import struct

def read_minist_imgs(path):
    with open(path, 'rb') as f:
        magic_number, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images_data = np.frombuffer(f.read(), dtype=np.uint8)
        return images_data.reshape(num, rows, cols)

def read_minist_labels(path):
    with open(path, 'rb') as f:
        magic_number, num = struct.unpack('>II', f.read(8))
        labels_data = np.frombuffer(f.read(), dtype=np.uint8)
        return labels_data

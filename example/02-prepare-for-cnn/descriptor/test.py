import numpy as np
import ctypes

mylib = np.ctypeslib.load_library("libdescriptor_ext", ".")

coords = np.array([[1.1,1.2,1.3,1.4],[2.1,2.2,2.3,2.4],[3.1,3.2,3.3,3.4],[4.1,4.2,4.3,4.4]])
print(coords)
print(coords[2:4,::2])
mylib.test(coords[2:4,::2].copy().ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(2), ctypes.c_int(2))


# a=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# b=a.reshape((2,2,3))
# print(b[0][1][0])

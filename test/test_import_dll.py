import sys
import os
import ctypes
import platform
from ctypes import cdll, c_double, WINFUNCTYPE, POINTER, CFUNCTYPE

# os.environ['PYTHONPATH'] = 'C:\Program Files\MATLAB\MATLAB Runtime\R2024b\runtime\win64' + os.environ.get('PYTHONPATH', '')

# 获取平台位数的信息
architecture = platform.architecture()
 
# 打印结果
print(architecture)

try:
    test_sum_lib = cdll.LoadLibrary('C:/Users/pdl/Desktop/test/test_sum.dll')
    # test_sum_lib = ctypes.CDLL('C:/Users/pdl/Desktop/test/test_sum.dll',winmode=0)
except OSError as e:
    print(f"Error Loading DLL: {e}")

test_sum_lib._FuncPtr.argtypes=[ctypes.c_double,ctypes.c_double]
test_sum_lib._FuncPtr.restype=ctypes.c_double
print(f"{test_sum_lib._name}")
# print(f"{test_sum_lib['FunctionName']}")
test_sum_lib

a = 1
b = 3.14
# res = test_sum_lib.test_sum

# try:
#     test_fmincon_dll = ctypes.CDLL('./test_sum.dll',winmode=0)
# except OSError as e:
#     print(f"Error Loading DLL: {e}")
# FUNCTYPE = WINFUNCTYPE(c_double, c_double)

# print(f"{test_fmincon_dll._FuncPtr}")
# func_ptr = test_fmincon_dll._FuncPtr
# test_fmincon = FUNCTYPE(func_ptr)

# print(f"{test_fmincon}")

# fun = lambda x: 100*(x[1]-x[0]^2)^2 + (1-x[0])^2
# x0 = [-1,2]
# A = [1,2]
# b = 1.0

# x_opt_matlab = test_fmincon(b)
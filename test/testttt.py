import anony_fun_module
import numpy as np
f = anony_fun_module.Whitley
f2 = anony_fun_module.anony_fun
x = [1.0000001,0.99999963,1.00000043,0.9999992,  1.00000063, 0.99999866,
 1.00000035, 0.99999743, 0.99999842, 0.9999932,  0.99999011, 0.99997657,
 0.99995655, 0.99990966, 0.99982186, 0.99964034, 0.99928117, 0.99855767,
 0.99711134, 0.99421603]
print(f"whitley {len(x)}D has value {f(x)}")
print(f"anony_fun has value {f2(x)}")

dim=5
random_array = np.random.random(dim,)*200 + np.ones((1,dim),dtype=float)*-100
print(f"random_array is {random_array[0]}")

x = random_array[0]
A = np.ones((1,dim),dtype=float)
b = np.array([1.0],dtype=float)
print(f"A is {A[0]}, x is {x}, b is {b[0]}")
print(f"Ax is {np.dot(A[0],x)}")

A = [1,2,3,4]
x = [1,2,3,4]
print(f"A is {A}, x is {x}")
print(f"Ax is {np.dot(A,x)}")
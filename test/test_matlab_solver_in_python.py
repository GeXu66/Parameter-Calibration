import os
import time
import numpy as np
import matlab.engine
import scipy.optimize
from ctypes import cdll, c_double, WINFUNCTYPE, POINTER, CFUNCTYPE
import anony_fun_module

eng = matlab.engine.start_matlab()
os.environ['PYTHONPATH'] = '' + os.environ.get('PYTHONPATH', '')
########## Test basic summation ###########################
a = 1
b = 3.14

print(f'sum_python is {a + b}')

sum_matlab = eng.test_sum(float(a), float(b), nargout=1)
print(f'sum_matlab is {sum_matlab}')

########## Test fmincon on easy function ##################
x0 = np.array([-1.0, 2.0], dtype=float)
A = np.array([1.0, 2.0], dtype=float)
b = np.array([1.0], dtype=float)
Aeq = np.array([0.0, 2.0], dtype=float)
beq = np.array([0.0], dtype=float)

start_time_matlab = time.time()
x_opt_matlab = eng.test_fmincon(x0, A, b, Aeq, beq)  # the obj function is constructed through python module as a callback function in matlab
stop_time_matlab = time.time()
elapsed_time_matlab = stop_time_matlab - start_time_matlab
print(f'x_opt_matlab is {x_opt_matlab}')

# solve the same problem using trust-constr (interior-point) in python
f = anony_fun_module.anony_fun
C1 = lambda x: 1 - x[0] - 2 * x[1]
constraints = [{'type': 'ineq', 'fun': lambda x: C1(x)},  ##大于等于0
               ]
start_time_python = time.time()
x_opt_python = scipy.optimize.minimize(f, x0, method="trust-constr", constraints=constraints)
stop_time_python = time.time()
elapsed_time_python = stop_time_python - start_time_python

# compare the results
res_matlab = f(x_opt_matlab[0])
res_python = f(x_opt_python.get('x'))

print(
    f"result of Matlab is {res_matlab} with solution {x_opt_matlab[0]} using {elapsed_time_matlab}s\nresult of Python is {res_python} with solution {x_opt_python.get('x')} using {elapsed_time_python}s")

####################Compare Matlab and Python Performance on Hard Unconstrained function###################
dim = 20  # test on ND whitley function
trial = 10  # repeat the experiment trials
elapsed_time_matlab = []
x_opt_matlab = []
x_opt_python = []
elapsed_time_python = []
res_matlab = []
res_python = []
eq_cons_vio_matlab = []
eq_cons_vio_python = []
for i in range(trial):
    random_array = np.random.random(dim, ) * 200 + np.ones((1, dim), dtype=float) * -100
    x0 = random_array[0]
    # A = np.zeros((1,dim),dtype=float)
    # b = np.zeros((1,1),dtype=float)
    A = np.ones((1, dim), dtype=float)
    b = np.array([1.0], dtype=float)
    Aeq = np.ones((1, dim), dtype=float)
    beq = np.array([1.0], dtype=float)

    start_time_matlab = time.time()
    x_opt_matlab.append(eng.test_fmincon(x0, A, b, Aeq, beq))  # the obj function is constructed through python module as a callback function in matlab
    stop_time_matlab = time.time()
    elapsed_time_matlab.append(stop_time_matlab - start_time_matlab)
    print(f'x_opt_matlab is {x_opt_matlab[i][0]}')

    # solve the same problem using trust-constr (interior-point) in python
    f = anony_fun_module.Whitley
    C1 = lambda x: b[0] - np.dot(A[0], x)
    C2 = lambda x: beq[0] - np.dot(Aeq[0], x)
    constraints = [{'type': 'ineq', 'fun': lambda x: C1(x)},  ##大于等于0
                   {'type': 'eq', 'fun': lambda x: C2(x)},  ##大于等于0
                   ]
    start_time_python = time.time()
    x_opt_python.append(scipy.optimize.minimize(f, x0, method="trust-constr", constraints=constraints))
    stop_time_python = time.time()
    elapsed_time_python.append(stop_time_python - start_time_python)
    print(f"x_opt_python is {x_opt_python[i].get('x')}")

    # compare the results
    res_matlab.append(f(x_opt_matlab[i][0]))
    res_python.append(f(x_opt_python[i].get('x')))
    print(f"Aeq is {Aeq[0]}, x_opt_matlab is {x_opt_matlab[i][0]}, x_opt_python is {x_opt_python[i].get('x')}")
    temp_matlab = 0
    for j in range(len(Aeq[0])):
        temp_matlab = temp_matlab + Aeq[0][j] * x_opt_matlab[i][0][j]
    temp_python = 0
    for j in range(len(Aeq[0])):
        temp_python = temp_python + Aeq[0][j] * x_opt_python[i].get('x')[j]
        # print(f"temp_matlab is {temp_matlab}, temp_python is {temp_python}")
    # input()
    eq_cons_vio_matlab.append(abs(temp_matlab - 1))
    eq_cons_vio_python.append(abs(temp_python - 1))
    print(f"res_matlab is {res_matlab[i]}\nres_python is {res_python[i]}")
    # 显示消息并等待用户按回车键
    # print(f"The {i}th optimization finished, press return to continue the next optimization")
    # input()

print(f"The comparison result on Whitley {dim}D function with linear inequality constraints")
print(
    f"result of Matlab is {res_matlab} with average {sum(res_matlab) / len(res_matlab)} using {sum(elapsed_time_matlab)}s\nresult of Python is {res_python} with average {sum(res_python) / len(res_python)} using {sum(elapsed_time_python)}s")
print(
    f"absolute constraint violation of Matlab is {eq_cons_vio_matlab} with average {sum(eq_cons_vio_matlab) / len(eq_cons_vio_matlab)}\nabsolute constraint violation of Python is {eq_cons_vio_python} with average {sum(eq_cons_vio_python) / len(eq_cons_vio_python)}")

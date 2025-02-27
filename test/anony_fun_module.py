# anony_fun_module.py

def anony_fun(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def Whitley(x):
    fx = 0
    for i in range(len(x)-1):
        fx = fx + 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
    return fx  
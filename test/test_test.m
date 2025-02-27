fun = @(x)100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
dim = 10;
x0 = rand(1,dim)*200+ones(1,dim)*-100;
A = [1,2];
b = 1;
x = fmincon(fun,x0,[],[])
y = test_fmincon(x0,[],[])

py_module.Whitley(y)
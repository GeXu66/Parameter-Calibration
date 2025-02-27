function x_opt = test_fmincon(x0,A,b,Aeq,beq)
    % fun = @(x)100*(x(2)-x(1)^2)^2 + (1-x(1))^2;
    options = optimset('MaxIter',200*length(x0),'MaxFunEvals',200*length(x0),'Algorithm','interior-point');
    x_opt = fmincon(@my_callback,x0,A,b,Aeq,beq,[],[],[],options);
    % disp(["x_opt is " num2str(x_opt)])
end

% 定义matlab回调函数
function y = my_callback(x)
    % 启动MATLAB引擎,从matlab内调用python模块    
    py_module = py.importlib.import_module('anony_fun_module');
    y = py_module.Whitley(x);    
end
    
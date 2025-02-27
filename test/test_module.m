clc
clear

% 启动MATLAB引擎
eng = py.importlib.import_module('anony_fun_module');
eng

% 调用Python函数
result = eng.anony_fun([1,2]);
result

% 显示结果
disp(result);
function obj = f(x)
x1 = x(1);
x2 = x(2);
obj = 0.2 + x1^2 + x2^2 - 0.1*cos(6.0*pi*x1) - 0.1*cos(6.0*pi*x2);
end
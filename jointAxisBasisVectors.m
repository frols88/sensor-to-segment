function [x,y] = jointAxisBasisVectors(j)

x = randn(3,1);
x = x - j'*x*j;
y = cross(x,j);
x = x/norm(x);
y = y/norm(y);
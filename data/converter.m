
clear all
n1 = 10000;
n2 = 40000;
f1 = fopen('t10k-images-idx3-ubyte','r');
xTe = fread(f1,28*28*n1,'uchar');
xTe = reshape(xTe,28*28,n1);
f2 = fopen('train-images-idx3-ubyte','r');
xTr = fread(f2,28*28*n2,'uchar');
xTr = reshape(xTr,28*28,n2);

fclose(f1);
fclose(f2);

g1 = fopen('t10k-labels-idx1-ubyte','r');
g2 = fopen('train-labels-idx1-ubyte','r');
yTe = fread(g1,n1,'uchar');
yTr = fread(g2,n2,'uchar');


fclose(g1);
fclose(g2);

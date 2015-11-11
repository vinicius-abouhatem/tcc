function [x y] = quasi_newton()
    n = 2;
    x1 = rand(n,1);
    H = eye(n);
    tol = 1e-5;
    do
        x = x1;
        GRAD = grad_f(x,n);
        if norm(GRAD) < tol
            break
        endif
        p = -H * grad_f(x,n);
        t = step(x,p);
        x1 = x + t*p;
        y = grad_f(x1,n) - grad_f(x,n);
        V = eye(n) - (p'*y)\(p*y');
        H = V*H*V' + t*(p'*y)\(p*p');
    until abs(f(x1) - f(x)) < tol
    y = f(x1);
    x = x1;
endfunction


function y = f(x)
    y = norm(x)^2;
endfunction

function dy = grad_f(x,n)
    dy = zeros(n,1);
    I = eye(n);
    eps = 0.0001;
    fx = f(x);
    for i=1:n
        dy(i) = (f(x + I(:,i).*eps) - fx)/eps;
    endfor
%    dy = 2*x;
endfunction

function step_size = step(x,p)
    step_size = 100;
    alpha = 0.5;
    fx = f(x);
    while f(x + p*step_size) > fx + 0.01
        step_size = step_size * alpha;
    endwhile
endfunction

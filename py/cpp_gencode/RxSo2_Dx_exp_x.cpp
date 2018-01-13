Scalar const c0 = exp(sigma);
Scalar const c1 = c0*cos(theta);
Scalar const c2 = c0*sin(theta);
result[0] = c1;
result[1] = c2;
result[2] = -c2;
result[3] = c1;

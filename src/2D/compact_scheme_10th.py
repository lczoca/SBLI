import numpy as np

def compact_scheme_10th(h,vec):

  n = vec.shape[0] # Number of snapshots or number of points

  ## Matrix coefficients ##
  # j = 1 and n
  alpha1 = 2.0

  # j = 2 and n-1
  alpha2 = 0.25
  a2 = 3.0/4.0

  # j = 3 and n-2
  alpha3 = 4.7435/10.67175
  beta3  = 0.2964375/10.67175
  a3 = 7.905/10.67175
  b3 = 1.23515625/10.67175

  # j = 4 and n-3 #
  alpha4 = 4.63271875/9.38146875
  beta4 = 0.451390625/9.38146875
  a4 = 6.66984375/9.38146875
  b4 = 1.53/9.38146875
  c4 = 0.015/9.38146875

  # else
  alpha = 0.5
  beta = 0.05
  a = 17.0/24.0
  b = 101/600
  c = 0.01/6.0

  ### Building pentadiagonal vector (diagonal vectors) ###

  # |D(1) C(1) F(1)                                   |
  # |A(1) D(2) C(2) F(2)                              |
  # |E(1) A(2) D(3) C(3) F(3)                         |
  # |     E(2) A(3) D(4) C(4) F(4)                    |
  # |                                                 |
  # |               E(n-4) A(n-3) D(n-2) C(n-2) F(n-2)|
  # |                      E(n-3) A(n-2) D(n-1) C(n-1)|
  # |                             E(n-1) A(n-1) D(N  )|

  E1 = np.zeros(n)
  A2 = np.zeros(n)
  D3 = np.zeros(n)
  C4 = np.zeros(n)
  F5 = np.zeros(n)

  E1[:] = beta
  A2[:] = alpha
  D3[:] = 1.0
  C4[:] = alpha
  F5[:] = beta

  C4[0] = alpha1
  C4[1] = alpha2
  C4[2] = alpha3
  C4[3] = alpha4

  F5[0] = 0.0
  F5[1] = 0.0
  F5[2] = beta3
  F5[3] = beta4

  A2[0] = alpha2
  A2[1] = alpha3
  A2[2] = alpha4

  E1[0] = beta3
  E1[1] = beta4

  E1[-1] = 0.0
  E1[-2] = 0.0
  E1[-3] = 0.0
  E1[-4] = 0.0
  E1[-5] = beta3
  E1[-6] = beta4

  A2[-1] = 0.0
  A2[-2] = alpha1
  A2[-3] = alpha2
  A2[-4] = alpha3
  A2[-5] = alpha4

  C4[-1] = 0.0
  C4[-2] = alpha2
  C4[-3] = alpha3
  C4[-4] = alpha4

  F5[-1] = 0.0
  F5[-2] = 0.0
  F5[-3] = beta3
  F5[-4] = beta4

  ### Compute RHS ###
  rhs = np.zeros(n)
  # j = 1
  rhs[0] = (-2.5*vec[0] + 2.0*vec[1] + 0.5*vec[2])/h
  # j = 2
  rhs[1] = (vec[2] - vec[0])*a2/h
  # j = 3
  rhs[2] = (vec[3] - vec[1])*a3/h + (vec[4] - vec[0])*b3/h
  # j = 4
  rhs[3] = (vec[4] - vec[2])*a4/h + (vec[5] - vec[1])*b4/h + (vec[6] - vec[0])*c4/h

  # j = n
  rhs[-1] = (2.5*vec[-1] - 2.0*vec[-2] - 0.5*vec[-3])/h
  # j = n-1
  rhs[-2] = (vec[-1] - vec[-3])*a2/h
  # j = n-2
  rhs[-3] = (vec[-2] - vec[-4])*a3/h + (vec[-1] - vec[-5])*b3/h
  # j = n-3
  rhs[-4] = (vec[-3] - vec[-5])*a4/h + (vec[-2] - vec[-6])*b4/h + (vec[-1] - vec[-7])*c4/h

  # Else #
  rhs[4:-4] = (vec[5:-3] - vec[3:-5])*a/h + (vec[6:-2] - vec[2:-6])*b/h + (vec[7:-1] - vec[1:-7])*c/h

  #### Solve the linear system to find the derivatives ###
  E = np.zeros(n)
  A = np.zeros(n)
  D = np.zeros(n)
  C = np.zeros(n)
  F = np.zeros(n)
  B = np.zeros(n)
  sol = np.zeros(n)

  E[:] = E1
  A[:] = A2
  D[:] = D3
  C[:] = C4
  F[:] = F5
  B[:] = rhs

  for i in range(1,n-1):
  #for i in range(1,2):
    xmult = A[i-1]/D[i-1]
    D[i] = D[i] - xmult*C[i-1]
    C[i] = C[i] - xmult*F[i-1]
    B[i] = B[i] - xmult*B[i-1]
    xmult = E[i-1]/D[i-1]
    A[i] = A[i] - xmult*C[i-1]
    D[i+1] = D[i+1] - xmult*F[i-1]
    B[i+1] = B[i+1] - xmult*B[i-1]

  xmult = A[-2]/D[-2]
  D[-1] = D[-1] - xmult*C[-2]
  sol[-1] = (B[-1] - xmult*B[-2])/D[-1]
  sol[-2] = (B[-2] - C[-2]*sol[-1])/D[-2]

  # Back substituion #
  for i in range(n-2)[::-1]:
    sol[i] = (B[i] - F[i]*sol[i+2] - C[i]*sol[i+1])/D[i]

  return sol

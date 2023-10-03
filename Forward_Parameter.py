year2sec = 365 * 24 * 3600

#Parameters in a spring-slider
a = 1.0e-4
a_b = -1.0e-5
dc = 5.0e-3 #[m]
b = a - a_b

sigma = 5.0e6 #[Pa]

kcri = sigma * abs((b - a)) / dc
k = 0.9999 * kcri

G = 3.0e10 #[Pa]
vs = 3.0e3 #[m/s]
eta = G / (2 * vs)

vpl = 5e-2 / year2sec #[m/s]
statepl = dc / vpl

#Calculation time
time_start = 0 #[yr]
time_end = 5.516144070244743 - 3.3535635785990974 #[yr]
t1 = time_start * year2sec #[s]
t2 = time_end * year2sec #[s]

#v and theta at t = 0
v_ini = 2.869366153135153e-10 #[m/s]
state_ini = 12971491.247524181 #[s]

#Interval for equidistant collocation points
time_span = 101.4670364924749 * 3600 #[s]

#Parameter for optimization
seed = 1236
max_iteration = 20000
epsilon = 1e-12 #convergence condition

#Output
filename = "PINN_SSE_Forward"
save = False #whether save the result figure or not
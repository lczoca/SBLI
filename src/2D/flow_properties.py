# =========================================================================
#   Code with all the flow properties.
# =========================================================================

# =====================================
#   Flow properties
# =====================================
gamma = 1.31                            # Ratio of specific heats
Ma = 2.0                                # Mach number
Re = 395000                             # Reynolds number (based on the freestream velocity )
Pr = 0.07182                            # Prandtl number (Ratio of the Sutherland constant over free-stream temperature)
R = (gamma - 1.0) / gamma               # Gas constant

# =====================================
#   Farfield properties
# =====================================
rho_infty = 1
P_infty = 1 / gamma
T_infty = 1 / (gamma - 1)
u_infty = 2
c_infty = (gamma * P_infty / rho_infty)**0.5


import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.integrate

def run_model(a, b, Pem=1.0, Peh=1.0, phi1=3.5):
    # geometry/model parameters
    eps_avg = 0.50      # average porosity across membrane
    eps0    = eps_avg - a/3.0 - b/2.0   # inlet porosity
    delta   = 10        # geometric parameter = r_in/L
    phi2    = 3.5       # Thiele modulus 1
    L       = 1.0       # dimensionless length
    p1      = 0.5       # reaction 1 order
    p2      = 0.5       # reaction 2 order
    beta1   = 0.01      # energy generation function 1
    beta2   = 0.01      # energy generation function 2
    gamma1  = 10        # Arrhenius number 1
    gamma2  = 15        # Arrhenius number 2
    xi      = 1.0       # reaction rate ratio
    psi     = 1.0       # diffusivity ratio
    
    # transport/flow constants
    DA0     = 1.0       # reference effective diffusivity
    mu      = 1.0       # dynamic viscosity
    c0      = 5.0       # Kozeny constant
    sigma   = 1.0       # specific surface area (m^2/m^3)
    dPdr    = -1.0      # pressure gradient
    k_solid = 1.0       # membrane thermal conductivity
    k_fluid = 1.0       # fluid thermal conductivity
    
    # Dirichlet BC
    ca = 1
    cb = 0
    T  = 1
    
    # spatial grid
    Nz = 199                    # number of grid points
    z  = np.linspace(0, 1, Nz)  # generate spatial grid
    zz = z.reshape(-1, 1)       # convert z to column (transpose)
    hz = 1/(Nz - 1)             # spatial mesh size
        
    # temporal grid
    Nt = 1000                   # number of grid points
    t  = np.linspace(0, 1, Nt)  # generate temporal grid
    ht = 1/(Nt - 1)             # temporal mesh size
    
    # initial conditions
    ca_old = np.ones((Nz - 2, 1))
    cb_old = np.zeros((Nz - 2, 1))
    T_old  = np.ones((Nz - 2, 1))
    
    # time stepping
    ca_new = np.zeros((Nz - 2, 1))
    cb_new = np.zeros((Nz - 2, 1))
    T_new  = np.zeros((Nz - 2, 1))
    
    # fixed point iteration at each time step
    ca_new_new = np.zeros((Nz - 2, 1))
    cb_new_new = np.zeros((Nz - 2, 1))
    T_new_new  = np.zeros((Nz - 2, 1))
    
    # full solution array with boundary nodes
    ca_full = np.zeros((Nz, 1))
    cb_full = np.zeros((Nz, 1))
    T_full  = np.zeros((Nz, 1))
    
    # transient solution array
    ca_time = np.zeros((Nt, Nz))
    cb_time = np.zeros((Nt, Nz))
    T_time  = np.zeros((Nt, Nz))
    
    # identity matrix
    I = np.eye(Nz - 2)
    
    # porosity & derivative
    def eps_fun(z):
        return a*(z**2) + b*z + eps0
    
    def deps_fun(z):
        return 2*a*z + b
    
    # effective diffusivity & derivative
    def De_fun(z):
        return DA0*(a*(z**2) + b*z + eps0)**(5.0/4.0)
    
    def dDe_fun(z):
        return DA0*(5.0/4.0)*(2*a*z + b)*(a*(z**2) + b*z + eps0)**(1.0/4.0)
    
    # Darcy velocity & derivative
    def uD_fun(z):
        eps = eps_fun(z)
        return -(eps**3)/(mu*c0*sigma**2*(1.0 - eps)**2) * dPdr
    
    def duD_fun(z):
        eps  = eps_fun(z)
        deps = deps_fun(z)
        
        K = dPdr/(mu*c0*sigma**2)
        dgdz = deps * (3.0*eps**2/(1.0 - eps)**2 + 2.0*eps**3/(1.0 - eps)**3)
    
        return -K*dgdz
    
    def Ke_fun(z):
        return \
        (1.0 - a*(z**2) - b*z - eps0)*k_solid + (a*(z**2) + b*z + eps0)*k_fluid
    
    def dKe_fun(z):
        return \
        (-2*a*z - b)*k_solid + (2*a*z + b)*k_fluid
    
    # evaluate properties on the interior nodes
    eps_i  = eps_fun(z[1:Nz-1])
    De_i   = De_fun(z[1:Nz-1])
    uD_i   = uD_fun(z[1:Nz-1])
    Ke_i   = Ke_fun(z[1:Nz-1])
    
    deps_i = deps_fun(z[1:Nz-1])
    dDe_i  = dDe_fun(z[1:Nz-1])
    duD_i  = duD_fun(z[1:Nz-1])
    dKe_i  = dKe_fun(z[1:Nz-1])
    
    ##########################################################################
    # mass balance on species A
    
    # coefficient of dcA/dz term
    A1coef = (dDe_i / De_i) + (deps_i / eps_i) + 1.0/(delta + z[1:Nz-1]) - Pem
    
    # coefficient of cA term
    B1coef = -(L / De_i)*duD_i \
            - Pem*((deps_i / eps_i) + 1.0/(delta + z[1:Nz-1]))
            
    # coefficient of reaction 1 term
    C1coef = ((1 - eps_i) / (eps_i)) * (phi1**2)
    
    # diagonals for matrix A
    main1  = np.zeros(Nz - 2)
    lower1 = np.zeros(Nz - 3)
    upper1 = np.zeros(Nz - 3)
    
    main1[:]  = -2.0/(hz**2) + A1coef/hz + B1coef
    lower1[:] = 1.0/(hz**2) - A1coef[1:]/hz
    upper1[:] = 1.0/(hz**2)
    
    # define the sparse matrix A
    A = scipy.sparse.diags(
        diagonals=[main1, lower1, upper1],
        offsets=[0, -1, 1],
        shape=(Nz - 2, Nz - 2),
        format='csr')
    
    # modify zero RHS because of Dirichlet BC
    fa     = np.zeros((Nz - 2, 1))
    fa[0]  = 1.0/(hz**2) - ((dDe_i[0] / De_i[0]) + (deps_i[0] / eps_i[0]) 
                            + 1.0/(delta + zz[1]) - Pem) / hz
    
    # apply Neumann BC at outlet
    A[Nz - 3, Nz - 3] = A[Nz - 3, Nz - 3] + 1.0/(hz**2)
    
    ##########################################################################
    # mass balance on species B
    
    # coefficient of dcB/dz term
    A2coef = (dDe_i / De_i) + (deps_i / eps_i) + 1.0/(delta + z[1:Nz-1]) - Pem*psi
    
    # coefficient of cB term
    B2coef = -(L / De_i)*duD_i \
            - Pem*psi*((deps_i / eps_i) + 1.0/(delta + z[1:Nz-1]))
    
    # coefficient of reaction 1 term
    C2coef = ((1 - eps_i) / (eps_i)) * (phi1**2) * psi
    
    # coefficient of reaction 2 term
    D2coef = ((1 - eps_i) / (eps_i)) * (phi2**2) * xi
    
    # diagonals for matrix B
    main2  = np.zeros(Nz - 2)
    lower2 = np.zeros(Nz - 3)
    upper2 = np.zeros(Nz - 3)
    
    main2[:]  = -2.0/(hz**2) + A2coef/hz + B2coef
    lower2[:] = 1.0/(hz**2) - A2coef[1:]/hz
    upper2[:] = 1.0/(hz**2)
    
    # define the sparse matrix B
    B = scipy.sparse.diags(
        diagonals=[main2, lower2, upper2],
        offsets=[0, -1, 1],
        shape=(Nz - 2, Nz - 2),
        format='csr')
    
    # modify zero RHS because of Dirichlet BC
    fb = np.zeros((Nz - 2, 1))
    
    # apply Neumann BC at outlet
    B[Nz - 3, Nz - 3] = B[Nz - 3, Nz - 3] + 1.0/(hz**2)
    
    ##########################################################################
    # energy balance
    
    # coefficient of dT/dz term
    A3coef = (dKe_i / Ke_i) + (deps_i / eps_i) + 1.0/(delta + z[1:Nz-1]) - Peh
    
    # coefficient of T term
    B3coef = Peh*((duD_i / uD_i) + 1.0/(delta + z[1:Nz-1]))
    
    # diagonals for matrix E
    main3  = np.zeros(Nz - 2)
    lower3 = np.zeros(Nz - 3)
    upper3 = np.zeros(Nz - 3)
    
    main3[:]  = -2.0/(hz**2) + A3coef/hz + B3coef
    lower3[:] = 1.0/(hz**2) - A3coef[1:]/hz
    upper3[:] = 1.0/(hz**2)
    
    # define the sparse matrix E
    E = scipy.sparse.diags(
        diagonals=[main3, lower3, upper3],
        offsets=[0, -1, 1],
        shape=(Nz - 2, Nz - 2),
        format='csr')
    
    # modify zero RHS because of Dirichlet BC
    fT     = np.zeros((Nz - 2, 1))
    fT[0]  = 1.0/(hz**2) - ((dKe_i[0] / Ke_i[0]) + (deps_i[0] / eps_i[0])
                            + 1.0/(delta + zz[1]) - Peh) / hz
    
    # apply Neumann BC at outlet
    E[Nz - 3, Nz - 3] = E[Nz - 3, Nz - 3] + 1.0/(hz**2)
    
    ##########################################################################
    # solve systems of equations
    
    # time-marching scheme
    Nt_iter = Nt
    for i in range(Nt_iter):
        ca_new_old = ca_old
        cb_new_old = cb_old
        T_new_old  = T_old
        for j in range(2):
            # define nonlinear reaction vectors
            R1 = -C1coef.reshape(-1, 1) * np.exp(gamma1*(1 - 1 / T_old)) \
                * (np.maximum(ca_old, 0)**p1 + np.maximum(ca_new_old, 0)**p1)
            R2 = 2*C2coef.reshape(-1, 1) * np.exp(gamma1*(1 - 1 / T_old)) \
                * np.maximum(ca_old, 0)**p1 \
                - D2coef.reshape(-1, 1) * np.exp(gamma2*(1 - 1 / T_old)) \
                * (np.maximum(cb_old, 0)**p2 + np.maximum(cb_new_old, 0)**p2) 
            # solve matrix
            ca_new_new = np.linalg.solve(I - 1/2*ht * A,
                         ca_old + 1/2*ht * (A @ ca_old) + 1/2*ht * R1 + ht*fa)
            cb_new_new = np.linalg.solve(I - 1/2*ht * B,
                         cb_old + 1/2*ht * (B @ cb_old) + 1/2*ht * R2 + ht*fb)      
            T_new_new  = np.linalg.solve(I - 1/2*ht * E,
                         T_old + 1/2*ht * (E @ T_old) + 1/2*ht 
                         * ((phi1**2 * beta1
                         * (np.exp(gamma1 * (1 - 1 / T_old)) 
                         + np.exp(gamma1 * (1 - 1 / T_new_old)))
                         * np.maximum(ca_old, 0)**p1
                         + (phi2**2 * beta2 * xi 
                         * (np.exp(gamma2 * (1 - 1 / T_old))
                         + np.exp(gamma2 * (1 - 1 / T_new_old)))
                         * np.maximum(cb_old, 0)**p2))) + ht*fT)                    
            ca_old = 0.5*np.maximum(ca_new_new, 0) + 0.5*np.maximum(ca_old, 0)
            cb_old = 0.5*np.maximum(cb_new_new, 0) + 0.5*np.maximum(cb_old, 0)
            T_old  = 0.5*np.maximum(T_new_new, 0) + 0.5*np.maximum(T_old, 0)
        ca_new          = np.maximum(ca_new_new, 0)
        ca_full[0]      = ca
        ca_full[1:Nz-1] = ca_new
        ca_full[Nz-1]   = ca_new[Nz-3]
        ca_time[i, :]   = ca_full[:, 0]
        cb_new          = np.maximum(cb_new_new, 0)
        cb_full[0]      = cb
        cb_full[1:Nz-1] = cb_new
        cb_full[Nz-1]   = cb_new[Nz-3]
        cb_time[i, :]   = cb_full[:, 0]
        T_new           = np.maximum(T_new_new, 0)
        T_full[0]       = T
        T_full[1:Nz-1]  = T_new
        T_full[Nz-1]    = T_new[Nz-3]
        T_time[i, :]    = T_full[:, 0]
        
    return z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time

# compute dead zone length & yield of B
def compute_metrics(z, ca_full, cb_full, ca_tol=1e-3):
    # convert from (Nz,1) arrays to 1D arrays
    ca_profile = ca_full[:, 0]
    cb_profile = cb_full[:, 0]

    # average B produced across membrane
    B_avg = np.trapezoid(cb_profile, z)
    
    # dead-zone length based on threshold concentration
    dead_nodes = np.where(ca_profile <= ca_tol)[0]

    if len(dead_nodes) == 0:
        z_dead_start = None
        dead_zone_length = 0.0
    else:
        first_dead_idx = dead_nodes[0]
        z_dead_start = z[first_dead_idx]
        dead_zone_length = 1.0 - z_dead_start

    return {"dead_zone_length": dead_zone_length,
            "z_dead_start": z_dead_start, 
            "B_avg": B_avg}

# check if porosity is within bounds
def porosity_check(a, b, eps_avg=0.50, eps_min=0.1, eps_max=0.9, Nz=199):
    eps0 = eps_avg - a/3.0 - b/2.0
    z = np.linspace(0, 1, Nz)
    eps_profile = a*(z**2) + b*z + eps0

    return np.all(eps_profile >= eps_min) and np.all(eps_profile <= eps_max)

# coarse parameter sweep
a_vals = np.linspace(-1.0, 1.0, 21)
b_vals = np.linspace(-1.0, 1.0, 21)

# baseline operating conditions
Pem_base  = 1.0
Peh_base  = 1.0
phi1_base = 3.5

# operating-condition values tested
Pem_vals  = [0.5, 1.0, 4.0, 10.0]
Peh_vals  = [0.5, 1.0, 4.0, 10.0]
phi1_vals = [1.0, 2.0, 3.5, 5.0]

# scenario list: vary one parameter at a time, keep others at baseline
scenarios = []

for Pem in Pem_vals:
    scenarios.append({
        "type": "Pem",
        "label": f"Pem = {Pem}",
        "Pem": Pem,
        "Peh": Peh_base,
        "phi1": phi1_base})

for Peh in Peh_vals:
    scenarios.append({
        "type": "Peh",
        "label": f"Peh = {Peh}",
        "Pem": Pem_base,
        "Peh": Peh,
        "phi1": phi1_base})

for phi1 in phi1_vals:
    scenarios.append({
        "type": "phi1",
        "label": f"phi1 = {phi1}",
        "Pem": Pem_base,
        "Peh": Peh_base,
        "phi1": phi1})

# plot effects of Pem on cA, cB, T with baseline parameters
ca_profiles_Pem = []
cb_profiles_Pem = []
T_profiles_Pem  = []

for Pem in Pem_vals:
    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = \
    run_model(a=0.0, b=0.0, Pem=Pem, Peh=Peh_base, phi1=phi1_base)

    ca_profiles_Pem.append((Pem, ca_full[:, 0]))
    cb_profiles_Pem.append((Pem, cb_full[:, 0]))
    T_profiles_Pem.append((Pem, T_full[:, 0]))

plt.figure()
for Pem, profile in ca_profiles_Pem:
    plt.plot(z, profile, label=f'Pem = {Pem}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless concentration, $c_A$')
plt.title('Effect of mass Peclet number on $c_A$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for Pem, profile in cb_profiles_Pem:
    plt.plot(z, profile, label=f'Pem = {Pem}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless concentration, $c_B$')
plt.title('Effect of mass Peclet number on $c_B$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for Pem, profile in T_profiles_Pem:
    plt.plot(z, profile, label=f'Pem = {Pem}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless temperature, $\\theta$')
plt.title('Effect of mass Peclet number on $\\theta$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()


# plot effects of Peh on cA, cB, T with baseline parameters
ca_profiles_Peh = []
cb_profiles_Peh = []
T_profiles_Peh  = []

for Peh in Peh_vals:
    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = \
    run_model(a=0.0, b=0.0, Pem=Pem_base, Peh=Peh, phi1=phi1_base)

    ca_profiles_Peh.append((Peh, ca_full[:, 0]))
    cb_profiles_Peh.append((Peh, cb_full[:, 0]))
    T_profiles_Peh.append((Peh, T_full[:, 0]))

plt.figure()
for Peh, profile in ca_profiles_Peh:
    plt.plot(z, profile, label=f'Peh = {Peh}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless concentration, $c_A$')
plt.title('Effect of heat Peclet number on $c_A$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for Peh, profile in cb_profiles_Peh:
    plt.plot(z, profile, label=f'Peh = {Peh}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless concentration, $c_B$')
plt.title('Effect of heat Peclet number on $c_B$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for Peh, profile in T_profiles_Peh:
    plt.plot(z, profile, label=f'Peh = {Peh}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless temperature, $\\theta$')
plt.title('Effect of heat Peclet number on $\\theta$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()


# plot effects of phi1 on cA, cB, T with baseline parameters
ca_profiles_phi1 = []
cb_profiles_phi1 = []
T_profiles_phi1  = []

for phi1 in phi1_vals:
    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = \
    run_model(a=0.0, b=0.0, Pem=Pem_base, Peh=Peh_base, phi1=phi1)

    ca_profiles_phi1.append((phi1, ca_full[:, 0]))
    cb_profiles_phi1.append((phi1, cb_full[:, 0]))
    T_profiles_phi1.append((phi1, T_full[:, 0]))

plt.figure()
for phi1, profile in ca_profiles_phi1:
    plt.plot(z, profile, label=f'$\\varphi_1$ = {phi1}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless concentration, $c_A$')
plt.title('Effect of $\\varphi_1$ on $c_A$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for phi1, profile in cb_profiles_phi1:
    plt.plot(z, profile, label=f'$\\varphi_1$ = {phi1}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless concentration, $c_B$')
plt.title('Effect of $\\varphi_1$ on $c_B$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
for phi1, profile in T_profiles_phi1:
    plt.plot(z, profile, label=f'$\\varphi_1$ = {phi1}')
plt.xlabel('Dimensionless distance, z')
plt.ylabel('Dimensionless temperature, $\\theta$')
plt.title('Effect of $\\varphi_1$ on $\\theta$ (no porosity gradient)')
plt.legend()
plt.grid(True)
plt.show()

# store scenario results
scenario_results = []
n_scenarios = len(scenarios)

# calculate results for all (a,b) pairs, update progress bar
for s_idx, scenario in enumerate(scenarios, start=1):
    print(f"Running scenario {s_idx}/{n_scenarios}: {scenario['label']}")

    a_store = []
    b_store = []
    dead_zone_store = []
    B_avg_store = []

    eval_count = 0

    for a in a_vals:
        for b in b_vals:
            # skip porosity profiles that are out of bounds
            if not porosity_check(a, b, eps_avg=0.50, eps_min=0.1,
                                  eps_max=0.9, Nz=199):
                continue

            z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = \
            run_model(a, b, Pem=scenario["Pem"], Peh=scenario["Peh"],
                      phi1=scenario["phi1"])

            metrics = compute_metrics(z, ca_full, cb_full, ca_tol=1e-3)

            B_val = metrics["B_avg"]

            # remove nonphysical / unstable solutions
            if np.isnan(B_val) or B_val > 1.0:
                continue

            a_store.append(a)
            b_store.append(b)
            dead_zone_store.append(metrics["dead_zone_length"])
            B_avg_store.append(metrics["B_avg"])

            eval_count += 1
            if eval_count % 25 == 0:
                print(f"  {scenario['label']}: \
                      {eval_count} feasible cases completed")

    # convert stored results to arrays
    a_store = np.array(a_store)
    b_store = np.array(b_store)
    dead_zone_store = np.array(dead_zone_store)
    B_avg_store = np.array(B_avg_store)

    # store full results for later optimum-point extraction
    scenario_results.append({
        "type": scenario["type"],
        "label": scenario["label"],
        "Pem": scenario["Pem"],
        "Peh": scenario["Peh"],
        "phi1": scenario["phi1"],
        "a_store": a_store,
        "b_store": b_store,
        "dead_zone_store": dead_zone_store,
        "B_avg_store": B_avg_store})

    # raw tradeoff plot for this scenario
    plt.figure()
    plt.scatter(dead_zone_store, B_avg_store)
    plt.xlabel('Dead-zone length, $L_{dz}$')
    plt.ylabel('B produced across membrane, $B_{avg}$')
    plt.title(f"Raw tradeoff: {scenario['label']}")
    plt.grid(True)
    plt.show()

##############################################################################
# plot cA, cB, T vs. z for each scenario 
# show solutions for no porosity gradient and optimal gradient
# CAN RUN SEPARATE

from matplotlib.lines import Line2D

a_base = 0.0
b_base = 0.0

def get_best_ab(result):
    a_vals  = np.array(result["a_store"])
    b_vals  = np.array(result["b_store"])
    dz_vals = np.array(result["dead_zone_store"])
    B_vals  = np.array(result["B_avg_store"])

    sort_idx = np.lexsort((-B_vals, dz_vals))
    best_idx = sort_idx[0]

    return a_vals[best_idx], b_vals[best_idx]

def find_result(varying_name, val):
    for result in scenario_results:
        label = result["label"]
        if varying_name == "Pem" and label == f"Pem = {val}":
            return result
        elif varying_name == "Peh" and label == f"Peh = {val}":
            return result
        elif varying_name == "phi1" and label == f"phi1 = {val}":
            return result
        
# store solutions
plot_results = {"Pem":  {"vals": Pem_vals,  "base_ca": [],"base_cb": [],
                "base_T": [], "opt_ca":  [], "opt_cb":  [], "opt_T":  []},
                "Peh":  {"vals": Peh_vals,  "base_ca": [], "base_cb": [],
                "base_T": [], "opt_ca":  [], "opt_cb":  [], "opt_T":  []},
                "phi1": {"vals": phi1_vals, "base_ca": [], "base_cb": [],
                "base_T": [], "opt_ca":  [], "opt_cb":  [], "opt_T":  []},}

for Pem in Pem_vals:
    # no porosity gradient
    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = run_model(
        a=a_base, b=b_base, Pem=Pem, Peh=Peh_base, phi1=phi1_base)
    z_plot = z
    plot_results["Pem"]["base_ca"].append(ca_full[:, 0])
    plot_results["Pem"]["base_cb"].append(cb_full[:, 0])
    plot_results["Pem"]["base_T"].append(T_full[:, 0])

    # optimal porosity gradient
    result = find_result("Pem", Pem)
    a_opt, b_opt = get_best_ab(result)

    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = run_model(
        a=a_opt, b=b_opt, Pem=Pem, Peh=Peh_base, phi1=phi1_base)
    plot_results["Pem"]["opt_ca"].append(ca_full[:, 0])
    plot_results["Pem"]["opt_cb"].append(cb_full[:, 0])
    plot_results["Pem"]["opt_T"].append(T_full[:, 0])

for Peh in Peh_vals:
    # no porosity gradient
    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = run_model(
        a=a_base, b=b_base, Pem=Pem_base, Peh=Peh, phi1=phi1_base)
    plot_results["Peh"]["base_ca"].append(ca_full[:, 0])
    plot_results["Peh"]["base_cb"].append(cb_full[:, 0])
    plot_results["Peh"]["base_T"].append(T_full[:, 0])

    # optimal porosity gradient
    result = find_result("Peh", Peh)
    a_opt, b_opt = get_best_ab(result)

    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = run_model(
        a=a_opt, b=b_opt, Pem=Pem_base, Peh=Peh, phi1=phi1_base)
    plot_results["Peh"]["opt_ca"].append(ca_full[:, 0])
    plot_results["Peh"]["opt_cb"].append(cb_full[:, 0])
    plot_results["Peh"]["opt_T"].append(T_full[:, 0])

for phi1 in phi1_vals:
    # no porosity gradient
    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = run_model(
        a=a_base, b=b_base, Pem=Pem_base, Peh=Peh_base, phi1=phi1)
    plot_results["phi1"]["base_ca"].append(ca_full[:, 0])
    plot_results["phi1"]["base_cb"].append(cb_full[:, 0])
    plot_results["phi1"]["base_T"].append(T_full[:, 0])

    # optimal porosity gradient
    result = find_result("phi1", phi1)
    a_opt, b_opt = get_best_ab(result)

    z, t, ca_full, cb_full, T_full, ca_time, cb_time, T_time = run_model(
        a=a_opt, b=b_opt, Pem=Pem_base, Peh=Peh_base, phi1=phi1)
    plot_results["phi1"]["opt_ca"].append(ca_full[:, 0])
    plot_results["phi1"]["opt_cb"].append(cb_full[:, 0])
    plot_results["phi1"]["opt_T"].append(T_full[:, 0])

def make_plot(z, xvals, ysets_base, ysets_opt,
              varying_name, ylabel, title_text):
    colors = plt.cm.viridis(np.linspace(0, 1, len(xvals)))

    plt.figure(figsize=(6, 4.5))
    for val, y_base, y_opt, color in zip(xvals, ysets_base, ysets_opt, colors):
        plt.plot(z, y_base, lw=2, ls='-', color=color)
        plt.plot(z, y_opt,  lw=2, ls='--', color=color)

    color_handles = []
    for val, color in zip(xvals, colors):
        if varying_name == "phi1":
            label = rf"$\varphi_1$ = {val:g}"
        else:
            label = f"{varying_name} = {val:g}"
        color_handles.append(Line2D([0], [0], color=color, lw=2, label=label))

    style_handles = [
        Line2D([0], [0], color='black', lw=2, ls='-',
               label='No gradient'),
        Line2D([0], [0], color='black', lw=2, ls='--',
               label='Optimal gradient')]

    plt.xlim(0, 1)
    plt.xlabel(r"Dimensionless distance, $z$")
    plt.ylabel(ylabel)
    plt.title(title_text)
    plt.legend(handles=color_handles + style_handles)
    plt.tight_layout()
    plt.show()

# effect of Pem
make_plot(
    z_plot,
    plot_results["Pem"]["vals"],
    plot_results["Pem"]["base_ca"],
    plot_results["Pem"]["opt_ca"],
    "Pem",
    r"Dimensionless concentration, $c_A$",
    r"Effect of $Pe_m$ on $c_A$")

make_plot(
    z_plot,
    plot_results["Pem"]["vals"],
    plot_results["Pem"]["base_cb"],
    plot_results["Pem"]["opt_cb"],
    "Pem",
    r"Dimensionless concentration, $c_B$",
    r"Effect of $Pe_m$ on $c_B$")

make_plot(
    z_plot,
    plot_results["Pem"]["vals"],
    plot_results["Pem"]["base_T"],
    plot_results["Pem"]["opt_T"],
    "Pem",
    r"Dimensionless temperature, $\theta$",
    r"Effect of $Pe_m$ on $\theta$")

# effect of Peh
make_plot(
    z_plot,
    plot_results["Peh"]["vals"],
    plot_results["Peh"]["base_ca"],
    plot_results["Peh"]["opt_ca"],
    "Peh",
    r"Dimensionless concentration, $c_A$",
    r"Effect of $Pe_h$ on $c_A$")

make_plot(
    z_plot,
    plot_results["Peh"]["vals"],
    plot_results["Peh"]["base_cb"],
    plot_results["Peh"]["opt_cb"],
    "Peh",
    r"Dimensionless concentration, $c_B$",
    r"Effect of $Pe_h$ on $c_B$")

make_plot(
    z_plot,
    plot_results["Peh"]["vals"],
    plot_results["Peh"]["base_T"],
    plot_results["Peh"]["opt_T"],
    "Peh",
    r"Dimensionless temperature, $\theta$",
    r"Effect of $Pe_h$ on $\theta$")

# effect of phi1
make_plot(
    z_plot,
    plot_results["phi1"]["vals"],
    plot_results["phi1"]["base_ca"],
    plot_results["phi1"]["opt_ca"],
    "phi1",
    r"Dimensionless concentration, $c_A$",
    r"Effect of $\varphi_1$ on $c_A$")

make_plot(
    z_plot,
    plot_results["phi1"]["vals"],
    plot_results["phi1"]["base_cb"],
    plot_results["phi1"]["opt_cb"],
    "phi1",
    r"Dimensionless concentration, $c_B$",
    r"Effect of $\varphi_1$ on $c_B$")

make_plot(
    z_plot,
    plot_results["phi1"]["vals"],
    plot_results["phi1"]["base_T"],
    plot_results["phi1"]["opt_T"],
    "phi1",
    r"Dimensionless temperature, $\theta$",
    r"Effect of $\varphi_1$ on $\theta$")

##############################################################################
# plot of top 4 optimal porosity profiles for each scenario
# CAN RUN SEPARATE

plt.figure(figsize=(6, 4.5))
colors = plt.cm.viridis(np.linspace(0, 1, len(scenario_results)))

for result, color in zip(scenario_results, colors):

    a_vals  = result["a_store"]
    b_vals  = result["b_store"]
    dz_vals = result["dead_zone_store"]
    B_vals  = result["B_avg_store"]

    # get top 4 indices (same ranking as before)
    sort_idx = np.lexsort((-B_vals, dz_vals))
    top_idx  = sort_idx[:4]

    for idx in top_idx:
        a = a_vals[idx]
        b = b_vals[idx]

        # reconstruct eps(z)
        eps0 = 0.50 - a/3.0 - b/2.0
        eps_z = a*(z_plot**2) + b*z_plot + eps0

        plt.plot(z_plot, eps_z, lw=1.5, color=color)

plt.xlim(0, 1)
plt.xlabel(r"Dimensionless distance, $z$")
plt.ylabel(r"Porosity, $\varepsilon(z)$")
plt.title(r"Top 4 optimal porosity profiles for each scenario")
plt.tight_layout()
plt.show()

##############################################################################
# # print and plot optimal (a,b) for each scenario
# # CAN RUN SEPARATE
# from collections import defaultdict
# from matplotlib.ticker import MultipleLocator
# top_points_all = []

# print("\nTop 10 points for every scenario")

# for result in scenario_results:
#     label = result["label"]
#     a_vals = np.array(result["a_store"])
#     b_vals = np.array(result["b_store"])
#     Ldz_vals = np.array(result["dead_zone_store"])
#     Bavg_vals = np.array(result["B_avg_store"])

#     # sort by L_dz ascending, then B_avg descending
#     sort_idx = np.lexsort((-Bavg_vals, Ldz_vals))
#     top_idx = sort_idx[:10]

#     print(f"\nScenario: {label}")
#     print("rank   a        b        L_dz      B_avg      diff_from(-a-0.1)")

#     for rank, idx in enumerate(top_idx, start=1):
#         a_i = a_vals[idx]
#         b_i = b_vals[idx]
#         L_i = Ldz_vals[idx]
#         B_i = Bavg_vals[idx]
#         diff_i = b_i - (-a_i - 0.1)

#         print(f"{rank:<6d}"
#               f"{a_i:<9.3f}"
#               f"{b_i:<9.3f}"
#               f"{L_i:<10.4f}"
#               f"{B_i:<11.4f}"
#               f"{diff_i:<10.4f}")

#         top_points_all.append({
#             "label": label,
#             "a": a_i,
#             "b": b_i,
#             "L_dz": L_i,
#             "B_avg": B_i,
#             "diff_line": diff_i})

# # count duplicates of optimal (a,b) pairs
# count_map = defaultdict(int)
# for pt in top_points_all:
#     key = (pt["a"], pt["b"])
#     count_map[key] += 1

# # plot optimal (a,b) pairs for all scenarios
# plt.figure(figsize=(8, 6))

# scenario_labels = [r["label"] for r in scenario_results]
# cmap = plt.get_cmap('viridis')   # fixes deprecation warning

# for i, label in enumerate(scenario_labels):
#     pts = [pt for pt in top_points_all if pt["label"] == label]

#     a_plot = [pt["a"] for pt in pts]
#     b_plot = [pt["b"] for pt in pts]

#     plt.scatter(
#         a_plot,
#         b_plot,
#         color = cmap(i / (len(scenario_labels) - 1)),
#         alpha=0.75,
#         linewidths=0.8,
#         label=label)

# # annotate duplicate counts only where count > 1
# for (a_pt, b_pt), n in count_map.items():
#     if n > 1:
#         plt.annotate(
#             f"{n}",
#             (a_pt, b_pt),
#             textcoords="offset points",
#             xytext=(5, 5),
#             fontsize=9)

# # linear trendline
# all_a = np.array([pt["a"] for pt in top_points_all])
# all_b = np.array([pt["b"] for pt in top_points_all])

# m, c = np.polyfit(all_a, all_b, 1)
# b_fit = m * all_a + c

# # R^2
# ss_res = np.sum((all_b - b_fit) ** 2)
# ss_tot = np.sum((all_b - np.mean(all_b)) ** 2)
# r2 = 1 - ss_res / ss_tot

# # plot trendline, floating label
# a_line = np.linspace(all_a.min() - 0.1, all_a.max() + 0.1, 200)
# b_line = m * a_line + c
# plt.plot(a_line, b_line, linestyle="--", linewidth=2, color="black")
# eq_text = f"b = {m:.3f}a + {c:.3f}\n$R^2$ = {r2:.4f}"
# ax = plt.gca()
# ax.text(0.25, 0.95, eq_text, transform=ax.transAxes, fontsize=10,
#         verticalalignment='top', bbox=dict(boxstyle="round", 
#         facecolor="white", alpha=0.8))

# plt.xlabel("a")
# plt.ylabel("b")
# ax = plt.gca()
# ax.xaxis.set_major_locator(MultipleLocator(0.2))
# ax.yaxis.set_major_locator(MultipleLocator(0.2))
# plt.title("Top 10 Optimal (a,b) Pairs for All Scenarios")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

##############################################################################
# solution plots

# # location in membrane
# j_in  = 0
# j_mid = Nz // 2
# j_out = Nz - 1

# plt.plot(t, ca_time[:, j_mid], label=f'ε₀ = {eps0:.1f}')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('Dimensionless Time, t(-)')
# plt.ylabel('Dimensionless Concentration, $c_A$(-)')
# plt.title('Effect of Inlet Porosity on $c_A$ at z = 0.5')
# plt.legend()
# plt.show()

# # cA 3D plot
# pos, time = np.meshgrid(z, t)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(pos, time, ca_time, cmap='viridis')
# ax.set_xlabel('Dimensionless Position, z(-)')
# ax.set_ylabel('Dimensionless Time, t(-)')
# ax.set_zlabel('Dimensionless Concentration, $c_A$(-)')
# plt.show()

# cA contour plot
# plt.figure()
# contour = plt.contourf(z, t, ca_time, levels=50, cmap='viridis')
# plt.xlabel('Dimensionless Position, z(-)')
# plt.ylabel('Dimensionless Time, t(-)')
# cbar = plt.colorbar(contour)
# cbar.set_label('Dimensionless Concentration, $c_A$(-)')
# plt.show()

# # cB contour plot
# plt.figure()
# contour = plt.contourf(z, t, cb_time, levels=50, cmap='viridis')
# plt.xlabel('Dimensionless Position, z(-)')
# plt.ylabel('Dimensionless Time, t(-)')
# cbar = plt.colorbar(contour)
# cbar.set_label('Dimensionless Concentration, $c_B$(-)')
# plt.show()

# # T contour plot
# plt.figure()
# contour = plt.contourf(z, t, T_time, levels=50, cmap='viridis')
# plt.xlabel('Dimensionless Position, z(-)')
# plt.ylabel('Dimensionless Time, t(-)')
# cbar = plt.colorbar(contour)
# cbar.set_label('Dimensionless Temperature, $\\theta$(-)')
# plt.show()

# cA vs. z at t = 1.0
# plt.figure()
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.plot(z, ca_full, '-k')
# plt.xlabel('Dimensionless Distance, z(-)')
# plt.ylabel('Dimensionless Concentration, $c_A$(-)')
# plt.legend()
# plt.show()

# # cB vs. z at t = 1.0
# plt.figure()
# plt.xlim(0, 1)
# plt.ylim(None, 1)
# plt.plot(z, cb_full, '-k')
# plt.xlabel('Dimensionless Distance, z(-)')
# plt.ylabel('Dimensionless Concentration, $c_B$(-)')
# plt.show()

# # T vs. z at t = 1.0
# plt.figure()
# plt.xlim(0, 1)
# plt.plot(z, T_full, '-k')
# plt.xlabel('Dimensionless Distance, z(-)')
# plt.ylabel('Dimensionless Temperature, $\\theta$(-)')
# plt.show()

# # cA vs. t at z = 0, 0.5, 1
# plt.figure()
# plt.plot(t, ca_time[:, j_in],  label='z = 0')
# plt.plot(t, ca_time[:, j_mid], label='z = 0.5')
# plt.plot(t, ca_time[:, j_out], label='z = 1')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('Dimensionless Time, t(-)')
# plt.ylabel('Dimensionless Concentration, $c_A$(-)')
# plt.legend()
# plt.show()

# # cB vs. t at z = 0, 0.5, 1
# plt.figure()
# plt.plot(t, cb_time[:, j_in],  label='z = 0')
# plt.plot(t, cb_time[:, j_mid], label='z = 0.5')
# plt.plot(t, cb_time[:, j_out], label='z = 1')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('Dimensionless Time, t(-)')
# plt.ylabel('Dimensionless Concentration, $c_B$(-)')
# plt.legend()
# plt.show()

# # T vs. t at z = 0, 0.5, 1
# plt.figure()
# plt.plot(t, T_time[:, j_in],  label='z = 0')
# plt.plot(t, T_time[:, j_mid], label='z = 0.5')
# plt.plot(t, T_time[:, j_out], label='z = 1')
# plt.xlim(0, 1)
# plt.xlabel('Dimensionless Time, t(-)')
# plt.ylabel('Dimensionless Temperature, $\\theta$(-)')
# plt.legend()
# plt.show()
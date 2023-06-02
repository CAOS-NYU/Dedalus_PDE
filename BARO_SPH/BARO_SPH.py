"""
Dedalus script simulating the viscous shallow water equations on a sphere. This
script demonstrates solving an initial value problem on the sphere. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_sphere.py` script can be used to produce
plots from the saved data. The simulation should about 5 cpu-minutes to run.

The script implements the test case of a barotropically unstable mid-latitude
jet from Galewsky et al. 2004 (https://doi.org/10.3402/tellusa.v56i5.14436).
The initial height field balanced the imposed jet is solved with an LBVP.
A perturbation is then added and the solution is evolved as an IVP.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shallow_water.py
    $ mpiexec -n 4 python3 plot_sphere.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
import os
logger = logging.getLogger(__name__)

output_root = '/scratch/op13/dedalus/BARO_SPH/'


# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# Parameters
#Nphi = 256
#Ntheta = 128

Nphi = 512
Ntheta = 256
dealias = 3/2
R = 6.37122e6 * meter
Omega = 7.292e-5 / second
nu = 1e5 * meter**2 / second / 32**2 # Hyperdiffusion matched at ell=32
timestep = 600 * second
stop_sim_time = 720 * hour
dtype = np.float64

expname = 'BARO_SPH_'+str(Nphi)+'_'+str(Ntheta)
output_dir = output_root+expname
if (not os.path.exists(output_dir)):
    os.makedirs(output_dir, exist_ok = True)

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
Psi = dist.Field(name='Psi', bases=basis)
q0 = dist.Field(name='q0', bases=basis)
pvort = dist.Field(name='pvort', bases=basis)


# Initial conditions: zonal jet
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
umax = 80 * meter / second
lat0 = np.pi / 7
lat1 = np.pi / 2 - lat0
en = np.exp(-4 / (lat1 - lat0)**2)
jet = (lat0 <= lat) * (lat <= lat1)
u_jet = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))
u['g'][0][jet]  = u_jet

# Initial conditions: vorticity perturbation
lat2 = np.pi / 4
qpert = 0.1*Omega 
alpha = 1 / 3
beta = 1 / 15
q0['g'] += qpert * np.cos(lat) * np.exp(-(phi/alpha)**2) * np.exp(-((lat2-lat)/beta)**2)
pvort['g'] = 2* Omega * np.sin(lat)

# Initial conditions: solve for streamfunction
c = dist.Field(name='c')
problem = d3.LBVP([Psi, c], namespace=locals())
problem.add_equation("lap(Psi) + c = -d3.div(d3.skew(u)) + q0")
problem.add_equation("ave(Psi) = 0")
solver = problem.build_solver()
solver.solve()

# Problem
q = d3.lap(Psi)
u = d3.skew(d3.grad(Psi))
problem = d3.IVP([Psi, c], namespace=locals())
problem.add_equation("dt(q) + nu*lap(lap(q))  + 2*Omega*div(d3.MulCosine(u)) + c = - u@grad(q)")
problem.add_equation("ave(Psi) = 0 ")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler(output_dir + '/snapshots', sim_dt=1*hour, max_writes=10)
snapshots.add_task(Psi*second/meter/meter, name='Psi')
snapshots.add_task(-d3.div(d3.skew(u))*second, name='vorticity')
snapshots.add_task(u*second/meter, name='velocity')
snapshots.add_task((pvort -d3.div(d3.skew(u)))*second, name='absolute vorticity')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()


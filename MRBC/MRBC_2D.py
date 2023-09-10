"""
Dedalus script simulating 2D horizontally-periodic Moist Rayleigh-Benard convection.
This script is based on the 2D Rayliegh-Benrard convection example from the Dedalus Project.
It solves a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. 

In contrast to the regular RBC problem, the moist version has two conserved tracer:
- a moist buyancy M for saturated parcell;
- a dry buoyancy for unsaturated parrcel. 
The actual buoyancy of a parcel is determined by the expression

B = max (M, D-N^2 z)

where N is the brunt-Vaisala frequency of a moist adiabat. See Pauluis and Schumacher 
(2010) for a more detailed description of the model. 

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and moist Rayleigh numbers as:

    kappa = (MRayleigh * Prandtl)**(-1/2)
    nu = (MRayleigh / Prandtl)**(-1/2)

In this version, we also set the dry buoyancy at the top, and assume that the linear profile is 
at saturation, i.e. M(z) = D(z) - N^2 z.  

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
import sys
import os
logger = logging.getLogger(__name__)


# Default Parameters

Lx, Lz = 32, 1 #domain size
Nx, Nz = 1024, 32 #number of spectral components in each direction

MRayleigh = 4.e5
M0 = 0 # boundary condition for M at z=0
D0 = 0 # boundary condition for D at z=0
MH = -1.0 # boundary condition for M at z= Lz
DH = 0.5 # boundary condition for D at z= Lz
N2 = 1.5 #saturared brunt vaisala frequency used in definition of buoyancy
U0 = 0.0 #BC f0r u at z =0
UH = 0.0 #BC for u at z = Lz
Q0 = 0.0 #cooling rate
Chi = 2.0 # ratio of cooling tendency on D and M
free_slip_lower = False #switch to free slip BC for horizontal wind at z= 0 
free_slip_upper = False #switch to free slip BC for horizontal wind at z = Lz

Prandtl = 1 #Prandtl number

dealias = 3/2 # scaling for grid to alaias computation of non-linear terms
stop_sim_time = 200.0 #simulation time in non-dimensional unit (t = 1 ~ vertical advection time scale)
timestepper = d3.RK222
max_timestep = 0.125
initial_timestep = 0.015
dtype = np.float64

# The code check for the existence of a MRBC2D_param.py and import it. This allows yo update the simulations parameters

if ( os.path.isfile('./MRBC2D_param.py')):
    from MRBC2D_param import *

DRayleigh= MRayleigh * (DH-D0) / (MH-M0) # Dry Rayleight number

# Viscosity and diffusivity based on Rayleigh and PRandtl numbers
kappa = (MRayleigh * Prandtl)**(-1/2)
nu = (MRayleigh / Prandtl)**(-1/2)

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
m = dist.Field(name='m', bases=(xbasis,zbasis))
d = dist.Field(name='d', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))

#tau fields to ensure quadratic matrix on the LHS
tau_p = dist.Field(name='tau_p')
tau_d1 = dist.Field(name='tau_d1', bases=xbasis)
tau_d2 = dist.Field(name='tau_d2', bases=xbasis)
tau_m1 = dist.Field(name='tau_m1', bases=xbasis)
tau_m2 = dist.Field(name='tau_m2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

#additional definition
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_m = d3.grad(m) + ez*lift(tau_m1) # First-order reduction
grad_d = d3.grad(d) + ez*lift(tau_d1) # First-order reduction

u_x = u @ ex
u_z = u @ ez
dzu_x = d3.Differentiate(u_x, coords['z'])

Q = dist.Field(name='Q', bases=(xbasis,zbasis))
Q['g'] = Q0 * np.sin(np.pi * z/Lz)


N2z = dist.Field(name='N2z', bases=(xbasis,zbasis))
N2z['g']= N2 * z
b_corr = 0.5 * (m - d + N2z+ np.abs(m - d + N2z) )  # coding max function - np.maximum is not available in dedalus


# Problem

problem = d3.IVP([p, m, d, u, tau_p, tau_m1, tau_m2, tau_d1,tau_d2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(m)  - kappa*div(grad_m) + lift(tau_m2)  = - (u)@grad(m) - Q / Chi")
problem.add_equation("dt(d)  - kappa*div(grad_d) + lift(tau_d2)  = - (u)@grad(d) - Q")
problem.add_equation("dt(u)  - nu*div(grad_u) + grad(p) - d*ez  + lift(tau_u2) = - u@grad(u) + b_corr * ez")
#BC for horiontal wind
if free_slip_lower:
    problem.add_equation("dzu_x(z=0) = 0")
else:
    problem.add_equation("u_x(z=0) = U0")
if free_slip_upper:
    problem.add_equation("dzu_x(z=Lz) = 0")
else: 
    problem.add_equation("u_x(z=Lz) = UH")
#BC for vertical wind
problem.add_equation("u_z(z=0) = 0")
problem.add_equation("u_z(z=Lz) = 0")
#BC for M and D
problem.add_equation("m(z=0) = M0")
problem.add_equation("d(z=0) = D0")
problem.add_equation("m(z=Lz) = MH")
problem.add_equation("d(z=Lz) = DH")
# Pressure gauge
problem.add_equation("integ(p) = 0") 

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time


# Initial conditions
u['g'][0,:,:]=U0 + (UH-U0) * z/Lz

m.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
m['g'] *= z * (Lz - z) # Damp noise at walls
m['g'] += M0 + MH * z # Add linear background
m['g'] += 0.01 * z * (Lz - z) * np.sin(2.0 * np.pi * x / Lx)

d.fill_random('g', seed=41, distribution='normal', scale=1e-3) # Random noise
d['g'] *= z * (Lz - z) # Damp noise at walls
d['g'] += D0 + DH * z # Add linear background
d['g'] += 0.01 * z * (Lz - z) * np.sin(2.0 * np.pi * x / Lx)

# Analysis
snapshots = solver.evaluator.add_file_handler(output_dir + '/snapshots', sim_dt=1.0, max_writes=20)
snapshots.add_task(m, name='moist buoyancy')
snapshots.add_task(d, name='dry buoyancy')
snapshots.add_task( u, name='velocity')

analysis = solver.evaluator.add_file_handler(output_dir + '/analysis', sim_dt=0.1, max_writes=1000)
analysis.add_task(d3.Average(u @ ez * m,'x'),layout='g', name='moist buoyancy flux')
analysis.add_task(d3.Average(u @ ez * d,'x'),layout='g', name='dry buoyancy flux')
analysis.add_task(d3.Average(u @ ez * (d + b_corr),'x'),layout='g', name='buoyancy flux')


# CFL
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=1, safety=0.3, threshold=0.05,
             max_change=1.5, min_change=0.25, max_dt=max_timestep)
CFL.add_velocity(u+ubar)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 50 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

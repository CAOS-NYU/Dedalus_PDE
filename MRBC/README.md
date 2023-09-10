Solving Moist Rayleigh Benard convection with Dedalus

The code solves the Moist Rayligih-Benard equations introduced by Pauluis and Schumacher (2010).. The corresponding equations are:
$$\frac{D {\mathbf U}}{DT} = -\nu \Delta {\mathbf U} - \nabla p + B \mathbf{k}$$
$$\frac{D M}{DT} = - \nu \Delta M$$
$$\frac{D D}{DT} = - \nu \Delta D$$
$$\nabla \cdot {\mathbf U} =0$$
$$B = \mathrm{max}(M,D-N^2z).$$
The equations are solved on a two or three-dimensional domain. The boundary conditions used are periodic in the horizontal directions and rigid lid at $z=0$ and $z=1$. More specifically, the default boundary conditions are
$$\mathbf{U}(z=0) =0 $$
$$\mathbf{U}(z=1) =0 $$
$$M(z=0) = 0$$
$$M(z=1) = -1$$
$$D(z=0) = 0 $$
$$D(1=1) = -1 + N^2.$$ 
While other choices for the boundary conditions are possible, these  correspond to the ones from  Pauluis and Schumacher (2010). 

The repository here contains four pieces of code:
- `MRBC_2D.py` and `MRBC_3D.py` implement the MRBC problem in Dedalus. 
- `launch_MRBC_2D.ipynb` and `launch_MRBC_3D.ipynb` are Jupyter Notebooks that organize the computate directory and create the slurm scripts.
- `Read_MRBC_2D.ipynb` and `Read_MRBC_3D.ipynb` are short Jupyer Notebooks that read the output files

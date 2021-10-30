import numpy as np
import os
import bempp.api
from mesh_converter import *
from force_calculation import *
import inspect
from scipy.sparse.linalg import gmres


# Lectura y creacion de mallado
# directory = '/home/ian/Desktop/Forces_bioelectrostatics' #Ubuntu
directory = 'C:\\Users\\ian\Desktop\\forces_calculation' #Windows
protein = 'sphere'
forcefield = 'nosymmetric'
#dir_prot = directory+'/pqr_files/'+protein #Ubuntu
dir_prot = directory+'\\pqr_files\\'+protein #Windows
density = 6.0
probe_radius = 1.4
pf = protein +'_' + forcefield
pfd = protein +'_' + forcefield + '_' +'d'+str(density)[::2]
convert_pqr2xyzr('{}/{}.pqr'.format(dir_prot,pf),'{}/{}.xyzr'.format(dir_prot,pf))
generate_nanoshaper_mesh('{}/{}.xyzr'.format(dir_prot,pf),dir_prot,pf,pfd,density,probe_radius,False)
grid = import_msms_mesh('{}/{}.face'.format(dir_prot,pfd),'{}/{}.vert'.format(dir_prot,pfd))

#Parametros del medio
q, x_q = np.array([]), np.empty((0,3))
ep_in = 4.
ep_ex = 80.
k = 0.125

#Leer cargas y coordenadas desde archivo .pqr
molecule_file = open('{}/{}.pqr'.format(dir_prot,pf), 'r').read().split('\n')
for line in molecule_file:
    line = line.split()
    if len(line)==0 or line[0]!='ATOM': continue
    q = np.append( q, float(line[8]))
    x_q = np.vstack(( x_q, np.array(line[5:8]).astype(float) ))

#Crear lado derecho de la ecuacion integral de frontera
dirichl_space = bempp.api.function_space(grid, "DP", 0)
neumann_space = bempp.api.function_space(grid, "DP", 0)
@bempp.api.real_callable
def charges_fun(x, n, domain_index, result):
    global q, x_q, ep_in
    suma = 0
    for k in range(len(q)):
        suma = suma + q[k]/(np.linalg.norm(x-x_q[k]))
    result[:] = suma/(4*np.pi*ep_in)
    #result[:] = np.sum(q / np.linalg.norm(x - x_q, axis=1)) #old    
charged_grid_fun = bempp.api.GridFunction(dirichl_space, fun=charges_fun)
#charged_grid_fun.plot()
rhs = np.concatenate([charged_grid_fun.coefficients, np.zeros(neumann_space.global_dof_count)])

# Define Operators
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
slp_in   = laplace.single_layer(neumann_space, dirichl_space, dirichl_space)
dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space)
slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, k)
dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, k)
# Matrix Assembly
blocked = bempp.api.BlockedOperator(2, 2)
blocked[0, 0] = 0.5*identity + dlp_in
blocked[0, 1] = -slp_in
blocked[1, 0] = 0.5*identity - dlp_out
blocked[1, 1] = (ep_in/ep_ex)*slp_out
op_discrete = blocked.strong_form()

#Solver (GMRES)
array_it, array_frame, it_count = np.array([]), np.array([]), 0
def iteration_counter(x):
        global array_it, array_frame, it_count
        it_count += 1
        frame = inspect.currentframe().f_back
        array_it = np.append(array_it, it_count)
        array_frame = np.append(array_frame, frame.f_locals["resid"])
        #print "It: {0} Error {1:.2E}".format(it_count, frame.f_locals["resid"])        
x, info = gmres(op_discrete, rhs, callback=iteration_counter, tol=1e-3, maxiter=1000, restart = 2000)
print("The linear system was solved in {0} iterations".format(it_count))
solution_dirichl = bempp.api.GridFunction(dirichl_space, coefficients=x[:dirichl_space.global_dof_count])
solution_neumann = bempp.api.GridFunction(neumann_space, coefficients=x[dirichl_space.global_dof_count:])

#Potential in domain
slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_q.transpose())
dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_q.transpose())
phi_q = slp_q*solution_neumann - dlp_q*solution_dirichl

cal2J = 4.184
qe = 1.60217646e-19
Na = 6.0221415e+23
ep_vacc = 8.854187818e-12
C0 = qe**2*Na*1e-3*1e10/(cal2J*ep_vacc)

# Reaction force calculation (qE)
h=0.001
grad_phi = solvent_potential_first_derivate(x_q, h, neumann_space, dirichl_space, solution_neumann, solution_dirichl)
F_reac = np.zeros([len(q),3])
for j in range(len(q)):
    F_reac[j,:] = -q[j]*grad_phi[j,:]
F_reactotal = np.zeros([3])
for j in range(len(q)):
    F_reactotal[:] = F_reactotal[:] + F_reac[j,:]
F_reactotal[:] = 4.184*4*np.pi*332.064*F_reactotal[:]
# Dielectric boundary force calculation (DBF)

grad_phi = solution_neumann.coefficients
f_dbf = np.zeros([grid.number_of_elements,3])
for j in range(grid.number_of_elements):
    f_dbf[j,:] = (grad_phi[j]**2)*grid.normals[j]*grid.volumes[j]
f_dbftotal = np.zeros([3])
for j in range(grid.number_of_elements):
    f_dbftotal[:] = f_dbftotal[:] + f_dbf[j,:]
f_dbftotal[:] = -4.184*0.5*332.064*(ep_ex-ep_in)*f_dbftotal[:]
#grid.number_of_elements
#grid.normals[1]
f_dbftotal
# Ionic boundary force calculation (IBF)
phi = solution_dirichl.coefficients
auxi = k*phi**2
f_ibf = np.zeros([grid.number_of_elements,3])
for j in range(grid.number_of_elements):
    f_ibf[j] = auxi[j]*grid.normals[j]*grid.volumes[j]
f_ibftotal = np.zeros([3])
for j in range(grid.number_of_elements):
    f_ibftotal[:] = f_ibftotal[:] + f_ibf[j,:]
f_ibftotal[:] = -4.184*0.5*332.064*(ep_ex)*f_ibftotal[:]


    
print("Total reaction force: {:10.4f}{:10.4f}{:10.4f} [kJ/molA]".format(F_reactotal[0],F_reactotal[1],F_reactotal[2]))
print("Total dielectric boundary force: {:10.4f}{:10.4f}{:10.4f} [kJ/molA]".format(f_dbftotal[0],f_dbftotal[1],f_dbftotal[2]))
print("Total ionic boundary force: {:5.3e} {:5.3e} {:5.3e} [kJ/molA]".format(f_ibftotal[0],f_ibftotal[1],f_ibftotal[2]))
# total dissolution energy applying constant to get units [kcal/mol]
total_energy = 4.184*2*np.pi*332.064*np.sum(q*phi_q).real
print("Total solvation energy: {:7.2f} [kJ/Mol]".format(total_energy))

grid.plot()
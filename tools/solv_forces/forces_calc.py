import numpy as np
import os
import bempp.api


def solvent_potential_first_derivate(dirichl_space, neumann_space, solution_dirichl, solution_neumann, xq, h):
    
    #Function from https://github.com/bem4solvation/Multipoles_calculations_python
    """
    Compute the first derivate of potential due to solvent
    in the position of the points
    Inputs:
    -------
        xq: Array size (Nx3) whit positions to calculate the derivate.
        h: Float number, distance for the central difference.
        neumann_space: Bempp Function space
        dirichl_space: Bempp Function space
        solution_neumann: Data of dphi in boundary
        solution_dirichl: Data of phi in boundary

    Return:

    dpdr: Derivate of the potential in the points positions.
    """
    dpdr = np.zeros([len(xq), 3]) #ceros de len(xq) filas y 3 columnas
    dist = np.array(([h,0,0],[0,h,0],[0,0,h])) #matriz 3x3 diagonal de h
    
    # x axis derivate
    dx = xq[:] + dist[0] # vector x + h
    dx = np.concatenate((dx, xq[:] - dist[0])) # vector x+h y luego x-h
    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dx.transpose()) #Evaluacion SL en x+h?
    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dx.transpose()) #Evaluacion DL en x+h?
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl) #Evaluacion de valores adentro
    # y el valor de la sumatoria de cargas???, tiene la resta incluida
    dpdx = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,0] = dpdx

    #y axis derivate
    dy = xq[:] + dist[1]
    dy = np.concatenate((dy, xq[:] - dist[1]))
    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dy.transpose())
    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dy.transpose())
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    dpdy = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,1] = dpdy

    #z axis derivate
    dz = xq[:] + dist[2]
    dz = np.concatenate((dz, xq[:] - dist[2]))
    slpo = bempp.api.operators.potential.laplace.single_layer(neumann_space, dz.transpose())
    dlpo = bempp.api.operators.potential.laplace.double_layer(dirichl_space, dz.transpose())
    phi = slpo.evaluate(solution_neumann) - dlpo.evaluate(solution_dirichl)
    dpdz = 0.5*(phi[0,:len(xq)] - phi[0,len(xq):])/h
    dpdr[:,2] = dpdz

    return dpdr
 
def fixed_charge_forces(solution_dirichl,solution_neumann,dirichl_space,neumann_space,x_q,q,h=0.001):
    
    convert_to_kcalmolA = 4 * np.pi * 332.0636817823836 #1e-3*Na*1e10*(qe**2/(ep_vacc*4*numpy.pi*cal2J))
    kcal_to_kJ = 4.184

    #fixed_charge_forces calculations
    grad_phi = solvent_potential_first_derivate(dirichl_space, neumann_space, solution_dirichl, solution_neumann, x_q, h)
    f_reac = np.zeros([len(q),3])
    for j in range(len(q)):
        f_reac[j,:] = -q[j]*grad_phi[j,:]
    f_reactotal = np.zeros([3])
    for j in range(len(q)):
        f_reactotal[:] = f_reactotal[:] + f_reac[j,:]
    f_reactotal[:] = kcal_to_kJ * convert_to_kcalmolA * f_reactotal[:]
    f_reac[:] = kcal_to_kJ * convert_to_kcalmolA * f_reac[:]
    Efield = -convert_to_kcalmolA*kcal_to_kJ*grad_phi

    return f_reactotal, f_reac, Efield

def boundary_forces(solution_dirichl,solution_neumann,grid,ep_in,ep_ex,k):

    convert_to_kcalmolA = 4 * np.pi * 332.0636817823836 #1e-3*Na*1e10*(qe**2/(ep_vacc*4*numpy.pi*cal2J))
    kcal_to_kJ = 4.184

    #Dielectric boundary force
    grad_phi = solution_neumann.coefficients[:]
    f_db = np.zeros([3])
    for j in range(grid.number_of_elements):
        f_db += (ep_in/ep_ex)*(grad_phi[j]**2)*grid.normals[j]*grid.volumes[j]
    f_db = - kcal_to_kJ * 0.5 * convert_to_kcalmolA * (ep_ex-ep_in) *f_db

    #Ionic boundary force
    phi = solution_dirichl.coefficients[:]
    f_ib = np.zeros([3])
    for j in range(grid.number_of_elements):
        f_ib += (k**2)*(phi[j]**2)*grid.normals[j]*grid.volumes[j]
    f_ib = - kcal_to_kJ * 0.5 * convert_to_kcalmolA * (ep_ex)*f_ib

    return f_db,f_ib

import numpy as np
import os
import bempp.api


def solvent_potential_first_derivate(xq, h, neumann_space, dirichl_space, solution_neumann, solution_dirichl):
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

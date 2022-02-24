import numpy as np
import os
import bempp.api
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

def coefficient_matrix(dirichl_space,neumann_space,ep_in,ep_ex,kappa):

    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dirichl_space)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space)
    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa)
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa)

    #Crear matriz del sistema lineal
    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = 0.5*identity + dlp_in
    blocked[0, 1] = -slp_in
    blocked[1, 0] = 0.5*identity - dlp_out
    blocked[1, 1] = (ep_in/ep_ex)*slp_out
    op_discrete = blocked.strong_form()

    return op_discrete

def coefficient_matrix_two(dirichl_space1,neumann_space1,dirichl_space2,neumann_space2,ep_in,ep_ex,kappa):

    #Operadores de Laplace para ecuaciones de Poisson para los solutos
    identity1 = sparse.identity(dirichl_space1, dirichl_space1, dirichl_space1)
    slp_in11   = laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1)
    dlp_in11   = laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1)
    slp_in22   = laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2)
    dlp_in22   = laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2)

    #Operadores de Yukawa para ecuacion de Poisson-Boltzmann para el solvente
    identity2 = sparse.identity(dirichl_space2, dirichl_space2, dirichl_space2)
    slp_out11  = modified_helmholtz.single_layer(neumann_space1, dirichl_space1, dirichl_space1, kappa)
    slp_out12  = modified_helmholtz.single_layer(neumann_space1, dirichl_space2, dirichl_space2, kappa)
    slp_out21  = modified_helmholtz.single_layer(neumann_space2, dirichl_space1, dirichl_space1, kappa)
    slp_out22  = modified_helmholtz.single_layer(neumann_space2, dirichl_space2, dirichl_space2, kappa)
    dlp_out11  = modified_helmholtz.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, kappa)
    dlp_out12  = modified_helmholtz.double_layer(dirichl_space1, dirichl_space2, dirichl_space2, kappa)
    dlp_out21  = modified_helmholtz.double_layer(dirichl_space2, dirichl_space1, dirichl_space1, kappa)
    dlp_out22  = modified_helmholtz.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, kappa)

    #Creacion de Matriz del sistema lineal
    blocked = bempp.api.BlockedOperator(4, 4)
    blocked[0, 0] = 0.5*identity1 + dlp_in11
    blocked[0, 1] = -slp_in11
    blocked[1, 0] = 0.5*identity1 - dlp_out11
    blocked[1, 1] = (ep_in/ep_ex)*slp_out11
    blocked[1, 2] = -dlp_out21
    blocked[1, 3] = (ep_in/ep_ex)*slp_out21
    blocked[2, 0] = - dlp_out12
    blocked[2, 1] = (ep_in/ep_ex)*slp_out12
    blocked[2, 2] = 0.5*identity2-dlp_out22
    blocked[2, 3] = (ep_in/ep_ex)*slp_out22
    blocked[3, 2] = 0.5*identity2 + dlp_in22
    blocked[3, 3] = -slp_in22
    op_discrete = blocked.strong_form()

    return op_discrete

def calc_rhs(dirichl_space,neumann_space,q,x_q,ep_in):

    @bempp.api.real_callable
    def charges_fun(x, n, domain_index, result):
        suma = 0
        for k in range(len(q)):
            suma = suma + q[k]/(np.linalg.norm(x-x_q[k]))
        result[:] = suma/(4*np.pi*ep_in)
    charged_grid_fun = bempp.api.GridFunction(dirichl_space, fun=charges_fun)
    rhs = np.concatenate([charged_grid_fun.coefficients, np.zeros(neumann_space.global_dof_count)])

    return rhs

def calc_rhs_two(dirichl_space1,neumann_space1,dirichl_space2,neumann_space2,\
    q1,q2,x_q1,x_q2,ep_in):

    @bempp.api.real_callable
    def charges_fun(x, n, domain_index, result):
        suma = 0
        for k in range(len(q1)):
            suma = suma + q1[k]/(np.linalg.norm(x-x_q1[k]))
        result[:] = suma/(4*np.pi*ep_in)

    @bempp.api.real_callable
    def charges_fun2(x, n, domain_index, result):
        suma = 0
        for k in range(len(q2)):
            suma = suma + q2[k]/(np.linalg.norm(x-x_q2[k]))
        result[:] = suma/(4*np.pi*ep_in)
    
    charged_grid_fun1 = bempp.api.GridFunction(dirichl_space1, fun=charges_fun)
    charged_grid_fun2 = bempp.api.GridFunction(dirichl_space2, fun=charges_fun2)

    rhs = np.concatenate([charged_grid_fun1.coefficients, np.zeros(neumann_space1.global_dof_count),\
        np.zeros(neumann_space2.global_dof_count),charged_grid_fun2.coefficients])

    return rhs


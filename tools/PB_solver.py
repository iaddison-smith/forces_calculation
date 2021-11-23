import numpy as np
import os
import bempp.api
from tools.mesh_converter import *
from tools.force_calculation import *
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import inspect
from scipy.sparse.linalg import gmres


class Molecule:

    """
    Simulacion PBE linealizadas con calculo de fuerzas incluidos
        directory_name = directorio donde se guarda el repositorio
        protein_name = proteina a simular (Ejemplo: arg)
        forcefield_name = campo de fuerzas para la generacion del .pqr
        density_grid = Grid scale en el caso de ocupar nanoshaper
        external_grid = ocupar malla ya generada en vez de Nanoshaper
        
    """

    def __init__(self, dir_name, pname, ffname, gs, external_grid = 'n', ep_in=4, ep_ex=80, kappa=0.125):
        
        #Crear malla de la proteina
        if external_grid == 'n':
            self.grid, q, x_q = pqrtomesh(directory=dir_name, protein=pname, forcefield=ffname, density=gs, probe_radius=1.4)
        else:
            self.grid, q, x_q = pqrtomesh(directory=dir_name, protein=pname, forcefield=ffname, density=gs, probe_radius=1.4, build_mesh='n')
        
        #Crear espacio de funciones y lado derecho del sistema de ecuaciones
        dirichl_space = bempp.api.function_space(self.grid, "DP", 0)
        neumann_space = bempp.api.function_space(self.grid, "DP", 0)
        @bempp.api.real_callable
        def charges_fun(x, n, domain_index, result):
            suma = 0
            for k in range(len(q)):
                suma = suma + q[k]/(np.linalg.norm(x-x_q[k]))
            result[:] = suma/(4*np.pi*ep_in)
        charged_grid_fun = bempp.api.GridFunction(dirichl_space, fun=charges_fun)
        rhs = np.concatenate([charged_grid_fun.coefficients, np.zeros(neumann_space.global_dof_count)])

        #Definir operadores matriz coeficientes
        identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
        slp_in   = laplace.single_layer(neumann_space, dirichl_space, dirichl_space)
        dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space)
        slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa)
        dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa)

        # Matrix Assembly
        blocked = bempp.api.BlockedOperator(2, 2)
        blocked[0, 0] = 0.5*identity + dlp_in
        blocked[0, 1] = -slp_in
        blocked[1, 0] = 0.5*identity - dlp_out
        blocked[1, 1] = (ep_in/ep_ex)*slp_out
        op_discrete = blocked.strong_form()

        #Resolucion del sistema de ecuaciones con GMRES
        x, info = gmres(op_discrete, rhs, tol=1e-3, maxiter=1000, restart = 2000)

        #Guardar potencial electrico y su derivada normal
        self.phi = bempp.api.GridFunction(dirichl_space, coefficients=x[:dirichl_space.global_dof_count])
        self.dphidn = bempp.api.GridFunction(neumann_space, coefficients=x[dirichl_space.global_dof_count:])

        #Calcular el potencial electrico en los puntos donde estan las cargas
        slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_q.transpose())
        dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_q.transpose())
        phi_q = slp_q*self.dphidn - dlp_q*self.phi
        self.f_solv = np.zeros([3])
        #Calcular las fuerzas de solvatacion para la moleculas
        convert_to_kcalmolA = 4 * np.pi * 332.0636817823836 #1e-3*Na*1e10*(qe**2/(ep_vacc*4*numpy.pi*cal2J))
        kcal_to_kJ = 4.184
        self.f_qf, self.f_qf_charges = fixed_charge_forces(self.dphidn,self.phi,neumann_space,dirichl_space,x_q,q)
        self.f_db,self.f_ib = boundary_forces(self.dphidn,self.phi,self.grid,kappa,ep_ex,ep_in)
        self.f_solv = self.f_qf+self.f_db+self.f_ib
        self.solv_energy = 0.5 * kcal_to_kJ * convert_to_kcalmolA * np.sum(q*phi_q).real

    def save_info(self, dir, pname, fname, gs):

        results_file = open(dir + '\\results\\results_' + pname + '_' + fname + '_' + str(gs) +'.txt', 'w')

        results_file.write('Results for ' +pname+'_'+fname +'\n')
        results_file.write('Grid scale: '+ str(gs) + '\n')
        results_file.write('Number of elements: '+str(self.grid.number_of_elements) + '\n')
        results_file.write('Total surface area (A^2): '+str(np.sum(self.grid.volumes)) + '\n')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write('Solvation energy for complex (kJ/mol): '+ str(self.solv_energy)+ '\n')
        results_file.write('Total solvation forces for complex (kJ/molA): ' + str(self.f_solv)+ '\n')
        results_file.write('Total fixed charge forces for complex (kJ/molA): ' + str(self.f_qf)+ '\n')
        results_file.write('Total dielectric boundary force for complex (kJ/molA): ' + str(self.f_db)+ '\n')
        results_file.write('Total ionic boundary force for complex (kJ/molA): ' + str(self.f_ib)+ '\n')

        results_file.close()

        return None

class Two_proteins:

    """
    Simulacion PBE linealizadas entre 2 proteinas con calculo de fuerzas incluidos
        directory_name = directorio donde se guarda el repositorio
        protein_name = proteina a simular (Ejemplo: arg)
        forcefield_name = campo de fuerzas para la generacion del .pqr
        density_grid = Grid scale en el caso de ocupar nanoshaper
        external_grid = ocupar malla ya generada en vez de Nanoshaper
        
    """

    def __init__(self, dir_name, pname1, ffname1, gs1, pname2, ffname2, gs2, external_grid = 'n', ep_in=4, ep_ex=80, kappa=0.125):
        
        #Crear malla de la proteina
        if external_grid == 'n':
            self.grid1, q1, x_q1 = pqrtomesh(directory=dir_name, protein=pname1, forcefield=ffname1, density=gs1, probe_radius=1.4)
            self.grid2, q2, x_q2 = pqrtomesh(directory=dir_name, protein=pname2, forcefield=ffname2, density=gs2, probe_radius=1.4)
        else:
            self.grid1, q1, x_q1 = pqrtomesh(directory=dir_name, protein=pname1, forcefield=ffname1, density=gs1, probe_radius=1.4, build_mesh='n')
            self.grid2, q2, x_q2 = pqrtomesh(directory=dir_name, protein=pname2, forcefield=ffname2, density=gs2, probe_radius=1.4, build_mesh='n')
        
        #Crear espacio de funciones y lado derecho del sistema de ecuaciones
        dirichl_space1 = bempp.api.function_space(self.grid1, "DP", 0)
        neumann_space1 = bempp.api.function_space(self.grid1, "DP", 0)
        dirichl_space2 = bempp.api.function_space(self.grid2, "DP", 0)
        neumann_space2 = bempp.api.function_space(self.grid2, "DP", 0)

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

        rhs = np.concatenate([charged_grid_fun1.coefficients, np.zeros(neumann_space1.global_dof_count),  
                      np.zeros(neumann_space2.global_dof_count),charged_grid_fun2.coefficients])

        #Poisson equation operators
        identity1 = sparse.identity(dirichl_space1, dirichl_space1, dirichl_space1)
        slp_in11   = laplace.single_layer(neumann_space1, dirichl_space1, dirichl_space1)
        dlp_in11   = laplace.double_layer(dirichl_space1, dirichl_space1, dirichl_space1)
        slp_in22   = laplace.single_layer(neumann_space2, dirichl_space2, dirichl_space2)
        dlp_in22   = laplace.double_layer(dirichl_space2, dirichl_space2, dirichl_space2)
        #Poisson-Boltzmann equations operators
        identity2 = sparse.identity(dirichl_space2, dirichl_space2, dirichl_space2)
        slp_out11  = modified_helmholtz.single_layer(neumann_space1, dirichl_space1, dirichl_space1, kappa)
        slp_out12  = modified_helmholtz.single_layer(neumann_space1, dirichl_space2, dirichl_space2, kappa)
        slp_out21  = modified_helmholtz.single_layer(neumann_space2, dirichl_space1, dirichl_space1, kappa)
        slp_out22  = modified_helmholtz.single_layer(neumann_space2, dirichl_space2, dirichl_space2, kappa)
        dlp_out11  = modified_helmholtz.double_layer(dirichl_space1, dirichl_space1, dirichl_space1, kappa)
        dlp_out12  = modified_helmholtz.double_layer(dirichl_space1, dirichl_space2, dirichl_space2, kappa)
        dlp_out21  = modified_helmholtz.double_layer(dirichl_space2, dirichl_space1, dirichl_space1, kappa)
        dlp_out22  = modified_helmholtz.double_layer(dirichl_space2, dirichl_space2, dirichl_space2, kappa)

        # Matrix Assembly
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

        #Resolucion del sistema de ecuaciones con GMRES
        x, info = gmres(op_discrete, rhs, tol=1e-3, maxiter=1000, restart = 2000)

        #Guardar potencial electrico y su derivada para cada una de las proteinas
        a = dirichl_space1.global_dof_count
        a1 = neumann_space1.global_dof_count
        b = dirichl_space2.global_dof_count
        b1 = neumann_space2.global_dof_count

        self.phi1 = bempp.api.GridFunction(dirichl_space1, coefficients=x[:a])
        self.dphi1dn = bempp.api.GridFunction(neumann_space1, coefficients=x[a:(a+a1)])                         
        self.phi2 = bempp.api.GridFunction(dirichl_space2, coefficients=x[(a+a1):(a+a1+b)])
        self.dphi2dn = bempp.api.GridFunction(neumann_space2, coefficients=x[(a+a1+b):])

        #Calcular el potencial electrico en los puntos donde estan las cargas

        slp_q1 = bempp.api.operators.potential.laplace.single_layer(neumann_space1, x_q1.transpose())
        dlp_q1 = bempp.api.operators.potential.laplace.double_layer(dirichl_space1, x_q1.transpose())
        phi_q1 = slp_q1*self.dphi1dn - dlp_q1*self.phi1

        slp_q2 = bempp.api.operators.potential.laplace.single_layer(neumann_space2, x_q2.transpose())
        dlp_q2 = bempp.api.operators.potential.laplace.double_layer(dirichl_space2, x_q2.transpose())
        phi_q2 = slp_q2*self.dphi2dn - dlp_q2*self.phi2

        #Calcular las fuerzas de solvatacion para la moleculas
        convert_to_kcalmolA = 4 * np.pi * 332.0636817823836 #1e-3*Na*1e10*(qe**2/(ep_vacc*4*numpy.pi*cal2J))
        kcal_to_kJ = 4.184

        self.f_solv1 = np.zeros([3])
        self.f_qf1, self.f_qf1_charges = fixed_charge_forces(self.dphi1dn,self.phi1,neumann_space1,dirichl_space1,x_q1,q1)
        self.f_db1,self.f_ib1 = boundary_forces(self.dphi1dn,self.phi1,self.grid1,kappa,ep_ex,ep_in)
        self.f_solv1 = self.f_qf1+self.f_db1+self.f_ib1
        self.solv_energy1 = 0.5 * kcal_to_kJ * convert_to_kcalmolA * np.sum(q1*phi_q1).real

        self.f_solv2 = np.zeros([3])
        self.f_qf2, self.f_qf2_charges = fixed_charge_forces(self.dphi2dn,self.phi2,neumann_space2,dirichl_space2,x_q2,q2)
        self.f_db2,self.f_ib2 = boundary_forces(self.dphi2dn,self.phi2,self.grid2,kappa,ep_ex,ep_in)
        self.f_solv2 = self.f_qf2+self.f_db2+self.f_ib2
        self.solv_energy2 = 0.5 * kcal_to_kJ * convert_to_kcalmolA * np.sum(q2*phi_q2).real

    def save_info(self, dir, pname1, fname1, gs1, pname2, fname2, gs2, dist):

        results_file = open(dir + '\\results\\results_' + pname1 +fname1+'gs'+ str(gs1)+'_' +'to'+'_'+pname2+fname2+'gs'+str(gs2)+'_'+str(dist)+'.txt', 'w')

        results_file.write('Results for ' +pname1+'_'+fname1 + ' to ' + pname2+ '_'+ fname2 + '\n')
        results_file.write('Distance between molecules: '+ str(dist)+ ' A \n')
        results_file.write('\n')
        results_file.write('Molecule 1: '+pname1+'_'+fname1 +'\n')
        results_file.write('Grid scale molecule 1: '+ str(gs1) + '\n')
        results_file.write('Number of elements 1: '+str(self.grid1.number_of_elements) + '\n')
        results_file.write('Total surface area 1 (A^2): '+str(np.sum(self.grid1.volumes)) + '\n')
        results_file.write('\n')
        results_file.write('Molecule 2: '+pname2+'_'+fname2 + '\n')
        results_file.write('Grid scale molecule 2: '+ str(gs2) + '\n')
        results_file.write('Number of elements 2: '+str(self.grid2.number_of_elements) + '\n')
        results_file.write('Total surface area 2 (A^2): '+str(np.sum(self.grid2.volumes)) + '\n')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write('Solvation energy for molecule 1 (kJ/mol): '+ str(self.solv_energy1)+ '\n')
        results_file.write('Total solvation forces for molecule 1 (kJ/molA): ' + str(self.f_solv1)+ '\n')
        results_file.write('Total fixed charge forces for molecule 1 (kJ/molA): ' + str(self.f_qf1)+ '\n')
        results_file.write('Total dielectric boundary force for molecule 1 (kJ/molA): ' + str(self.f_db1)+ '\n')
        results_file.write('Total ionic boundary force for molecule 1 (kJ/molA): ' + str(self.f_ib1)+ '\n')
        results_file.write('\n')
        results_file.write('\n')
        results_file.write('Solvation energy for molecule 2 (kJ/mol): '+ str(self.solv_energy2)+ '\n')
        results_file.write('Total solvation forces for molecule 2 (kJ/molA): ' + str(self.f_solv2)+ '\n')
        results_file.write('Total fixed charge forces for molecule 2 (kJ/molA): ' + str(self.f_qf2)+ '\n')
        results_file.write('Total dielectric boundary force for molecule 2 (kJ/molA): ' + str(self.f_db2)+ '\n')
        results_file.write('Total ionic boundary force for molecule 2 (kJ/molA): ' + str(self.f_ib2)+ '\n')

        results_file.close()

        return None
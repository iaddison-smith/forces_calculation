from fnmatch import fnmatchcase
import numpy as np
import os
import bempp.api
from tools.electrostatics import *
from tools.solv_forces import *
from tools.mesh import *
from tools.solver import *
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz


class Molecule:

    def __init__(self, folder, pname, ffname, gs, ep_in=4, ep_ex=80, kappa=0.125,external_grid=False, probe_radius=1.4,\
        tol=1e-8):

        self.folder = folder
        self.pname = pname
        self.ffname = ffname
        self.gs = gs
        self.ep_in = ep_in
        self.ep_ex = ep_ex
        self.kappa = kappa
        
        if external_grid == False:
            self.grid, self.q, self.x_q = mesh.pqrtomesh(self.folder, self.pname, self.ffname,\
                density_mesh=self.gs, probe_radius_mesh=probe_radius)
        else:
            self.grid, self.q, self.x_q = mesh.pqrtomesh(self.folder, self.pname, self.ffname, \
                density_mesh=self.gs, probe_radius_mesh=1.4, build_mesh=False)

        self.dirichl_space = bempp.api.function_space(self.grid, "DP", 0)
        self.neumann_space = bempp.api.function_space(self.grid, "DP", 0)

        rhs = solute.calc_rhs(self.dirichl_space,self.neumann_space,self.q,self.x_q,self.ep_in)
        op_discrete = solute.coefficient_matrix(self.dirichl_space,self.neumann_space,\
            self.ep_in,self.ep_ex,self.kappa)

        #Resolucion del sistema de ecuaciones con GMRES
        x, info, it_count = solver.gmres(op_discrete, rhs, tol_solver=tol)
        self.it_count = it_count
        self.tol = tol
        #Guardar potencial electrico y su derivada normal
        self.phi = bempp.api.GridFunction(self.dirichl_space, coefficients=x[:self.dirichl_space.global_dof_count])
        self.dphidn = bempp.api.GridFunction(self.neumann_space, coefficients=x[self.dirichl_space.global_dof_count:])

        #Calcular el potencial electrico en los puntos donde estan las cargas
        slp_q = bempp.api.operators.potential.laplace.single_layer(self.neumann_space, self.x_q.transpose())
        dlp_q = bempp.api.operators.potential.laplace.double_layer(self.dirichl_space, self.x_q.transpose())
        self.phi_q = slp_q*self.dphidn - dlp_q*self.phi

        return None

    def get_solv_energy(self):

        convert_to_kcalmolA = 4 * np.pi * 332.0636817823836 #1e-3*Na*1e10*(qe**2/(ep_vacc*4*numpy.pi*cal2J))
        kcal_to_kJ = 4.184
        self.solv_energy = 0.5 * kcal_to_kJ * convert_to_kcalmolA * np.sum(self.q*self.phi_q).real

        return self.solv_energy
    
    def get_coulomb(self):

        self.phi_c = energy_calc.calculate_phic(self.x_q, self.q, ep_in=self.ep_in)
        self.coul_energy = energy_calc.calculate_Gcoul(self.x_q,self.q, ep_in=self.ep_in)

        return self.phi_c, self.coul_energy

    def get_solv_forces(self):

        self.f_solv = np.zeros([3])
        self.f_qf, _, _ = forces_calc.fixed_charge_forces(self.phi,self.dphidn,\
            self.dirichl_space, self.neumann_space, self.x_q, self.q)
        self.f_db,self.f_ib = forces_calc.boundary_forces(self.phi,self.dphidn,self.grid,self.ep_in,self.ep_ex,self.kappa)
        self.f_solv = self.f_qf+self.f_db+self.f_ib

        return self.f_solv, self.f_qf, self.f_db, self.f_ib

    def get_fixed_forces(self):

        self.f_qf, self.f_qf_charges, self.Efield = forces_calc.fixed_charge_forces(self.phi,self.dphidn,\
            self.dirichl_space, self.neumann_space, self.x_q, self.q)

        return self.f_qf, self.f_qf_charges, self.Efield
    
    def save_info(self, forces=False, energy=False):

        results_file = open('results\\results_' + self.pname + '_' + self.ffname + '_' + str(self.gs) +'.txt', 'w')
        results_file.write('Results for ' +self.pname+'_'+self.ffname +'\n')
        results_file.write('Solute dielectric : ' + str(self.ep_in)+ '\n')
        results_file.write('Solvent dielectric: ' + str(self.ep_ex)+ '\n')
        results_file.write('Grid scale: '+ str(self.gs) + '\n')
        results_file.write('Number of elements: '+str(self.grid.number_of_elements) + '\n')
        results_file.write('Total surface area (A^2): '+str(np.sum(self.grid.volumes)) + '\n')
        results_file.write('Grid per surface (el/A^2): ' + str(self.grid.number_of_elements/(np.sum(self.grid.volumes))) + '\n')
        results_file.write('\n')
        results_file.write('Tolerance for GMRES: ' + str(self.tol) + '\n')
        results_file.write('Iterations for solution: ' +str(self.it_count) + '\n')
        results_file.write('\n')
        if energy == True:
            results_file.write('Solvation energy for complex (kJ/mol): '+ str(self.solv_energy)+ '\n')
        if forces == True:
            results_file.write('Total solvation forces for complex (kJ/molA): {} {} {}'.format(self.f_solv[0],\
                self.f_solv[1],self.f_solv[2])+ '\n')
            results_file.write('Total fixed charge forces for complex (kJ/molA): {} {} {}'.format(self.f_qf[0],\
                self.f_qf[1],self.f_qf[2])+ '\n')
            results_file.write('Total dielectric boundary force for complex (kJ/molA): {} {} {}'.format(self.f_db[0],\
                self.f_db[1],self.f_db[2])+ '\n')
            results_file.write('Total ionic boundary force for complex (kJ/molA): {} {} {}'.format(self.f_ib[0],\
                self.f_ib[1],self.f_ib[2])+ '\n')
            results_file.write('\n')
            results_file.write('\n')
            results_file.write('Electric field (kJ/moleA) in solute charge points \n')
            results_file.write('x_q, q, E \n')
            for j in range(len(self.Efield)):
                results_file.write('{} {} {}  {}  {: 5.3e} {: 5.3e} {: 5.3e} \n'.format(self.x_q[j,0],\
                    self.x_q[j,1],self.x_q[j,2],self.q[j],self.Efield[j][0],self.Efield[j][1],self.Efield[j][2]))
        results_file.write('\n')
        results_file.close()

        return None

class Two_molecules:

    def __init__(self, folder1, folder2, pname1, ffname1, gs1, pname2, ffname2, gs2, distance, ep_in=4,\
        ep_ex=80, kappa=0.125, external_grid=False, probe_radius=1.4, tol=1e-8):


        self.folder1 = folder1
        self.folder2 = folder2
        self.pname1 = pname1
        self.ffname1 = ffname1
        self.gs1 = gs1
        self.pname2 = pname2
        self.ffname2 = ffname2
        self.gs2 = gs2
        self.dist = distance
        self.ep_in = ep_in
        self.ep_ex = ep_ex
        self.kappa = kappa
    
        if external_grid == False:
            mesh.mesh_translate(folder2, pname2, ffname2, gs2, distance)
            mesh.pqr_translate(folder2, pname2, ffname2, distance)

            self.grid1, self.q1, self.x_q1 = mesh.pqrtomesh(self.folder1, self.pname1, self.ffname1,\
                density_mesh=self.gs1, probe_radius_mesh=probe_radius)
            ff = ffname2+'_t'+str(distance[0])
            self.grid2, self.q2, self.x_q2 = mesh.pqrtomesh(self.folder2, self.pname2, ff,\
                density_mesh=self.gs2, probe_radius_mesh=probe_radius)
        else:
            self.grid1, self.q1, self.x_q1 = mesh.pqrtomesh(self.folder1, self.pname1, self.ffname1,\
                density_mesh=self.gs1, probe_radius_mesh=probe_radius,build_mesh=False)
            ff = ffname2+'_t'+str(distance[0])
            self.grid2, self.q2, self.x_q2 = mesh.pqrtomesh(self.folder2, self.pname2, ff,\
                density_mesh=self.gs2, probe_radius_mesh=probe_radius,build_mesh=False)

        self.dirichl_space1 = bempp.api.function_space(self.grid1, "DP", 0)
        self.neumann_space1 = bempp.api.function_space(self.grid1, "DP", 0)
        self.dirichl_space2 = bempp.api.function_space(self.grid2, "DP", 0)
        self.neumann_space2 = bempp.api.function_space(self.grid2, "DP", 0)

        rhs = solute.calc_rhs_two(self.dirichl_space1,self.neumann_space1,self.dirichl_space2,\
            self.neumann_space2,self.q1,self.q2,self.x_q1,self.x_q2,self.ep_in)
        op_discrete = solute.coefficient_matrix_two(self.dirichl_space1,self.neumann_space1,\
            self.dirichl_space2,self.neumann_space2,self.ep_in,self.ep_ex,self.kappa)

        #Resolucion del sistema de ecuaciones con GMRES
        x, info, it_count = solver.gmres(op_discrete, rhs, tol_solver=tol)
        self.it_count = it_count
        self.tol = tol

        #Guardar potencial electrico y su derivada para cada una de las proteinas
        a = self.dirichl_space1.global_dof_count
        a1 = self.neumann_space1.global_dof_count
        b = self.dirichl_space2.global_dof_count

        self.phi1 = bempp.api.GridFunction(self.dirichl_space1, coefficients=x[:a])
        self.dphi1dn = bempp.api.GridFunction(self.neumann_space1, coefficients=x[a:(a+a1)])                         
        self.phi2 = bempp.api.GridFunction(self.dirichl_space2, coefficients=x[(a+a1):(a+a1+b)])
        self.dphi2dn = bempp.api.GridFunction(self.neumann_space2, coefficients=x[(a+a1+b):])

        #Calcular el potencial electrico en los puntos donde estan las cargas
        slp_q1 = bempp.api.operators.potential.laplace.single_layer(self.neumann_space1, self.x_q1.transpose())
        dlp_q1 = bempp.api.operators.potential.laplace.double_layer(self.dirichl_space1, self.x_q1.transpose())
        self.phi_q1 = slp_q1*self.dphi1dn - dlp_q1*self.phi1

        slp_q2 = bempp.api.operators.potential.laplace.single_layer(self.neumann_space2, self.x_q2.transpose())
        dlp_q2 = bempp.api.operators.potential.laplace.double_layer(self.dirichl_space2, self.x_q2.transpose())
        self.phi_q2 = slp_q2*self.dphi2dn - dlp_q2*self.phi2

        return None

    def get_solv_energy(self):

        convert_to_kcalmolA = 4 * np.pi * 332.0636817823836 #1e-3*Na*1e10*(qe**2/(ep_vacc*4*numpy.pi*cal2J))
        kcal_to_kJ = 4.184
        self.solv_energy1 = 0.5 * kcal_to_kJ * convert_to_kcalmolA * np.sum(self.q1*self.phi_q1).real
        self.solv_energy2 = 0.5 * kcal_to_kJ * convert_to_kcalmolA * np.sum(self.q2*self.phi_q2).real

        return self.solv_energy1, self.solv_energy2
    
    def get_coulomb(self):

        self.phi_c1 = energy_calc.calculate_phic(self.x_q1, self.q1, ep_in=self.ep_in)
        self.coul_energy1 = energy_calc.calculate_Gcoul(self.x_q1,self.q1, ep_in=self.ep_in)

        self.phi_c2 = energy_calc.calculate_phic(self.x_q2, self.q2, ep_in=self.ep_in)
        self.coul_energy2 = energy_calc.calculate_Gcoul(self.x_q2,self.q2, ep_in=self.ep_in)

        return self.phi_c1, self.phi_c2, self.coul_energy1, self.coul_energy2

    def get_solv_forces(self):

        self.f_solv1 = np.zeros([3])
        self.f_qf1, _, _ = forces_calc.fixed_charge_forces(self.phi1,self.dphi1dn,\
            self.dirichl_space1, self.neumann_space1, self.x_q1, self.q1)
        self.f_db1,self.f_ib1 = forces_calc.boundary_forces(self.phi1,self.dphi1dn,\
            self.grid1,self.ep_in,self.ep_ex,self.kappa)
        self.f_solv1 = self.f_qf1+self.f_db1+self.f_ib1

        self.f_solv2 = np.zeros([3])
        self.f_qf2, _, _ = forces_calc.fixed_charge_forces(self.phi2,self.dphi2dn,\
            self.dirichl_space2, self.neumann_space2, self.x_q2, self.q2)
        self.f_db2,self.f_ib2 = forces_calc.boundary_forces(self.phi2,self.dphi2dn, \
            self.grid2,self.ep_in,self.ep_ex,self.kappa)
        self.f_solv2 = self.f_qf2+self.f_db2+self.f_ib2

        return self.f_solv1, self.f_qf1, self.f_db1, self.f_ib1, self.f_solv2, self.f_qf2, self.f_db2, self.f_ib2

    def get_fixed_forces(self):

        self.f_qf1, self.f_qf_charges1, self.Efield1 = forces_calc.fixed_charge_forces(self.phi1,self.dphi1dn,\
            self.dirichl_space1, self.neumann_space1, self.x_q1, self.q1)

        self.f_qf2, self.f_qf_charges2, self.Efield2 = forces_calc.fixed_charge_forces(self.phi2,self.dphi2dn,\
            self.dirichl_space2, self.neumann_space2, self.x_q2, self.q2)

        return self.f_qf1, self.f_qf_charges1, self.Efield1, self.f_qf2, self.f_qf_charges2, self.Efield2

    def save_info(self, forces=False, energy=False):

        results_file = open('results\\results_' + self.pname1 +self.ffname1+'gs'+ str(self.gs1)+'_' +'to'+'_'+self.pname2+self.ffname2+'gs'+str(self.gs2)+'_'+'dist'+str(self.dist)+'.txt', 'w')
        results_file.write('Results for ' +self.pname1+'_'+self.ffname1 + ' to ' + self.pname2+ '_'+ self.ffname2 + '\n')
        results_file.write('Distance between molecules (x-axis): '+ str(self.dist[0])+ ' A \n')
        results_file.write('Distance between molecules (y-axis): '+ str(self.dist[1])+ ' A \n')
        results_file.write('Distance between molecules (z-axis): '+ str(self.dist[2])+ ' A \n')
        results_file.write('Solute dielectric : ' + str(self.ep_in)+ '\n')
        results_file.write('Solvent dielectric: ' + str(self.ep_ex)+ '\n')
        results_file.write('Debye length inverse: ' + str(self.kappa) + '\n')
        results_file.write('\n')
        results_file.write('Molecule 1: '+self.pname1+'_'+self.ffname1 +'\n')
        results_file.write('Grid scale molecule 1: '+ str(self.gs1) + '\n')
        results_file.write('Number of elements 1: '+str(self.grid1.number_of_elements) + '\n')
        results_file.write('Total surface area 1 (A^2): '+str(np.sum(self.grid1.volumes)) + '\n')
        results_file.write('Grid per surface (el/A^2): ' + str(self.grid1.number_of_elements/(np.sum(self.grid1.volumes))) + '\n')
        results_file.write('\n')
        results_file.write('Molecule 2: '+self.pname2+'_'+self.ffname2 + '\n')
        results_file.write('Grid scale molecule 2: '+ str(self.gs2) + '\n')
        results_file.write('Number of elements 2: '+str(self.grid2.number_of_elements) + '\n')
        results_file.write('Total surface area 2 (A^2): '+str(np.sum(self.grid2.volumes)) + '\n')
        results_file.write('Grid per surface (el/A^2): ' + str(self.grid2.number_of_elements/(np.sum(self.grid2.volumes))) + '\n')
        results_file.write('\n')
        results_file.write('Tolerance for GMRES: ' + str(self.tol) + '\n')
        results_file.write('Iterations for solution: ' +str(self.it_count) + '\n')
        results_file.write('\n')
        if energy == True:
            results_file.write('Solvation energy for molecule 1 (kJ/mol): '+ str(self.solv_energy1)+ '\n')
        if forces == True:
            results_file.write('Total solvation forces for molecule 1 (kJ/molA): {} {} {}'.format(self.f_solv1[0],self.f_solv1[1],self.f_solv1[2]) +  '\n')
            results_file.write('Total fixed charge forces for molecule 1 (kJ/molA): {} {} {}'.format(self.f_qf1[0],self.f_qf1[1],self.f_qf1[2]) + '\n')
            results_file.write('Total dielectric boundary force for molecule 1 (kJ/molA): {} {} {}'.format(self.f_db1[0],self.f_db1[1],self.f_db1[2]) + '\n')
            results_file.write('Total ionic boundary force for molecule 1 (kJ/molA): {} {} {}'.format(self.f_ib1[0],self.f_ib1[1],self.f_ib1[2]) + '\n')
            results_file.write('\n')
            results_file.write('\n')
        if energy == True:
            results_file.write('Solvation energy for molecule 2 (kJ/mol): '+ str(self.solv_energy2)+ '\n')
        if forces==True:
            results_file.write('Total solvation forces for molecule 2 (kJ/molA): {} {} {}'.format(self.f_solv2[0],self.f_solv2[1],self.f_solv2[2]) +  '\n')
            results_file.write('Total fixed charge forces for molecule 2 (kJ/molA): {} {} {}'.format(self.f_qf2[0],self.f_qf2[1],self.f_qf2[2]) + '\n')
            results_file.write('Total dielectric boundary force for molecule 2 (kJ/molA): {} {} {}'.format(self.f_db2[0],self.f_db2[1],self.f_db2[2]) + '\n')
            results_file.write('Total ionic boundary force for molecule 2 (kJ/molA): {} {} {}'.format(self.f_ib2[0],self.f_ib2[1],self.f_ib2[2]) + '\n')
            results_file.write('\n')
            results_file.write('\n')
            results_file.write('Electric field in solute charge points for molecule 1 (kJ/molAe) \n')
            results_file.write('x_q, q, E \n')
            for j in range(len(self.Efield1)):
                results_file.write('{} {} {}  {}  {: 5.3e} {: 5.3e} {: 5.3e} \n'.format(self.x_q1[j,0],self.x_q1[j,1],self.x_q1[j,2],self.q1[j],self.Efield1[j][0],self.Efield1[j][1],self.Efield1[j][2]))
            results_file.write('\n')
            results_file.write('\n')
            results_file.write('Electric field in solute charge points for molecule 2 (kJ/molAe) \n')
            results_file.write('x_q, q, E \n')
            for j in range(len(self.Efield2)):
                results_file.write('{} {} {}  {}  {: 5.3e} {: 5.3e} {: 5.3e} \n'.format(self.x_q2[j,0],self.x_q2[j,1],self.x_q2[j,2],self.q2[j],self.Efield2[j][0],self.Efield2[j][1],self.Efield2[j][2]))
            results_file.write('\n')
            results_file.close()

        return None 
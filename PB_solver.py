import numpy as np
import os
import bempp.api
from mesh_converter import *
from force_calculation import *
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import inspect
from scipy.sparse.linalg import gmres


class Molecule:
    """AAAAAAAAAAAA"""

    def __init__(self, protein_name,forcefield_name,density_grid,external_grid):
        
        ep_in = 4.
        ep_ex = 80.
        kappa = 0.125
        if external_grid is not None:
            directory = 'C:\\Users\\ian\Desktop\\forces_calculation'
            dir_prot = directory+'\\pqr_files\\'+protein_name
            pf = protein_name +'_' + forcefield_name
            pfd = protein_name +'_' + forcefield_name + '_' +'d'+str(density_grid)[::2]
            self.grid = import_msms_mesh('{}/{}.face'.format(dir_prot,pfd),'{}/{}.vert'.format(dir_prot,pfd))
            q, x_q = np.array([]), np.empty((0,3))
            molecule_file = open('{}/{}.pqr'.format(dir_prot,pf), 'r').read().split('\n')
            for line in molecule_file:
                line = line.split()
                if len(line)==0 or line[0]!='ATOM': continue
                q = np.append( q, float(line[8]))
                x_q = np.vstack(( x_q, np.array(line[5:8]).astype(float) ))

        else:
            self.grid, q, x_q = pqrtomesh(directory='C:\\Users\\ian\Desktop\\forces_calculation',protein=protein_name,forcefield=forcefield_name,density=density_grid,probe_radius=1.4)

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

        x, info = gmres(op_discrete, rhs, tol=1e-3, maxiter=1000, restart = 2000)


        phi = bempp.api.GridFunction(dirichl_space, coefficients=x[:dirichl_space.global_dof_count])
        dphidn = bempp.api.GridFunction(neumann_space, coefficients=x[dirichl_space.global_dof_count:])
        h=0.001

        slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_q.transpose())
        dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_q.transpose())
        phi_q = slp_q*dphidn - dlp_q*phi

        self.F_qf, _ = fixedcharge_forces(x_q,q,h,neumann_space,dirichl_space,dphidn,phi)
        self.F_db,self.F_ib = boundary_forces(dphidn,phi,self.grid,kappa,ep_ex,ep_in)
        self.solv_energy = 4.184*2*np.pi*332.064*np.sum(q*phi_q).real
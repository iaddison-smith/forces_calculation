import numpy as np
import os
import griddata

def apbs_simulation(protein,pqr_path,dime_len=100,grid_len=30,mol_surface='spl4',calc_force='yes',write_dx='no'):

    '''
    Runs a simulation with APBS installed using a Apbs_template file:

    protein : name of the protein
    pqr_path : path of the pqr file
    dime_len = length of the box
    grid_len = number of grid points in dime_len
    mol_surface = spl4 or mol
    calc_force = calculate forces with APBS
    write_dx = write potential across the domain in dx format 
    '''

    if calc_force == 'yes' and mol_surface == 'mol':
        print('mol surfaces dont calculate forces in APBS!!!')

    apbs_template_file = open('APBS_sim\\Apbs_template.in', 'r')
    apbs_file = open('APBS_sim\\Apbs_simulation.in','w')
    for line in apbs_template_file:
        if 'mol pqr' in line:
            line = 'mol pqr '+ pqr_path + ' \n'
        elif 'elec name' in line:
            status = line[10:13]
        elif 'dime' in line:
            line = 'dime {} {} {} \n'.format(dime_len,dime_len,dime_len)
        elif 'glen' in line:
            line = 'glen {} {} {} \n'.format(grid_len,grid_len,grid_len)
        elif 'srfm' in line:
            line = 'srfm '+ mol_surface + ' \n'
        elif 'calcforce' in line and calc_force=='yes':
            line = 'calcforce total \n'
        elif 'calcforce' in line and calc_force=='no':
            line = ''
        elif 'write pot' in line and write_dx=='yes':
            line = 'write pot dx APBS_sim\\{}\\pot_{}_{}_{}_{}_{} \n'.format(protein,status,protein,str(dime_len),\
                str(grid_len),str(mol_surface))
        elif 'write pot' in line and write_dx=='no':
            line = ''
        apbs_file.write(line)
    apbs_file.close()
    apbs_template_file.close()
    simul_data = 'APBS_sim\\{}\\{}_{}_{}_{}.txt'.format(protein,protein,str(dime_len),str(grid_len),str(mol_surface))
    os.system('apbs APBS_sim\\Apbs_simulation.in > '+simul_data)
    
    return None

def fixedchargeforces_mol(q,x_q,phi,T=298.15):

    N = phi.edges[1].shape[0]
    f_qf = np.zeros([3])
    R_idealgas = 8.314e-3 #kJ/molK Ideal gas constant (kB * Na)
    for charge in range(len(q)):
        for i in range(N):
            if phi.edges[1][i] > x_q[charge][0]:
                x_i = i-1
                break
        for j in range(N):
            if phi.edges[1][j] > x_q[charge][1]:
                y_i = j-1
                break
        for k in range(N):
            if phi.edges[1][k] > x_q[charge][2]:
                z_i = k-1
                break
        
        #phi in dimensionless unit (kBT/Ec)
        # x derivate
        dphidx = (phi.grid[x_i+1][y_i][z_i] -phi.grid[x_i-1][y_i][z_i]) / (2 * (phi.edges[1][x_i]-phi.edges[1][x_i-1]))
        f_qf[0] = f_qf[0] + R_idealgas * T * q[charge] * dphidx

        # y derivate
        dphidy = (phi.grid[x_i][y_i+1][z_i]-phi.grid[x_i][y_i-1][z_i]) / (2 * (phi.edges[1][y_i]-phi.edges[1][y_i-1]) )
        f_qf[1] = f_qf[1] + R_idealgas * T * q[charge] * dphidy
        
        # z derivate
        dphidz = (phi.grid[x_i][y_i][z_i+1]-phi.grid[x_i][y_i][z_i-1]) / (2 * (phi.edges[1][z_i]-phi.edges[1][z_i-1]) )
        f_qf[2] = f_qf[2] + R_idealgas * T * q[charge] * dphidz

    return f_qf

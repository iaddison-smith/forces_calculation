import numpy as np
import os

def apbs_simulation(protein,pqr_path,apbs_dir,dime_len,grid_len,mol_surface,calc_force,write_dx):

    apbs_template_file = open(apbs_dir + '\\Apbs_template.in', 'r')
    apbs_file = open(apbs_dir + '\\Apbs_simulation.in','w')
    for line in apbs_template_file:
        if 'mol pqr' in line:
            line = 'mol pqr '+pqr_path + ' \n'
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
            line = 'write pot dx pot_'+status+'_'+protein+'_'+str(dime_len)+'_'+str(grid_len)+'_'+str(mol_surface)+' \n'
        apbs_file.write(line)
    apbs_file.close()
    apbs_template_file.close()
    simul_data = protein+'_'+str(dime_len)+'_'+str(grid_len)+'_'+str(mol_surface)+'.txt'
    os.chdir(apbs_dir)
    os.system('apbs Apbs_simulation.in > '+simul_data)
    
    return None
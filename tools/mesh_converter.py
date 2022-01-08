import numpy as np
import os
import bempp.api

def convert_pqr2xyzr(mesh_pqr_path, mesh_xyzr_path):

    pqr_file = open(mesh_pqr_path, 'r')
    pqr_data = pqr_file.read().split('\n')
    xyzr_file = open(mesh_xyzr_path, 'w')
    for line in pqr_data:
        line = line.split()
        if len(line) == 0 or line[0] != 'ATOM':
            continue
        xyzr_file.write(line[5]+"\t"+line[6]+"\t"+line[7]+"\t"+line[9]+"\n")
    pqr_file.close()
    xyzr_file.close()

    return None
    
def import_msms_mesh(mesh_face_path, mesh_vert_path):

    face = open(mesh_face_path, 'r').read()
    vert = open(mesh_vert_path, 'r').read()

    faces = np.vstack(np.char.split(face.split('\n')[0:-1]))[:, :3].astype(int) - 1
    verts = np.vstack(np.char.split(vert.split('\n')[0:-1]))[:, :3].astype(float)

    #grid = bempp.api.grid_from_element_data(verts.transpose(), faces.transpose())
    grid = bempp.api.Grid(verts.transpose(), faces.transpose())

    return grid

def generate_nanoshaper_mesh(mesh_xyzr_path, output_dir, output_name_temp, output_name, density, probe_radius, save_mesh_build_files):
    
    nanoshaper_dir = "C:\\APBS-3.0.0\\bin" #NanoShaper dir for windows!
    nanoshaper_temp_dir = os.path.join(output_dir, "nano\\")
    mesh_dir = output_dir

    if not os.path.exists(nanoshaper_temp_dir):
        os.makedirs(nanoshaper_temp_dir)

    # Execute NanoShaper
    config_template_file = open(nanoshaper_dir+'\\config', 'r')
    config_file = open(nanoshaper_temp_dir + 'surfaceConfiguration.prm', 'w')
    for line in config_template_file:
        if 'XYZR_FileName' in line:
            path = os.path.join(mesh_dir, output_name_temp +'.xyzr')
            line = 'XYZR_FileName = ' + path + ' \n'
        elif 'Grid_scale' in line:
            line = 'Grid_scale = {:04.1f} \n'.format(density)
        elif 'Probe_Radius' in line:
            line = 'Probe_Radius = {:03.1f} \n'.format(probe_radius)

        config_file.write(line)

    config_file.close()
    config_template_file.close()

    os.chdir(nanoshaper_temp_dir)
    os.system(nanoshaper_dir+"\\Nanoshaper")
    os.chdir('..')
    
    os.system('move ' + nanoshaper_temp_dir + 'triangulatedSurf.vert  ' + output_name + '.vert')
    os.system('move ' + nanoshaper_temp_dir + 'triangulatedSurf.face  ' + output_name + '.face')
    
    vert_file = open(output_name + '.vert', 'r')
    vert = vert_file.readlines()
    vert_file.close()
    face_file = open(output_name + '.face', 'r')
    face = face_file.readlines()
    face_file.close()
    
    os.remove(output_name + '.vert')
    os.remove(output_name + '.face')  

    vert_file = open(output_name + '.vert', 'w')
    vert_file.write(''.join(vert[3:]))
    vert_file.close()
    face_file = open(output_name + '.face', 'w')
    face_file.write(''.join(face[3:]))
    face_file.close()

    os.chdir(nanoshaper_temp_dir)
    os.chdir('..')
    os.system('powershell rm -r nano')

    return None


def pqrtomesh(directory,protein,forcefield,density,probe_radius,build_mesh='yes'):

    dir_prot = directory + '\\pqr_files\\' + protein
    pf = protein +'_' + forcefield
    if density < 10.0:
        pfd = protein +'_' + forcefield + '_' +'d'+str(density)[::2]
    else:
        pfd = protein +'_' + forcefield + '_' +'d'+str(density)[:2]+'0'
    if build_mesh=='yes':
        convert_pqr2xyzr('{}/{}.pqr'.format(dir_prot,pf),'{}/{}.xyzr'.format(dir_prot,pf))
        generate_nanoshaper_mesh('{}/{}.xyzr'.format(dir_prot,pf),dir_prot,pf,pfd,density,probe_radius,False)
        grid = import_msms_mesh('{}/{}.face'.format(dir_prot,pfd),'{}/{}.vert'.format(dir_prot,pfd))
    else:
        grid = import_msms_mesh('{}/{}.face'.format(dir_prot,pfd),'{}/{}.vert'.format(dir_prot,pfd))
        
    q, x_q = np.array([]), np.empty((0,3))
    molecule_file = open('{}/{}.pqr'.format(dir_prot,pf), 'r').read().split('\n')
    for line in molecule_file:
        line = line.split()
        if len(line)==0 or line[0]!='ATOM': continue
        q = np.append( q, float(line[8]))
        x_q = np.vstack(( x_q, np.array(line[5:8]).astype(float) ))
    
    return grid, q, x_q

def mesh_translate(directory, protein, ff_ref, gs, distance):

    ff = ff_ref+'_t'+str(distance[0])
    if gs < 10.0:
        gs_str = 'd'+str(gs)[::2]
    else:
        gs_str = 'd'+str(gs)[:2]+'0'
    file_ref = protein+'_'+ff_ref+'_'+gs_str
    file = protein + '_' + ff + '_' + gs_str

    dir_prot = directory + '\\pqr_files\\' + protein
    mesh_face_path = '{}\\{}.face'.format(dir_prot,file_ref)
    mesh_face_path_out = '{}\\{}.face'.format(dir_prot,file)
    mesh_vert_path = '{}\\{}.vert'.format(dir_prot,file_ref)
    mesh_vert_path_out = '{}\\{}.vert'.format(dir_prot,file)

    face_file = open(mesh_face_path, 'r')
    face_data = face_file.read().split('\n')
    face_t_file = open(mesh_face_path_out, 'w')
    for line in face_data:
        if len(line) == 0:
            continue
        face_t_file.write(line + ' \n')

    vert_file = open(mesh_vert_path, 'r')
    vert_data = vert_file.read().split('\n')
    vert_t_file = open(mesh_vert_path_out, 'w')
    for line in vert_data:
        line = line.split()
        if len(line) == 0:
            continue
        tx,ty,tz = distance
        x_new = float(line[0])+ float(tx)
        y_new = float(line[1])+ float(ty)
        z_new = float(line[2])+ float(tz)
        text_template = '    {: 1.3f}    {: 1.3f}    {: 1.3f}    {: 1.3f}    {: 1.3f}    {: 1.3f}       {}       {}  {} \n'
        vert_t_file.write(text_template.format(x_new,y_new,z_new,float(line[3]),float(line[4]),float(line[5]),line[6],line[7],line[8]))

    face_file.close() 
    vert_file.close()
    face_t_file.close()
    vert_t_file.close()

    return None

def pqr_translate(directory, protein, ff_ref, distance):

    ff = ff_ref+'_t'+str(distance[0])
    file_ref_pqr = protein + '_' + ff_ref
    file_pqr = protein + '_' + ff
    dir_prot = directory + '\\pqr_files\\' + protein
    mesh_pqr_path = '{}\\{}.pqr'.format(dir_prot,file_ref_pqr)
    mesh_pqr_path_out = '{}\\{}.pqr'.format(dir_prot,file_pqr)

    pqr_file = open(mesh_pqr_path, 'r')
    pqr_data = pqr_file.read().split('\n')
    pqr_t_file = open(mesh_pqr_path_out, 'w')
    for line in pqr_data:
        line = line.split()
        if len(line) == 0 or line[0] != 'ATOM':
            continue
        tx,ty,tz = distance
        x_new = float(line[5])+ float(tx)
        y_new = float(line[6])+ float(ty)
        z_new = float(line[7])+ float(tz)
        text_template = '{} {}  {}   {} {}        {: 1.3f}  {: 1.3f}  {: 1.3f}  {}  {} \n'
        pqr_t_file.write(text_template.format(line[0],line[1],line[2],line[3],line[4],x_new,y_new,z_new,line[8],line[9]))

    pqr_file.close() 
    pqr_t_file.close()

    return None
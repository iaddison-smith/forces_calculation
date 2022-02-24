import numpy as np
import os
import bempp.api


def calculate_phic(x_q, q, ep_in):

    phi_c = np.zeros(len(q)) #Calcular potencial coulomb
    for j in range(len(q)):
        for k in range(len(q)):
            if j==k:
                continue
            else:
                phi_c[k] = q[k]/(4*np.pi*ep_in*np.linalg.norm(x_q[j]-x_q[k]))

    return phi_c

def calculate_Gcoul(x_q,q,ep_in):

    phi_c = calculate_phic(x_q,q,ep_in)
    G_coul = 0
    convert_to_kcalmolA = 4 * np.pi * 332.0636817823836
    kcal_to_kJ = 4.184
    for k in range(len(q)):
        G_coul += q[k]*phi_c[k]

    return kcal_to_kJ*convert_to_kcalmolA*G_coul

from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import numpy as np
import phonopy
from copy import deepcopy
from ase import units
from ase.io import read, write
from ase.calculators.eam import EAM
import os

def get_LD_configs(calc, at_i, config_type, iterations, T, adjust_iterations=10, accept_tol=0.79, reject_tol=0.21, s=0.1):
    at_i.set_calculator(calc)
    acc = 0.0
    al = [None] * iterations
    El = [None] * iterations

    for i in xrange(iterations):
        if i % adjust_iterations == 0:
            print i, acc/adjust_iterations, s
            if acc/adjust_iterations > accept_tol:
                s = s * 1.1
            if acc/adjust_iterations < reject_tol:
                s = s * 0.9
            acc = 0
        if i == 0:
            E_i = at_i.get_potential_energy()
            El[0] = E_i
        else:
            E_i = El[i-1]


        at_j = deepcopy(at_i)
        at_j.set_cell(at_j.cell + np.random.normal(0,s,(3,3)))
        at_j.set_positions(at_j.positions + np.random.normal(0,s,(len(at_i),3)))

        E_j = at_j.get_potential_energy(force_consistent=True)
        fo = at_j.get_forces()
        st = at_j.get_stress(voigt=False)

        at_j.arrays["force"] = fo
        at_j.info["virial"] = -1.0 * at_j.get_volume() * st
        at_j.info["energy"] = E_j
        at_j.info["config_type"] = "LD_" + config_type + "_{}K".format(T)

        write("LD_{}_{}K_{}.xyz".format(config_type,T,i), at_j)

        if E_j < E_i or np.random.rand() < np.exp( - (E_j - E_i)/(units.kB * T) ):
            El[i] = E_j
            at_i = at_j
            acc += 1.0
            al[i] = at_i
        else:
            El[i] = E_i
            al[i] = at_i

    return al

def get_phonon_configs(calc, at, disps, scell, config_type):

    cell = PhonopyAtoms(symbols=at.get_chemical_symbols(),
                    cell=at.get_cell(),
                    positions=at.get_positions())

    phonon = Phonopy(cell, np.eye(3)*scell)

    al = []

    for disp in disps:
        phonon.generate_displacements(distance=disp)
        supercells = phonon.get_supercells_with_displacements()

        for (i,scell) in enumerate(supercells):
            at = Atoms(symbols=scell.get_chemical_symbols(),
                      scaled_positions=scell.get_scaled_positions(),
                      cell=scell.get_cell(),
                      pbc=True)

            at.set_calculator(calc)

            energy = at.get_potential_energy(force_consistent=True)
            forces = at.get_forces()
            stress = at.get_stress(voigt=False)

            drift_force = forces.sum(axis=0)
            print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
            # Simple translational invariance
            for force in forces:
                force -= drift_force / forces.shape[0]
            at.arrays["force"] = forces
            at.info["virial"] = -1.0 * at.get_volume() * stress
            at.info["energy"] = energy
            at.info["config_type"] = "PH_" + config_type
            write("PH_{}_{}_scell_{}.xyz".format(config_type, i, disp), at)
            al.append(at)

    return al

def get_MD_configs(calculator, at, scell, config_type, T, dt=1, N=5000, MD_eV=500, decor=50):

    if not os.path.exists("MD"):
        os.mkdir("MD")
        os.chdir("MD")
    else:
        raise NameError('Folder Already Exists')

    big_at = at * (scell, scell, scell)
    write("{}.cell".format(config_type), big_at)

    with open('{}.cell'.format(config_type), 'a') as file:
        file.write('kpoint_mp_grid 1 1 1')

    param_str = """task : MolecularDynamics
opt_strategy : speed
md_delta_t : {} fs
md_num_iter : {}
md_ensemble : nvt
md_temperature : {} K
cut_off_energy : {}
mixing_scheme : Pulay
    """.format(dt, N, T, MD_eV)

    with open('{}.param'.format(config_type), 'w') as file:
        file.write(param_str)

    #os.system("castep.mpi -n procs {}".format(config_type))

    al = read("./Ti.md", ":")

    al_sel = []

    for (i,at) in enumerate(al[::decor]):
        at.set_calculator(calculator)

        en = at.get_potential_energy(force_consistent=True)
        fo = at.get_forces()
        st = at.get_stress(voigt=False)

        at.arrays["force"] = fo
        at.info["virial"] = -1.0 * at.get_volume() * st
        at.info["config_type"] = "MD_" + config_type
        at.info["energy"] = en

        write("MD_{}_{}.xyz".format(config_type,i), at)
        al_sel.append(at)

    return al_sel

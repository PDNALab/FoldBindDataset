#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import meld
import meld.system
from meld.system.scalers import LinearRamp
from meld.remd import ladder, adaptor, leader
import meld.system.montecarlo as mc
from meld.system.meld_system import System
from meld.system import patchers
from meld import comm, vault
from meld import parse
from meld import remd
from meld.system import param_sampling
from openmm import unit as u
from openmm.app import PDBFile, Modeller, ForceField
from meld.system.builders.grappa import GrappaOptions, GrappaSystemBuilder


N_REPLICAS = 30
N_STEPS = 40000
BLOCK_SIZE = 50

def gen_state(s, index):
    state = s.get_state_template()
    state.alpha = index / (N_REPLICAS - 1.0)
    return state


def get_dist_restraints(filename, s, scaler, ramp, seq):
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])-1
            name_i = cols[1]
            j = int(cols[2])-1
            name_j = cols[3]
            dist = float(cols[4])
            #
            # note: manually over riding distances
            #
            #dist = 0.45
        
            rest = s.restraints.create_restraint('distance', scaler, ramp,
                                                 r1=0.0*u.nanometer, r2=0.0*u.nanometer, r3=dist*u.nanometer, r4=(dist+0.2)*u.nanometer, 
                                                 k=350*u.kilojoule_per_mole/u.nanometer **2,
                                                 atom1=s.index.atom(i,name_i, expected_resname=seq[i][-3:]),
                                                 atom2=s.index.atom(j,name_j, expected_resname=seq[j][-3:]))
            rest_group.append(rest)
    return dists

def get_dist_restraints_protein(filename, s, scaler, ramp, seq):
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])-1
            name_i = cols[1]
            j = int(cols[2])-1
            name_j = cols[3]
            dist = float(cols[4])
            #
            # note: manually over riding distances
            #
            #dist = 0.45

            rest = s.restraints.create_restraint('distance', scaler, ramp,
                                                 r1=0.0*u.nanometer, r2=(dist-0.1)*u.nanometer, r3=dist*u.nanometer, r4=(dist+0.1)*u.nanometer,
                                                 k=350*u.kilojoule_per_mole/u.nanometer **2,
                                                 atom1=s.index.atom(i,name_i, expected_resname=seq[i][-3:]),
                                                 atom2=s.index.atom(j,name_j, expected_resname=seq[j][-3:]))
            rest_group.append(rest)

    return dists

def setup_system():

    # load the sequence
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    n_res = len(sequence.split())
    
    # build the system
    pdb_file = PDBFile('complex_min_openmm_renum.pdb')


    forcefield = ForceField('amber14/protein.ff14SB.xml', 'implicit/gbn2.xml')
    modeller = Modeller(pdb_file.topology, pdb_file.positions)
    modeller.addHydrogens(forcefield)

    topology = modeller.topology
    positions = modeller.positions

    grappa_options = GrappaOptions(
        solvation_type="implicit",
        grappa_model_tag="grappa-1.4.0", 
        base_forcefield_files=['amber14/protein.ff14SB.xml', 'implicit/gbn2.xml'],
        default_temperature=300.0 * u.kelvin,
        cutoff=None,
        use_big_timestep=False, 
        remove_com = False,
        enable_amap= False,
        amap_beta_bias = 1.0
    )

    grappa_builder = GrappaSystemBuilder(grappa_options)
    system_spec = grappa_builder.build_system(topology, positions)

    s = system_spec.finalize()

    s.temperature_scaler = meld.system.temperature.GeometricTemperatureScaler(0, 0.3, 300.*u.kelvin, 550.*u.kelvin)


    ramp = s.restraints.create_scaler('nonlinear_ramp', start_time=1, end_time=200,
                                      start_weight=1e-3, end_weight=1, factor=4.0)
    seq = sequence.split()
    for i in range(len(seq)):
        if seq[i][-3:] =='HIE': seq[i]='HIS'
    print(seq)



    # Setup Scaler
    scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    prot_scaler = s.restraints.create_scaler('constant')
    
    
    # TO keep the restraints between protein and peptide 
    dists = get_dist_restraints('protein_pep_all.dat', s, scaler, ramp, seq)
    s.restraints.add_selectively_active_collection(dists, int(len(dists)*0.20)) #>> 5 restraints
    
    # To keep the protein folded
    prot_rest = get_dist_restraints_protein('protein_contacts.dat',s,prot_scaler,ramp,seq)
    s.restraints.add_selectively_active_collection(prot_rest, int(len(prot_rest)*0.90))

    # To prevent the peptide go far away from the protein (~70 Å)
    mdm2_center = (34, 'CA')
    p53_center = (93, 'CA')

    scaler3 = s.restraints.create_scaler('constant')
    conf_rest = []
    atom_mdm2 = s.index.atom(mdm2_center[0], 'CA', expected_resname=seq[mdm2_center[0]][-3:])
    atom_pdiq = s.index.atom(p53_center[0], 'CA', expected_resname=seq[p53_center[0]][-3:])
    conf_rest.append(
        s.restraints.create_restraint(
            'distance',
            scaler3,
            ramp=LinearRamp(0, 100, 0, 1),
            r1=0.0 * u.nanometer,
            r2=0.0 * u.nanometer,
            r3=7.0 * u.nanometer,
            r4=8.0 * u.nanometer,
            k=250.0 * u.kilojoule_per_mole / u.nanometer ** 2,
            atom1=atom_mdm2,
            atom2=atom_pdiq,
        )
    )
    s.restraints.add_as_always_active_list(conf_rest)

    # create the options
    options = meld.RunOptions(
        timesteps = 14286,
        minimize_steps = 20000,
        #min_mc = sched,
        param_mcmc_steps=200
    )

    # create a store
    store = vault.DataStore(s.get_state_template(),N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=48 * 48)
    policy_1 = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy_1, min_acc_prob=0.02)

    remd_runner = remd.leader.LeaderReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS,
                                                            ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS, timeout=60000)
    store.save_communicator(c)

    # create and save the initial states
    states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

    return s.n_atoms


setup_system()

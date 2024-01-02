#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:54:19 2023

@author: davide
"""
import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
import jax.scipy
from jax import vmap
from jax import lax
from jaxopt import LBFGS
from jax import grad
from jax import jit
import numpy as real_np
from pennylane_qchem.qchem.structure import convert_observable
from pennylane.qchem.observable_hf import fermionic_observable, qubit_observable
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.chem import molecular_data
import openfermion.ops.representations as reps
from scipy import optimize
from scipy.linalg import expm


geometry =  1.88973*np.array([[ 0.0000,   0.00000,  0.0000],
                               [ 0.00000, 0.00000,  1.0000],
                               [ 0.00000 , 0.00000 , 2.0000],
                               [ 0.00000, 0.00000, 3.0000],
                               [0.00000, 0.00000, 4.00000],
                               [0.00000, 0.00000, 5.00000],
                               [0.00000, 0.00000, 6.00000],
                               [0.00000, 0.00000, 7.00000]
                               ],
                               requires_grad = False) ### HF optimized

orbital_optimization = False

symbols = ["H", "H", "H","H", "H", "H", "H", "H"]

total_T = 150

initial_params = np.random.rand(len(symbols) + 2) #### params for the driving hamiltonia 

args = []

mol = qml.qchem.Molecule(symbols, geometry)

v_fock, coeffs, fock_matrix, h_core, rep_tensor = qml.qchem.scf(mol)() ### orbital energies, coeffs, fock_matrix, h_core, rep_tensor

f_mo = coeffs.T @ fock_matrix @ coeffs ##### N.B. mo-to-ao requires overlap matrix!

H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)

generators = qml.symmetry_generators(H)
paulixops = qml.paulix_ops(generators, len(H.wires))
paulix_sector = qml.qchem.optimal_sector(H, generators, mol.n_electrons)

tapered_hamiltonian = qml.qchem.taper(H, generators, paulixops, paulix_sector)

hamiltonian = tapered_hamiltonian.sparse_matrix().todense().real

fci_energy = np.linalg.eigh(hamiltonian)[0][0]

hf_state = qml.qchem.hf_state(8, int(2 * np.shape(coeffs)[0]))  ### first number is electrons

tapered_hf_state = qml.qchem.taper_hf(
                    generators,
                    paulixops,
                    paulix_sector,
                    mol.n_electrons,
                    len(tapered_hamiltonian.wires),
                )

print("HF Energy is : " + str(qml.qchem.hf_energy(mol)()))

print("FCI energy is : " + str(fci_energy))

print("Number of qubits is : " + str(qubits))

non_tapered_qubits = len(tapered_hf_state)

qubits = non_tapered_qubits

core_constant, one_electron_integrals, two_electron_integrals = qml.qchem.electron_integrals(mol)()


### prepare HF hamiltonian ###


def one_body2_densequbit(one_body_matrix):
    core_hamiltonian = reps.InteractionOperator(0, one_body_matrix, np.zeros(( non_tapered_qubits, non_tapered_qubits, non_tapered_qubits,  non_tapered_qubits)))
    obs_core_hamiltonian = convert_observable( jordan_wigner(get_fermion_operator(core_hamiltonian)) )
    
    return obs_core_hamiltonian

def two_body2_densequbit(two_body_matrix):
    twobody_hamiltonian = reps.InteractionOperator(0, np.zeros((non_tapered_qubits, non_tapered_qubits)), 0.5 * two_body_matrix)
    obs_two_body_hamiltonian = convert_observable( jordan_wigner(get_fermion_operator(twobody_hamiltonian)) )
    
    return obs_two_body_hamiltonian


so_one_el_int, so_two_el_int = molecular_data.spinorb_from_spatial(one_electron_integrals, two_electron_integrals)

obs_two_body_hamiltonian = two_body2_densequbit(so_two_el_int)

fock_hamiltonian = qml.qchem.taper(one_body2_densequbit(f_mo.numpy()), generators, paulixops, paulix_sector).sparse_matrix().todense().real

el_el_drive = jnp.array(qml.qchem.taper(obs_two_body_hamiltonian, generators, paulixops, paulix_sector).sparse_matrix().todense().real)

nstep = 10

dt = 0.05

dev = qml.device("default.qubit", wires=tapered_hamiltonian.wires)
@qml.qnode(dev, interface="autograd")
def circuit():
    qml.BasisState(np.array(tapered_hf_state), wires=tapered_hamiltonian.wires)
    return qml.state()

hf_state = circuit()


############## HERE WE WRITE ALL THE FUNCTIONS (ENERGY AND GRADIENTS) AS MATRIX MULTIPLICATIONS


def numerical_exponentiation(n,a):     ###### check_exp_diff returns reasonable gradient 
    q = 6
    a2 = a
    a_norm = max(sum(abs(a)))   ### check for differentiable max
    ee = (np.log2( a_norm)) + 1
    print(ee)
    s = np.ceil(max( 0, ee + 1 ))  ###   check for differentiable ceiling
    a2 = a2 / ( 2.0 ** s )
    x = a2
    c = 0.5
    e = np.eye( n, dtype = np.complex64 ) + c * a2
    d = np.eye( n, dtype = np.complex64 ) - c * a2
    p = True
    for k in range ( 2, q + 1 ):
        c = c * float( q - k + 1 ) / float( k * ( 2 * q - k + 1 ) )
        x = np.dot( a2, x )
        e = e + c * x
        if ( p ):
            d = d + c * x
        else:
            d = d - c * x
        p = not p
            #  E -> inverse(D) * E
    e = np.linalg.solve(d, e)
            #  E -> E^(2*S)
    for k in range ( 0, int(s.data.tolist())):   ####
        e = np.dot ( e, e )
    return e


def populate_upper_triangular(vector):
    """
    Populate the upper triangular part of a square matrix with elements from the input vector.

    Args:
    vector (numpy.ndarray or list): Input vector of length n(n-1)/2.

    Returns:
    numpy.ndarray: Square matrix with the upper triangular part filled.
    """
    n = int((1 + np.sqrt(1 + 8 * len(vector)))/2)  # Calculate the size of the square matrix

    # Create a square matrix filled with zeros
    matrix = np.zeros((n, n))

    # Calculate the indices for accessing elements in the upper triangular part
    indices = np.triu_indices(n, k=1)

    # Check if the input vector has the correct length
    if len(vector) != len(indices[0]):
        raise ValueError("Input vector length does not match the expected length.")

    # Populate the upper triangular part of the matrix with elements from the vector
    matrix[indices] = vector

    return matrix

def vector_to_skew_matrix(vector):
    a = populate_upper_triangular(vector)
    
    return (a - a.T)/2

def vector_to_rotation_matrix(vector):
    
    k = vector_to_skew_matrix(vector)
    
    u = expm(k)
    
    return u


def rotate_integrals(coeffs, h_core, repulsion_tensor):
    one = np.einsum("qr,rs,st->qt", coeffs.T, h_core, coeffs, optimize = True)
    two = np.swapaxes(
        np.einsum(
            "ab,cd,bdeg,ef,gh->acfh", coeffs.T, coeffs.T, repulsion_tensor, coeffs, coeffs,
          optimize = True),
        1,
        3,
    )
    
    return one, two

def update_attraction_matrix_drive(params):
    
    scaled_att_matrix = qml.qchem.attraction_matrix(mol.basis_set, np.multiply(params, mol.nuclear_charges), mol.coordinates)()

    one = np.einsum("qr,rs,st->qt", coeffs.T, scaled_att_matrix, coeffs, optimize = True)
    obs_core_hamiltonian = qubit_observable(fermionic_observable(0, one, None))


    taper_obs_core_hamiltonian = qml.qchem.taper(obs_core_hamiltonian, generators, paulixops, paulix_sector).sparse_matrix().todense().real

    
    return taper_obs_core_hamiltonian


    
def update_kinetic_matrix_drive(param):

    scaled_kinetic_matrix = param * qml.qchem.kinetic_matrix(mol.basis_set)()
    one = np.einsum("qr,rs,st->qt", coeffs.T, scaled_kinetic_matrix, coeffs, optimize = True)
    obs_core_hamiltonian = qubit_observable(fermionic_observable(0, one, None))
    taper_obs_core_hamiltonian = qml.qchem.taper(obs_core_hamiltonian, generators, paulixops, paulix_sector).sparse_matrix().todense().real
    
    return taper_obs_core_hamiltonian

def rotate_hamiltonian(rotations):
    
    u = vector_to_rotation_matrix(rotations)
    
    new_coeffs = coeffs @ u
    
    one, two = rotate_integrals(new_coeffs, h_core, rep_tensor)
    
    non_tapered_rotated_hamiltonian = qubit_observable(fermionic_observable(core_constant, one, two))
        
    rotated_hamiltonian = non_tapered_rotated_hamiltonian.sparse_matrix().todense().real
    
    
    return rotated_hamiltonian

def compute_attraction_drivers():
    
    kinetic_matrices = qml.qchem.attraction_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates, atomic_wise = True)()
    
    attraction_drivers = []
    
    for matrix in kinetic_matrices:
        one = np.einsum("qr,rs,st->qt", coeffs.T, matrix, coeffs, optimize = True)
        obs_core_hamiltonian = qubit_observable(fermionic_observable(0, one, None))
        taper_obs_core_hamiltonian = jnp.array(qml.qchem.taper(obs_core_hamiltonian, generators, paulixops, paulix_sector).sparse_matrix().todense().real)
        
        attraction_drivers.append(taper_obs_core_hamiltonian)
    
    return attraction_drivers

def compute_annealing_schedule(nstep):
    
    times = np.arange(nstep)
    
    A_schedule = 1 - times * (dt/total_T)
    
    B_schedule = times * (dt/total_T)
    
    return A_schedule, B_schedule

def elementwise_add(scaling, operators):
    return jnp.sum(scaling * operators)

#def get_single_exp_step(j, params, rotated_hamiltonian):
#    params_index = j * (len(initial_params) + 2)
#    # Extract parameters for the current step
#    kinetic = params[params_index] * kinetic_drive
#    params_index += 1
#    attraction = vmap(elementwise_add)(params[params_index:params_index+len(attraction_drivers)], attractions_drivers)
#    params_index += len(attraction_drivers)
#
#    h_drive = kinetic + attraction
#    coulomb_drive = (1 - params[params_index]) * el_el_drive
#    params_index += 1
#    scaled_ham = params[params_index] * rotated_hamiltonian
#    params_index += 1
#    scaled_fock = params[params_index] * fock_hamiltonian
#    params_index += 1
#    h_t = h_drive + coulomb_drive + scaled_ham + scaled_fock
#    u_t = jax.scipy.linalg.expm(-1j * h_t * dt)
#    return u_t

def multiply(a, b):
    return a * b

### PROBABLY THE VERSION HERE BELOW IS THE FASTEST BUT REQUIRES HIGHER MEMORY CONSUMPTION ###

#def compute_u_step(j, params, rotated_hamiltonian):
#    params_index = j * (len(initial_params) + 2)
#    kinetic = params[params_index] * kinetic_drive
#    params_index += 1
#    attraction = sum(vmap(multiply)(lax.dynamic_slice(params, (params_index,), (len(attraction_drivers),)), jnp.asarray(attraction_drivers)))
#    params_index += len(attraction_drivers)
#    h_drive = kinetic + attraction
#    coulomb_drive = (1 - params[params_index]) * el_el_drive
#    params_index += 1
#    scaled_ham = params[params_index] * rotated_hamiltonian
#    params_index += 1
#    scaled_fock = params[params_index] * fock_hamiltonian
#    params_index += 1
#    h_t = h_drive + coulomb_drive + scaled_ham + scaled_fock
#    return jax.scipy.linalg.expm(-1j * h_t * dt)



#def energy_control_cost(params, rotated_hamiltonian):
#    steps = jnp.arange(nstep)
#    # Vectorize the function over the number of steps
#    vmap_compute_u_step = vmap(compute_u_step, (0, None, None), 0)
#    # Compute u_t for all steps in parallel
#    u_t = jnp.linalg.multi_dot(vmap_compute_u_step(steps, params, rotated_hamiltonian))
#
#    psi_t = jnp.dot(u_t, hf_state)
#    energy = jnp.dot(jnp.dot(jnp.conj(jnp.transpose(psi_t)), rotated_hamiltonian), psi_t)
#        
#    return energy.real
    

#### PRECOMPUTE KINETIC MATRIX TERM ####

kinetic_drive = jnp.array(update_kinetic_matrix_drive(1))

attraction_drivers = compute_attraction_drivers()


#### COST FUNCTIONS ######

def energy_control_cost(params, rotated_hamiltonian):
    
    def body(carry, j):
        
        params_index = j * (len(initial_params) + 2)
        # Extract parameters for the current step
        kinetic = params[params_index] * kinetic_drive
        params_index += 1

        attraction = sum(vmap(multiply)(lax.dynamic_slice(params, (params_index,), (len(attraction_drivers),)), jnp.asarray(attraction_drivers)))
        params_index += len(attraction_drivers)

        h_drive = kinetic + attraction

        coulomb_drive = (1 - params[params_index]) * el_el_drive
        params_index += 1

        scaled_ham = params[params_index] * rotated_hamiltonian
        params_index += 1

        scaled_fock = params[params_index] * fock_hamiltonian
        params_index += 1

        h_t = h_drive + coulomb_drive + scaled_ham + scaled_fock

        u_t = jax.scipy.linalg.expm(-1j * h_t * dt)

        carry = jnp.dot(carry, u_t)

        return carry, params_index

    # Initialize total_evolution matrix
    total_evolution = jnp.eye(2**qubits, dtype = jnp.complex64)

    # Create a sequence of step indices
    step_indices = jnp.arange(nstep)

#    # Use lax.scan to perform the loop
    total_evolution, _ = lax.scan(body, total_evolution, step_indices)

    psi_t = jnp.dot(total_evolution, hf_state)
    energy = jnp.dot(jnp.dot(jnp.conj(jnp.transpose(psi_t)), rotated_hamiltonian), psi_t)

    return energy.real

def get_opt_wf(params, rotated_hamiltonian):

    total_evolution = np.eye(2**qubits)

    for j in range(nstep):

        kinetic = params[len(initial_params) * j] * kinetic_drive
        
        attraction = jnp.zeros((2**qubits, 2**qubits))
        for k in range(len(attraction_drivers)):
            attraction = attraction + params[len(initial_params) * j + 1 : len(initial_params) * j + len(initial_params) - 1][k] * attraction_drivers[k]

        h_drive = kinetic_drive + jnp.sum(attraction_drivers) - kinetic - attraction

        coulomb_drive =  (1 - params[len(initial_params) * j + len(initial_params) - 1] ) * el_el_drive


        h_exp = jax.scipy.linalg.expm(-1j * h_drive * dt)
        coulomb_exp = jax.scipy.linalg.expm(-1j * coulomb_drive * dt)
        
        new_step = jnp.dot(h_exp, coulomb_exp) 
        total_evolution = jnp.dot(total_evolution, new_step)

    print(total_evolution, flush = True)
    
    psi_t = jnp.dot(total_evolution, hf_state)    

    return psi_t

def compute_variance(hamiltonian, state):
    # Calculate the expectation value of the Hamiltonian
    expectation_H = jnp.real(jnp.vdot(state, jnp.matmul(hamiltonian, state)))

    # Calculate the expectation value of the Hamiltonian squared
    expectation_H_squared = jnp.real(jnp.vdot(state, jnp.matmul(hamiltonian, jnp.matmul(hamiltonian, state))))
    # Calculate the variance
    variance = expectation_H_squared - expectation_H**2
    return variance

def energy_orb_cost(rotations, psi_t):
    
    
    rotated_hamiltonian = rotate_hamiltonian(rotations)

    energy = jnp.dot(jnp.dot(jnp.conj(jnp.transpose(psi_t)), rotated_hamiltonian), psi_t)

    
    return energy.real


def grad_energy_cost(params, rotated_hamiltonian):
    return grad(energy_control_cost, argnums = (0))(params, rotated_hamiltonian)


def compute_qsl(params):

    variance_at_t = []
    hamiltonian_norm = []
    total_evolution = np.eye(2**qubits)
    
    length_params = len(initial_params) + 2
    
    #h_squared = jnp.dot(hamiltonian, hamiltonian)
        
    for j in range(nstep):
        kinetic = params[j * length_params] * kinetic_drive

        attraction = np.zeros((2**qubits, 2**qubits))
        for k in range(len(attraction_drivers)):
            attraction = attraction + params[j * length_params + k + 1] * attraction_drivers[k]
                
        h_drive = kinetic + attraction

        coulomb_drive =  (1 - params[j * length_params + k + 2]) * el_el_drive
        
        scaled_ham = params[j * length_params + k + 3] * hamiltonian
        
        scaled_fock = params[j * length_params + k + 4] * fock_hamiltonian
 
        h_t = h_drive + coulomb_drive + scaled_ham + scaled_fock
        u_t = jax.scipy.linalg.expm(-1j * h_t * dt)
         
        total_evolution = jnp.dot(total_evolution, u_t)
        
        psi_t = jnp.dot(total_evolution, hf_state)

        variance_at_t.append(jnp.sqrt(compute_variance(h_t, psi_t)))
        hamiltonian_norm.append(jnp.linalg.norm(h_t)/jnp.linalg.norm(hamiltonian))
        
    mean_variance = (1/len(variance_at_t)) * jnp.sum(jnp.array(variance_at_t))

    return jnp.pi / 2 * mean_variance, max(hamiltonian_norm), np.mean(hamiltonian_norm)

###### THEN WE OPTIMIZE ###########


mo = int(non_tapered_qubits // 2)

#rotated_hamiltonian = rotate_hamiltonian(rotations)

max_iterations = 10
# step_size = 0.1
conv_tol = 1e-08

a_t, b_t = compute_annealing_schedule(nstep)

first_params = np.append(initial_params, jnp.array([a_t[0], b_t[0]])).flatten()
params = first_params
#prev_energy = energy_control_cost(params, hamiltonian)

for k in range(1, nstep):
    to_add = np.random.rand(len(symbols) + 2)
    to_add = np.append(to_add, np.array([a_t[k], b_t[k]])).flatten()
    params = np.append(params, to_add)
    
params = jnp.asarray(params)

fun = jit(energy_control_cost)

opt = LBFGS(fun, implicit_diff = False)

state = opt.init_state(params, hamiltonian)

energy_at_iter = []

params_at_iter = []

maxiter = 500

for k in range(maxiter):

    new_params, new_state = opt.update(params, state, hamiltonian) # do LBFGS STEP

    params = new_params
    state = new_state

    print("Energy at iter " + str(k) + " is " + str(state.value), flush = True)

    energy_at_iter.append(state.value)
    params_at_iter.append(params)
    np.save("energy_at_iter_h8_oc_asp_lax_scan_faster_eq5", energy_at_iter)

    np.save("params_at_iter_h8_oc_asp_lax_scan_faster_eq5", params_at_iter)

    if state.value - fci_energy < 1e-3:
        break



np.save("energy_at_iter_h8_oc_asp_lax_scan_faster_eq5", energy_at_iter)

np.save("params_at_iter_h8_oc_asp_lax_scan_faster_eq5", params_at_iter)

print("Total number of f. eval is : " + str(state.num_fun_eval), flush = True)

print("Total number of j. eval is : " + str(state.num_grad_eval), flush = True)

print("Total number of parameters is : " + str(len(params)), flush = True)

qsl, max_hamiltonian, mean_hamiltonian = compute_qsl(params)

print("Quantum speed limit estimated at : " + str(qsl))

print("Max. driving hamiltonian norm is : " + str(max_hamiltonian))

print("Mean driving hamiltonian norm is : " + str(mean_hamiltonian))


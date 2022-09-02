"""
This script contains all the functions for doing vector quantization.
"""

import numpy as np
from numpy import inf
from sympy import fwht


def naive_quantizer(x: np.ndarray, bits: float):
    """
    :param x: Input vector
    :param bits: Bit-budget per dimension
    :return: Naively scalar quantized vector.
    """
    d = len(x)                                      # Dimension
    tot_bits = np.floor(bits * d)                   # Total number of bits

    # Allocation of tot_bits to each dimension (for naive quantizer)
    num_bits_per_dim = np.floor(tot_bits / d)
    num_bits = num_bits_per_dim * np.ones(d)        # Initial allocation
    rem_bits = tot_bits - d * num_bits_per_dim
    rand_perm = np.random.permutation(d)
    for j in range(int(rem_bits)):
        num_bits[int(rand_perm[j])] += 1

    num_points_per_dim = 2 ** num_bits              # Number of points per dimension
    res_per_dim = 2 / num_points_per_dim            # Length of interval in each dimension with dynamic range = 1

    dyn_range = np.linalg.norm(x, ord=inf)
    x_N = np.copy(x) / dyn_range

    # Quantize each coordinate with a scalar quantizer of unit dynamic range
    q = np.zeros(d)
    for j in range(d):
        first_quant_point = -1 + res_per_dim[j] / 2
        last_quant_point = 1 - res_per_dim[j] / 2

        if num_bits[j] == 0:
            q[j] = 0
        elif x_N[j] <= first_quant_point:
            q[j] = first_quant_point
        elif x_N[j] >= last_quant_point:
            q[j] = last_quant_point
        else:
            lower_quant_idx = np.floor((x_N[j] - first_quant_point) / res_per_dim[j])
            lower_quant_point = first_quant_point + lower_quant_idx * res_per_dim[j]
            upper_quant_point = first_quant_point + (lower_quant_idx + 1) * res_per_dim[j]
            if x_N[j] - lower_quant_point <= res_per_dim[j] / 2:
                q[j] = lower_quant_point
            else:
                q[j] = upper_quant_point

    q *= dyn_range

    return q


def hadamard_quantizer(x: np.ndarray, bits: float):
    """
    :param x: Input vector
    :param bits: Bit-budget per dimension
    :return: Quantized vector
    """
    d = len(x)                                               # Dimension of the input vector
    x = np.reshape(x, (len(x), ), order='C')
    q_vec = np.zeros(d)                                      # Initializing the quantized vector

    # Get dimensions of subvectors for which Hadamard matrices can be constructed
    res = d
    subvec_dims = []
    while res > 0:
        t = int(np.log2(res))
        res -= 2 ** t
        subvec_dims.append(2 ** t)

    # Get indices where the whole vector is partitioned
    part_idx = np.append([0], np.cumsum(subvec_dims))

    # Quantize each subvector
    for i in range(len(subvec_dims)):
        N = subvec_dims[i]                                      # Dimension of the subvector

        # Allocate the total number of bits to each dimension
        tot_bits = np.floor(bits * N)
        num_bits_per_dim = np.floor(tot_bits / N)
        num_bits = num_bits_per_dim * np.ones(N)                # Initial allocation
        rem_bits = tot_bits - N * num_bits_per_dim
        rand_perm = np.random.permutation(N)
        for j in range(int(rem_bits)):
            num_bits[int(rand_perm[j])] += 1                    # Allocate the remaining bits

        num_points_per_dim = 2 ** num_bits                # Number of points per dimension
        res_per_dim = 2 / num_points_per_dim              # Length of interval in each dimension with dynamic range = 1

        inp = np.copy(x[part_idx[i]:part_idx[i+1]])           # Subvector input to quantizer
        assert len(inp) == N, "Something is wrong!"

        # Get the transform coefficients
        signs = 2 * (np.random.randint(2, size=N) - 0.5)
        D_inp = signs * inp
        HD_inp = np.array(fwht(D_inp))
        coeff_inp = (1 / np.sqrt(N)) * HD_inp
        dyn_range = np.linalg.norm(coeff_inp, ord=inf)
        coeff_inp /= dyn_range

        # Quantize each coordinate with a scalar quantizer of unit dynamic range
        q = np.zeros(N)
        for j in range(N):
            first_quant_point = -1 + res_per_dim[j] / 2
            last_quant_point = 1 - res_per_dim[j] / 2

            if num_bits[j] == 0:
                q[j] = 0
            elif coeff_inp[j] <= first_quant_point:
                q[j] = first_quant_point
            elif coeff_inp[j] >= last_quant_point:
                q[j] = last_quant_point
            else:
                lower_quant_idx = np.floor((coeff_inp[j] - first_quant_point) / res_per_dim[j])
                lower_quant_point = first_quant_point + lower_quant_idx * res_per_dim[j]
                upper_quant_point = first_quant_point + (lower_quant_idx + 1) * res_per_dim[j]
                if coeff_inp[j] - lower_quant_point <= res_per_dim[j] / 2:
                    q[j] = lower_quant_point
                else:
                    q[j] = upper_quant_point

        q = q * dyn_range
        H_q = np.array(fwht(q))
        DH_q = (1 / np.sqrt(N)) * np.array(signs) * H_q
        q_vec[part_idx[i]:part_idx[i+1]] = np.copy(DH_q)

    q_vec = np.reshape(q_vec, (len(q_vec),), order='C')

    return q_vec


def rnd_orth_quantizer(x: np.ndarray, bits: float):
    """
    :param x: Input vector
    :param bits: Bit-budget per dimension
    :return: Quantized vector
    """
    rdn = 1
    d = len(x)                                                      # Dimension
    x = np.reshape(x, (len(x), ), order='C')
    tot_bits = np.floor(bits * d)                                   # Total number of bits

    N = int(np.ceil(rdn * d))                                   # Higher dimension
    rand_matrix = np.random.randn(N, N)                         # Construct a random orthonormal frame
    U, Sigma, Vt = np.linalg.svd(rand_matrix)
    rand_orth = U @ Vt
    rand_perm = np.random.permutation(N)
    S = rand_orth[rand_perm[0:d], :]                            # Columns constitute the tight frame

    # Allocation of tot_bits to each dimension (for democratic quantizer)
    num_bits_per_dim = np.floor(tot_bits / N)
    num_bits = num_bits_per_dim * np.ones(N)            # Initial allocation
    rem_bits = tot_bits - N * num_bits_per_dim
    rand_perm = np.random.permutation(N)
    for j in range(int(rem_bits)):
        num_bits[int(rand_perm[j])] += 1

    num_points_per_dim = 2 ** num_bits              # Number of points per dimension
    res_per_dim = 2 / num_points_per_dim            # Length of interval in each dimension with dynamic range = 1

    # Get the transform coefficients
    x_coeff = np.transpose(S) @ x
    dyn_range = np.linalg.norm(x_coeff, ord=inf)
    x_coeff /= dyn_range

    # Quantize each coordinate with a scalar quantizer of unit dynamic range
    q = np.zeros(N)
    for j in range(N):
        first_quant_point = -1 + res_per_dim[j] / 2
        last_quant_point = 1 - res_per_dim[j] / 2

        if num_bits[j] == 0:
            q[j] = 0
        elif x_coeff[j] <= first_quant_point:
            q[j] = first_quant_point
        elif x_coeff[j] >= last_quant_point:
            q[j] = last_quant_point
        else:
            lower_quant_idx = np.floor((x_coeff[j] - first_quant_point) / res_per_dim[j])
            lower_quant_point = first_quant_point + lower_quant_idx * res_per_dim[j]
            upper_quant_point = first_quant_point + (lower_quant_idx + 1) * res_per_dim[j]
            if x_coeff[j] - lower_quant_point <= res_per_dim[j] / 2:
                q[j] = lower_quant_point
            else:
                q[j] = upper_quant_point

    q *= dyn_range
    q_model = S @ q
    q_model = np.reshape(q_model, (len(q_model),), order='C')

    return q_model

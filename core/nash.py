import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

def symmetrize_winrate_matrix(W):
	"""
	Creates a symmetric winrate matrix by averaging White and Black results.

	Args:
		W (np.ndarray): Winrate matrix.

	Returns:
		np.ndarray: Symmetric meta-game matrix.
	"""
	return (W + (1 - W.T)) / 2


def logit_transform(W, epsilon=1e-6):
	"""
	Converts winrates into log-odds space.

	Args:
		W (np.ndarray): Winrate matrix in range [0,1].
		epsilon (float): Small constant to prevent log(0).

	Returns:
		np.ndarray: Logit-transformed payoff matrix.
	"""
	return np.log((W + epsilon) / (1 - W + epsilon))


def initialize_dual_variables(N):
	"""
	Initializes dual variables for Nash equilibrium computation.

	Args:
		N (int): Number of strategies.

	Returns:
		dict: Dual variable dictionary.
	"""
	dual_variables = {}

	for i in range(N):
		for j in range(N):
			if i != j:
				joint_action = (i, j)  # (current action, alternative action)
				dual_variables[joint_action] = 0

	return dual_variables


def compute_lower_bound_c(A):
	"""
	Computes the lower bound normalization factor c.

	Args:
		A (np.ndarray): Logit payoff matrix.

	Returns:
		float: Lower bound c.
	"""
	c = 0
	N = A.shape[0]

	for i in range(N):
		for i_prime in range(N):
			if i != i_prime:
				for j in range(N):
					c += abs(A[i, j] - A[i_prime, j])

	return c


def compute_Z(A, dv):
	"""
	Computes Z(lambda) normalization term for Nash equilibrium.

	Args:
		A (np.ndarray): Logit payoff matrix.
		dv (dict): Dictionary of dual variables.

	Returns:
		float: Z normalization term.
	"""
	sum_Z = 0
	N = A.shape[0]

	for i in range(N):
		sum_Z += np.exp(-sum(dv[(i, j)] * (A[j, i] - A[i, i]) for j in range(N) if j != i))

	return sum_Z


def compute_P(A, dv, action):
	"""
	Computes the mixed strategy probability for a given action.

	Args:
		A (np.ndarray): Logit payoff matrix.
		dv (dict): Dictionary of dual variables.
		action (int): Action index.

	Returns:
		float: Probability of choosing the action.
	"""
	sum_one = sum(dv[(action, j)] * (A[j, action] - A[action, action]) for j in range(A.shape[0]) if j != action)
	Z = compute_Z(A, dv)

	return np.exp(-sum_one - np.log(Z))


def compute_regret(A, dv, action, alt_action):
	"""
	Computes regret for switching actions.

	Args:
		A (np.ndarray): Logit payoff matrix.
		dv (dict): Dictionary of dual variables.
		action (int): Current action.
		alt_action (int): Alternative action.

	Returns:
		tuple: (positive regret, negative regret)
	"""
	P_action = compute_P(A, dv, action)
	p, n = 0, 0
	N = A.shape[0]

	for j in range(N):
		P_op = compute_P(A, dv, j)
		p_gain = A[alt_action, j] - A[action, j]
		p += (P_action * P_op) * max(0, p_gain)
		n += (P_action * P_op) * max(0, -p_gain)

	return p, n


def compute_nash_equilibrium(A, num_iterations=5000):
	"""
	Computes MaxEnt Nash equilibrium using dynamic gradient ascent.

	Args:
		A (np.ndarray): Logit payoff matrix.
		num_iterations (int): Number of iterations.

	Returns:
		np.ndarray: Nash equilibrium distribution.
	"""
	N = A.shape[0]
	dv = initialize_dual_variables(N)
	c = compute_lower_bound_c(A)

	for _ in tqdm(range(num_iterations), desc="Optimizing Nash Equilibrium"):
		step_dict = {}

		for i in range(N):
			for j in range(N):
				if i != j:
					pos_r, neg_r = compute_regret(A, dv, i, j)
					step_dict[(i, j)] = (1 / c) * ((pos_r / (pos_r + neg_r)) - 0.5)

		for key, step in step_dict.items():
			dv[key] = max(0, dv[key] + step)

	nash_distribution = np.array([compute_P(A, dv, i) for i in range(N)])
	return nash_distribution / nash_distribution.sum()

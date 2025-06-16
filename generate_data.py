# This script generates random LP instances for training and testing

import numpy as np
import scipy.optimize as opt
import random as rd
import os
import argparse
import random

from math import sqrt
from pandas import read_csv
from pathlib import Path
from qpsolvers import solve_qp, Problem
from sklearn.datasets import make_sparse_spd_matrix


## ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--k-train", type=int, default=0, help="number of training data")
parser.add_argument("--k-valid", type=int, default=0, help="number of validation data")
parser.add_argument("--k-test", type=int, default=0, help="number of testing data")
parser.add_argument("--m", type=int, default=10, help="number of constraints")
parser.add_argument("--n", type=int, default=50, help="number of variables")
parser.add_argument("--nnz", type=int, default=100, help="number of nonzero elements in A")
parser.add_argument("--prob-equal", type=float, default=0.3, help="the probability that a constraint is a equality constraint")
parser.add_argument("--mixed-integer", action="store_true", help="include integer variables")
parser.add_argument("--path", type=str, default="data/lcqp", help="path to save generated samples")
parser.add_argument("--feasible-only", action="store_true", help="generate feasible instances only")
parser.add_argument("--q-alpha", type=float, default=0.9, help="ratio of non-zeros in Q matrices")
parser.add_argument("--fix-Q", action="store_true", help="fix Q matrix for all instances generated")
parser.add_argument("--fix-A", action="store_true", help="fix A matrix for all instances generated")
parser.add_argument("--fix-b", action="store_true", help="fix b vector for all instances generated")
parser.add_argument("--fix-c", action="store_true", help="fix c vector for all instances generated")
parser.add_argument("--fix-bounds", action="store_true", help="fix bounds for all instances generated")
parser.add_argument("--fix-circ", action="store_true", help="fix circ vector for all instances generated")
parser.add_argument("--seed", type=int, default=7, help="random seed for reproducibility")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed+1)


def generate_lcqp(m, n, nnz, q_alpha, prob_equal, mixed_integer=False,
                  Q=None, A=None, b=None, c=None, bounds=None, circ=None):
    """This function generates a random instance of linear constrained quadratic
    programming.

        min (1/2) x^T Q x + c^T x
        s.t. Aub x <= bub, Aeq x = beq, lb <= x <= ub

    Arguments:
        - m (int): Number of constraints.
        - n (int): Number of variables.
        - nnz (int): Number of nonzero elements in A.
        - q_alpha (float): Coefficient in range [0,1] that controls sparsity in
            ``Q`` matrix. Larger values enforce more sparsity. Check reference:
            https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_sparse_spd_matrix.html
        - prob_equal (float): Ratio of equality constraints.
        - mixed_integer (bool): If ``True``, a mixed-integer linearly constrained
            quadratic program will be generated.
        - Q (np.ndarray, optional): The Q matrix of the quadratic term in the
            objective function. If not ``None``, the passed value will be used
            for generating the problem instance.
        - A (np.ndarray, optional): The coefficient matrix of the linear constraints.
            If not ``None``, the passed value will be used for generating the
            problem instance.
        - b (np.ndarray, optional): The RHS of the linear constraints.
            If not ``None``, the passed value will be used for generating the
            problem instance.
        - c (np.ndarray, optional): The coefficient of the linear term in the
            objective function. If not ``None``, the passed value will be used
            for generating the problem instance.
        - bounds (np.ndarray, optional): The box bounds of variables.
            If not ``None``, the passed value will be used for generating the
            problem instance.
        - circ (np.ndarray, optional): Indicator of the constraint types.
            1 means equality constraint; 0 means inequality constraints (<=).
            If not ``None``, the passed value will be used for generating the
            problem instance.
    """
    if b is None:
        b = np.random.normal(size=(m,))
    else:
        assert b.size == m

    if c is None:
        c = np.random.normal(size=(n,)) * 0.1
    else:
        assert c.size == n

    if mixed_integer:
        vtype = np.random.randint(0, 2, size=(n,))

    if Q is None:
        Q = make_sparse_spd_matrix(n_dim=n, alpha=q_alpha, norm_diag=True)
    else:
        assert Q.shape[0] == n and Q.shape[1] == n

    def qp_obj(x):
        return np.dot(x, np.dot(Q, x)) / 2.0 + np.dot(c,x)

    if bounds is None:
        bounds = np.random.normal(0, 10., size=(n,2))
    else:
        assert bounds.shape[0] == n and bounds.shape[1] == 2
    bounds.sort(axis=1)
    lb, ub = bounds[:,0], bounds[:,1]

    if A is None:
        A = np.zeros((m, n))
        edge_inds = np.zeros((nnz, 2))
        edge_inds_1d = rd.sample(range(m * n), nnz)
        edge_feats = np.random.normal(size=(nnz,))
        for l in range(nnz):
            i = edge_inds_1d[l] // n
            j = edge_inds_1d[l] % n
            edge_inds[l, 0] = i
            edge_inds[l, 1] = j
            A[i, j] = edge_feats[l]
    else:
        edge_inds = []
        edge_feats = []
        for i in range(m):
            for j in range(n):
                if A[i,j] != 0.0:
                    edge_inds.append((i,j))
                    edge_feats.append(A[i,j])
        edge_inds = np.array(edge_inds, dtype=np.int32)
        edge_feats = np.array(edge_feats, dtype=np.float32)

    # 1 means ``=`` constraint, 0 means ``<=`` constraint
    if circ is None:
        circ = np.random.binomial(1, prob_equal, size=(m,))
    else:
        assert circ.size == m
    A_ub = A[circ == 0, :]
    b_ub = b[circ == 0]
    A_eq = A[circ == 1, :]
    b_eq = b[circ == 1]

    Q_inds = []
    Q_feats = []
    for i in range(n):
        for j in range(n):
            if Q[i,j] != 0.0:
                Q_inds.append((i,j))
                Q_feats.append(Q[i,j])
    Q_inds = np.array(Q_inds, dtype=np.int32)
    Q_feats = np.array(Q_feats, dtype=np.float32)

    cons_feats = np.hstack((b.reshape(m, 1) / n,
                            circ.reshape(m, 1)))
    if mixed_integer:
        var_feats = np.hstack((c.reshape(n, 1),
                               lb.reshape(n, 1),
                               ub.reshape(n, 1),
                               vtype.reshape(n, 1).astype(float) / 10.0))
    else:
        var_feats = np.hstack((c.reshape(n, 1),
                               lb.reshape(n, 1),
                               ub.reshape(n, 1)))

    # solve the problem
    if mixed_integer:
        raise NotImplementedError("You can use whatever sovlers you would like "
                                  "to collect the optimial solution and objective "
                                  "value here.")

    else:
        mp = Problem(Q, c, A_ub, b_ub, A_eq, b_eq, lb, ub)
        solu = solve_qp(*mp.unpack(), solver="highs", time_limit=10)
        obj = qp_obj(solu) if solu is not None else None

    return (Q_inds, Q_feats, solu, obj, cons_feats, edge_feats, edge_inds, var_feats, mp,)


def generate_lcqp_set(path, num_instances, args,
                      Q=None, A=None, b=None, c=None, bounds=None, circ=None):
    count = 0
    while count < num_instances:
        Q_inds, Q_feats, x, obj, cons_feats, edge_feats, edge_inds, var_feats, mp = generate_lcqp(
            args.m, args.n, args.nnz, args.q_alpha, args.prob_equal,
            mixed_integer=args.mixed_integer,
            Q=Q, A=A, b=b, c=c, bounds=bounds, circ=circ
        )

        if args.feasible_only and x is None:
            continue

        cur_path = path / f"instance_{count}"
        os.makedirs(cur_path, exist_ok=True)
        # write to CSV files
        np.savetxt(cur_path/"ConFeatures.csv", cons_feats, delimiter=",", fmt="%10.5f")
        np.savetxt(cur_path/"EdgeFeatures.csv", edge_feats, fmt="%10.5f")
        np.savetxt(cur_path/"EdgeIndices.csv", edge_inds, delimiter=",", fmt="%d")
        np.savetxt(cur_path/"VarFeatures.csv", var_feats, delimiter=",", fmt="%10.5f")
        np.savetxt(cur_path/"QIndices.csv", Q_inds, delimiter=",", fmt="%d")
        np.savetxt(cur_path/"QFeatures.csv", Q_feats, delimiter=",", fmt="%10.5f")
        if x is not None:
            np.savetxt(cur_path/"Labels_obj.csv", [obj], fmt = "%10.5f")
            np.savetxt(cur_path/"Labels_solu.csv", x, fmt = "%10.5f")

        if isinstance(mp, Problem):
            mp.save(cur_path/"ModelQP.npz")
        elif mp is not None:
            raise NotImplementedError

        count += 1


if __name__ == "__main__":
    args = parser.parse_args()

    set_seed(args.seed)

    path = Path(args.path)
    os.makedirs(path, exist_ok=True)

    if args.fix_Q:
        Q = make_sparse_spd_matrix(n_dim=args.n, alpha=args.q_alpha, norm_diag=True)
    else:
        Q = None

    if args.fix_A:
        A = np.zeros((args.m, args.n))
        edge_inds_1d = rd.sample(range(args.m * args.n), args.nnz)
        edge_feats = np.random.normal(size=(args.nnz,))
        for l in range(args.nnz):
            i = edge_inds_1d[l] // args.n
            j = edge_inds_1d[l] % args.n
            A[i, j] = edge_feats[l]
    else:
        A = None

    if args.fix_b:
        b = np.random.uniform(size=(args.m,))
    else:
        b = None

    if args.fix_c:
        c = np.random.normal(size=(args.n,)) * 0.1
    else:
        c = None

    if args.fix_bounds:
        bounds = np.random.normal(0, 10., size=(args.n,2))
    else:
        bounds = None

    if args.fix_circ:
        circ = np.random.binomial(1, args.prob_equal, size=(args.m,))
    else:
        circ = None

    if args.k_valid > 0:
        generate_lcqp_set(path / 'valid', args.k_valid, args,
                          Q=Q, A=A, b=b, c=c, bounds=bounds, circ=circ)
    if args.k_test > 0:
        generate_lcqp_set(path / 'test', args.k_test, args,
                          Q=Q, A=A, b=b, c=c, bounds=bounds, circ=circ)
    if args.k_train > 0:
        generate_lcqp_set(path / 'train', args.k_train, args,
                          Q=Q, A=A, b=b, c=c, bounds=bounds, circ=circ)

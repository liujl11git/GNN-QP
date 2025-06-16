# tensorflow=2.16

import numpy as np
import tensorflow as tf
import tensorflow.keras as K

from pathlib import Path
from qpsolvers import Problem


SOLVER_DICT = {
    "clarabel": {},
    "cvxopt": {},
    "daqp": {},
    "gurobi": {},
    "highs": {},
    "osqp": {"time_limit": 10, "eps_rel": 1e-3},
    "piqp": {},
    "proxqp": {},
}


def load_data_folder(dir_path, return_model=False):
    """Load a dataset.

    Read information of LCQP or QCQP instances stored in `dir_path` as
    sub-directories.
    Note that this function can only process QP instances with of the same
    numbers of variables, constraints and nonzeros in coefficient matrices.

    The structure of `dir_path` should be:
    dir_path
    |
    |--- instance_0
    |    |-- ConFeatures.csv
    |    |-- EdgeFeatures.csv
    |    |-- EdgeIndices.csv
    |    |-- Labels_obj.csv
    |    |-- Labels_solu.csv
    |    |-- QMatrix.csv
    |    |-- QIndices.csv
    |    |-- QFeatures.csv
    |    |-- VarFeatures.csv
    |
    |--- instance_1
    |    |-- ...
    |
    |--- ...

    Returns:
        - var_features: list of TF tensors of shape (num_vars, var_dim)
        - cons_features: list of TF tensors of shape (num_conss, cons_dim)
        - edge_features: list of TF tensors of shape (num_edges, edge_dim)
        - edge_indices: list of TF tensors of shape (2, num_edges)
        - Q_matrices: list of TF tensors of shape (num_vars, num_vars)
        - labels_obj: list of TF tensors of shape (1,)
        - labels_solu: list of TF tensors of shape (num_vars, 1)
        - var_dim: int, the dimension of the variable features
        - cons_dim: int, the dimension of the constraint features
        - edge_dim: int, the dimension of the edge features
    """
    dir_path = Path(dir_path)

    (var_features, cons_features, edge_features, edge_indices,
     Q_matrices, labels_obj, labels_solu) = [],[],[],[],[],[],[]
    models = [] if return_model else None

    for prob in dir_path.glob("*"):
        vf = np.loadtxt(prob/'VarFeatures.csv', delimiter=',', ndmin=2)
        cf = np.loadtxt(prob/'ConFeatures.csv', delimiter=',', ndmin=2)
        ef = np.loadtxt(prob/'EdgeFeatures.csv', delimiter=',', ndmin=2)
        ei = np.loadtxt(prob/'EdgeIndices.csv', delimiter=',', ndmin=2)
        lo = np.loadtxt(prob/'Labels_obj.csv', delimiter=',', ndmin=1)
        ls = np.loadtxt(prob/'Labels_solu.csv', delimiter=',', ndmin=2)
        if (prob / 'QMatrix.csv').exists():
            Qm = np.loadtxt(prob/'QMatrix.csv', delimiter=',', ndmin=2)
            Q_matrices.append(tf.constant(Qm, dtype=tf.float32))
        elif (prob / 'QIndices.csv').exists() and (prob / 'QFeatures.csv').exists():
            Qi = np.loadtxt(prob / 'QIndices.csv', delimiter=',', ndmin=2)
            Qf = np.loadtxt(prob / 'QFeatures.csv', delimiter=',', ndmin=2)
            Q_matrices.append((tf.constant(Qi.transpose(1,0), dtype=tf.int32),
                               tf.constant(Qf, dtype=tf.float32)))

        var_features.append(tf.constant(vf, dtype=tf.float32))
        cons_features.append(tf.constant(cf, dtype=tf.float32))
        edge_features.append(tf.constant(ef, dtype=tf.float32))
        edge_indices.append(tf.constant(ei.transpose(1,0), dtype=tf.int32))
        labels_obj.append(tf.constant(lo, dtype=tf.float32))
        labels_solu.append(tf.constant(ls, dtype=tf.float32))

        if vf.shape[0] != ls.shape[0]:
            import ipdb; ipdb.set_trace(context=10)

        if return_model:
            models.append(Problem.load(prob/'ModelQP.npz'))

    # num_data = len(var_features)
    var_dim = int(var_features[0].shape[1])
    cons_dim = int(cons_features[0].shape[1])
    edge_dim = int(edge_features[0].shape[1])

    ret = (var_features, cons_features, edge_features, edge_indices,
           Q_matrices, labels_obj, labels_solu, var_dim, cons_dim, edge_dim,)
    if return_model:
        ret = ret + (models,)
    return ret


def prep_batch_data(cons_features, edge_indices, edge_features,
                    var_features, Q_matrices, labels):
    batch_size = len(cons_features)
    Q_as_tuple = isinstance(Q_matrices[0], tuple)

    nconss_per_sample = tf.constant([int(c.shape[0]) for c in cons_features], dtype=tf.int32)
    nvars_per_sample = tf.constant([int(v.shape[0]) for v in var_features], dtype=tf.int32)
    nedges_per_sample = tf.constant([int(ei.shape[1]) for ei in edge_indices], dtype=tf.int32)
    if Q_as_tuple:
        nqelems_per_sample = tf.constant([int(Qi.shape[1]) for Qi,_ in Q_matrices], dtype=tf.int32)

    cons_features = tf.concat(cons_features, axis=0)
    var_features = tf.concat(var_features, axis=0)
    edge_features = tf.concat(edge_features, axis=0)
    edge_indices = tf.concat(edge_indices, axis=1)
    label = tf.concat(labels, axis=0)

    cons_shift = tf.math.cumsum(nconss_per_sample, exclusive=True)
    var_shift = tf.math.cumsum(nvars_per_sample, exclusive=True)
    edge_indices = edge_indices + tf.stack([tf.repeat(cons_shift, nedges_per_sample),
                                            tf.repeat(var_shift, nedges_per_sample)])

    # shifted = []
    # for i, e_inds in enumerate(edge_indices):
    #     shift = tf.constant([[cons_shift[i]], [var_shift[i]]], dtype=tf.int32)
    #     shifted.append(e_inds + shift)
    # edge_indices = tf.concat(shifted, axis=1)

    if Q_as_tuple:
        Qi_shift = tf.repeat(var_shift, nqelems_per_sample)
        Q_inds = tf.concat([Qi for Qi,_ in Q_matrices], axis=1)
        Q_inds = Q_inds + tf.stack([Qi_shift, Qi_shift])
        Q_feats = tf.concat([Qf for _, Qf in Q_matrices], axis=0)
        # Q_inds_shifted = [Qi + var_shift[i] for i, (Qi, _) in enumerate(Q_matrices)]
        # Q_inds = tf.concat(Q_inds_shifted, axis=1)
        new_Q = (Q_inds, Q_feats,)
    else:
        raise ValueError("Not supported now")
        # new_nvars = sum(nvars_per_sample)
        # new_Q = tf.zeros(shape=(new_nvars, new_nvars),
        #                  dtype=Q_matrices[0].dtype)
        # for i, Q in enumerate(Q_matrices):
        #     new_Q[var_shift[i]:var_shift[i+1], var_shift[i]:var_shift[i+1]] = Q

    inputs = (cons_features, edge_indices, edge_features, var_features, new_Q,)
    cons_segments = tf.repeat(tf.range(batch_size, dtype=tf.int32), nconss_per_sample)
    var_segments = tf.repeat(tf.range(batch_size, dtype=tf.int32), nvars_per_sample)
    # import ipdb; ipdb.set_trace(context=20)

    return inputs, label, cons_segments, var_segments, nvars_per_sample


if __name__ == "__main__":
    pass

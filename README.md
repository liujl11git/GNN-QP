# Expressive Power of Graph Neural Networks for (Mixed-Integer) Quadratic Programs


This repository is an implementation of the paper accepted at ICML 2025 entitled "Expressive Power of Graph Neural Networks for (Mixed-Integer) Quadratic Programs." The preprint version can be found on [arXiv](https://arxiv.org/abs/2406.05938). We will provide the link to the ICML camera-ready version once it is publicly available. Our codes are modified from [learn2branch](https://github.com/ds4dm/learn2branch) and our previous work [GNN-LP](https://github.com/liujl11git/GNN-LP).

## Introduction

Quadratic programming (QP) is the most widely applied category of problems in nonlinear programming. Many applications require real-time/fast solutions, though not necessarily with high precision. Existing methods either involve matrix decomposition or use the preconditioned conjugate gradient method. For relatively large instances, these methods cannot achieve the real-time requirement unless there is an effective preconditioner. Recently, graph neural networks (GNNs) opened new possibilities for QP. Some promising empirical studies of applying GNNs for QP tasks show that GNNs can capture key characteristics of an optimization instance and provide adaptive guidance accordingly to crucial configurations during the solving process, or directly provide an approximate solution.

However, the theoretical understanding of GNNs in this context remains limited. Specifically, it is unclear what GNNs can and cannot achieve for QP tasks in theory. This work addresses this gap in the context of linearly constrained QP tasks. In the continuous setting, we prove that message-passing GNNs can universally represent fundamental properties of quadratic programs, including feasibility, optimal objective values, and optimal solutions. In the more challenging mixed-integer setting, while GNNs are not universal approximators, we identify a subclass of QP problems that GNNs can reliably represent.

## Dependencies

Our implementation is tested with `python3.10`. It should work for `python >= 3.9`. Our implementation depdends on the following libraries.

```
tensorflow==2.16.1
qpsolvers==4.3.2
tqdm
```

If you use `conda`, you can create an environment using

```
conda env create -f environment.yml
```

## A quick start guide

### LCQP experiments

```bash
bash run.sh --ktrain 500 --embsize 256 --task solution --gpu 0 --seed 42 [--mixed-integer]
```

Explanation of the arguments:
- `--ktrain (int): the size of the training set.`
- `--embsize (int): the hidden embedding size of the GNN.`
- `--task (str): should be chosen between ``solution`` and ``objective``, representing training the GNN to fit the optimal solutions or the objective values.`
- `--gpu (int): the index of the GPU device to use.`
- `--seed (int): random seed, which influences data generation, weight initialization and the stochastic optimization of the GNN.`
- `--mixed-integer (optional): if set to ``True``, mixed-integer LCQP experiment is performed. See the next section for more details.`

### MI-LCQP experiments

To generate the labels (optimal solutions and objective values) for mixed-integer LCQP instances, you can select your favorite solvers and implement the solving process in [`generate_data.py`](https://github.com/xhchrn/GNN-QP/blob/4da4dfb6e93097f33e7cbe90345e0387aa0e127b/generate_data.py#L166).

## Related repositories

On Representing Linear Programs by Graph Neural Networks:

https://github.com/liujl11git/GNN-LP

On Representing Mixed-Integer Linear Programs by Graph Neural Networks

https://github.com/liujl11git/GNN-MILP

Exact Combinatorial Optimization with Graph Convolutional Neural Networks:

https://github.com/ds4dm/learn2branch


## Citing our work

If you find our code helpful in your resarch or work, please cite our paper.

```Latex
@inproceedings{
    chen2025expressive,
    title={Expressive Power of Graph Neural Networks for (Mixed-Integer) Quadratic Programs},
    author={Ziang Chen, Xiaohan Chen, Jialin Liu, Xinshang Wang, Wotao Yin},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=rcMeab1QVn}
}
```


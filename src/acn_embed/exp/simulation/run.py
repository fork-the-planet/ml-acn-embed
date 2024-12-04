#!/usr/bin/env python3
"""
Run the simulation for Fig. 3, and dump all the coordinates into pickle files
"""
import pickle

import numpy as np


# pylint: disable=invalid-name


def _get_Eij(F):
    Eij = np.ones((F.shape[0], F.shape[0]))
    for i in range(F.shape[0]):
        for j in range(F.shape[0]):
            if i >= j:
                continue
            Eij[i, j] = np.exp(-np.sum(np.power(F[i, :] - F[j, :], 2.0)))
            Eij[j, i] = Eij[i, j]
    return Eij


def _get_Qij(Eij):
    Qij = np.zeros((Eij.shape[0], Eij.shape[0]))
    for i in range(Qij.shape[0]):
        for j in range(Qij.shape[0]):
            Qij[i, j] = Eij[i, j] / (np.sum(Eij[i, :]) - 1)
    return Qij


def _get_grad(F, Qij):
    grad = np.zeros((F.shape[0], F.shape[1]))
    ci = F.shape[0] - 1
    for i in range(F.shape[0]):
        grad[i, :] = 2 * np.sum(
            (F[i, :] - F) * np.expand_dims(2 / ci - Qij[i, :] - Qij[:, i], 1), axis=0
        )
    return grad


def get_grad(F):
    Eij = _get_Eij(F)
    Qij = _get_Qij(Eij)
    return _get_grad(F, Qij)


def main():
    np.random.seed(0)
    F1 = np.random.multivariate_normal(
        mean=np.array([0.5, 0.8]), cov=np.array([[0.6, -0.7], [-0.7, 1]]), size=2000
    )
    F2 = np.random.multivariate_normal(
        mean=np.array([-0.3, -0.4]), cov=np.array([[4, 2], [2, 2]]), size=2000
    )
    lrate = 0.1

    for itr in range(40):
        print(itr)
        with open(f"iter{itr:03d}.pkl", "wb") as fobj:
            pickle.dump(F1, fobj)
            pickle.dump(F2, fobj)

        grad1 = get_grad(F1)
        F1 = F1 - lrate * grad1
        grad2 = get_grad(F2)
        F2 = F2 - lrate * grad2


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn


class SinkhornDistance(nn.Module):
    def __init__(self, eps, reg, max_iter):
        super(SinkhornDistance, self).__init__()
        self.eps      = eps
        self.max_iter = max_iter
        self.reg      = reg

    def forward(self, cost_matrix, source, target):
        # size of source: N
        # size of target: K
        # size of CostMatrix: K by N

        M = torch.exp(-cost_matrix / self.reg)
        u = torch.ones_like(source) # size of u: N
        v = torch.ones_like(target) # size of v: K

        err = 1
        ii = 0
        P = torch.diag(u) @ M @ torch.diag(v)
        P_prev = torch.clone(P)
        for ii in range(self.max_iter):

            ii +=1
            u = source/(M@v)
            v = target/(M.T@u)

            P = torch.diag(u) @ M @ torch.diag(v)

            err = torch.linalg.norm(P_prev-P,"fro")
            if err < self.eps:
                break
            P_prev = torch.clone(P)
        min_cost = torch.sum(P * cost_matrix)
        return P, min_cost

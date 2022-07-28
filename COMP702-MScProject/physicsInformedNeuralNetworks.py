# contains classes for creating physics informed neural networks for MSc
# Project based on PyTorch library

import numpy as np
import torch
from time import perf_counter

# define neural network
class PINN_1D_Burgers(torch.nn.Module):
    def __init__(self,
                 hiddenLayers,
                 nodesPerLayer,
                 activationFunction,
                 maxIterations,
                 maxEvaluations,
                 l1,
                 l2,
                 inverseProblem = False,
                 ):
        super(PINN_1D_Burgers, self).__init__()
        # network structure
        self.layers = torch.nn.ModuleList()
        self.activationFunction = activationFunction
        # input layer
        self.layers.append(torch.nn.Linear(2, nodesPerLayer))
        # hidden layers
        for _ in range(hiddenLayers):
            self.layers.append(torch.nn.Linear(nodesPerLayer, nodesPerLayer))
        # output layer
        self.layers.append(torch.nn.Linear(nodesPerLayer, 1))

        self.inverseProblem = inverseProblem
        # check if inverse problem
        if self.inverseProblem:
            # add l1 and l2 as trainable parameters
            self.l1 = torch.nn.Parameter(torch.tensor(l1))
            self.l2 = torch.nn.Parameter(torch.tensor(l2))
            self.lxHistory = np.array([[l1, l2]])
        # otherwise not trainable for forward problem
        else:
            self.l1 = torch.tensor(l1)
            self.l2 = torch.tensor(l2)
        # define L-BFGS optimiser
        self.optimiser = torch.optim.LBFGS(
            self.parameters(),
            max_iter=maxIterations,
            max_eval=maxEvaluations,
            tolerance_change= 1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
            )

    # compute solution via forward pass
    def forward(self, t, x):
        input = torch.stack((t, x)).T
        for layer in self.layers[:-1]:
            input = self.activationFunction(layer(input))
        return self.layers[-1](input)

    # compute PDE residual
    def residual(self, t, x):
        # compute solution
        u = self.forward(t, x).flatten()
        # compute derivatives
        ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
        # compute PDE residual
        return ut + self.l1 * u * ux - self.l2 * uxx

    def lossFunc(self, D, Y, R):
        self.iterationCount += 1
        # compute constituent loss components
        solutionLoss = torch.mean(torch.square(D - Y))
        residualLoss = torch.mean(torch.square(R))
        totalLoss = solutionLoss + residualLoss
        # store losses
        self.history = np.append(self.history,
                                 np.array([
                                           [totalLoss.item()],
                                           [solutionLoss.item()],
                                           [residualLoss.item()]
                                           ]).T,
                                 axis=0)
        # check if inverse problem
        if self.inverseProblem:
            # store l1 and l2 training history
            self.lxHistory = np.append(self.lxHistory,
                                 np.array([
                                           [self.l1.item()],
                                           [self.l2.item()],
                                           ]).T,
                                 axis=0)

        # print training progress update (every (2**2)th epoch initially, then every 500th)
        if (np.log2(self.iterationCount) % 1 == 0 and self.iterationCount < 500) or self.iterationCount % 500 == 0:
            print(f'Epoch: {self.iterationCount} --- Elapsed time: {(perf_counter()-self.startTime):.2f}s --- Loss: {self.history[-1,0]}')
        return totalLoss

    def trainer(self, NuArray, NfArray):
        # set to train mode
        self.train()
        # initialise variables to track progress
        self.history = np.empty((0,3), float)
        self.iterationCount = 0
        self.startTime = perf_counter()
        # Prepare Nu training tensors
        Nu_t = torch.from_numpy(NuArray[:,0].astype(np.float32)).requires_grad_()
        Nu_x = torch.from_numpy(NuArray[:,1].astype(np.float32)).requires_grad_()
        Nu_d = torch.from_numpy(NuArray[:,2].astype(np.float32).reshape(NuArray.shape[0],1))
        # Prepare Nf training tensors
        Nf_t = torch.from_numpy(NfArray[:,0].astype(np.float32)).requires_grad_()
        Nf_x = torch.from_numpy(NfArray[:,1].astype(np.float32)).requires_grad_()
        # define closure function for L-BFGS optimiser
        def closure():
            self.optimiser.zero_grad()
            Nu_u = self.forward(Nu_t, Nu_x)
            Nf_r = self.residual(Nf_t, Nf_x)
            lossValue = self.lossFunc(Nu_d, Nu_u, Nf_r)
            lossValue.backward()
            return lossValue
        # run optimiser
        self.optimiser.step(closure)
        # training complete, set to evaluation mode
        self.eval()

# contains classes for creating physics informed neural networks for MSc
# Project based on PyTorch library

import numpy as np
rng = np.random.default_rng()
import torch
from time import perf_counter

###########################################################################

# define physics informed neural network for 1D Burgers' equation
class PINN_1D_Burgers(torch.nn.Module):
    def __init__(self,
                 hiddenLayers,
                 nodesPerLayer,
                 activationFunction,
                 maxIterations,
                 maxEvaluations,
                 l1_init,
                 l2_init,
                 inverseProblem = False,
                 verbose = True
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
            # add l1 and l2_exp as trainable parameters
            self.l1 = torch.nn.Parameter(torch.tensor(l1_init))
            self.l2_exp = torch.nn.Parameter(torch.tensor(l2_init))
            self.lxHistory = np.array([[l1_init, np.exp(l2_init)]])
        # otherwise not trainable for forward problem
        else:
            self.l1 = torch.tensor(l1_init)
            self.l2 = torch.tensor(l2_init)
        # boolean whether to print update during training
        self.verbose = verbose
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
        # if inverse compute l2 value from natural log of l2
        if self.inverseProblem:
            self.l2 = torch.exp(self.l2_exp)
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
        if self.verbose and ((np.log2(self.iterationCount) % 1 == 0 and self.iterationCount < 500) or self.iterationCount % 500 == 0):
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

###########################################################################

# define physics informed neural network for 2D wave propagation
class PINN_2D_AcousticWave(torch.nn.Module):
    def __init__(self,
                 hiddenLayers,
                 nodesPerLayer,
                 activationFunction,
                 uTrain,
                 uTrainDeltas,
                 maxArray,
                 NuTotal,
                 NfTotal,
                 trainingResample,
                 NuRange,
                 maxIterations,
                 maxEvaluations,
                 l1,
                 l2,
                 device,
                 verbose=True,
                 ):
        super(PINN_2D_AcousticWave, self).__init__()
        # network structure
        self.layers = torch.nn.ModuleList()
        self.activationFunction = activationFunction
        # input layer
        self.layers.append(torch.nn.Linear(3, nodesPerLayer))
        # hidden layers
        for _ in range(hiddenLayers):
            self.layers.append(torch.nn.Linear(nodesPerLayer, nodesPerLayer))
        # output layer
        self.layers.append(torch.nn.Linear(nodesPerLayer, 1))
        # training data comprising early snapshots of wave field
        self.uTrain = uTrain
        # array comprising dt, dx, dz used by finite difference computation to generate solution
        self.uTrainDeltas = uTrainDeltas
        # array defining max extent of domain for PINN to solve for
        self.maxArray = maxArray
        # max of the 3 input dimensions for scaling
        self.inputMax = max(maxArray[0]*1.e3, maxArray[1], maxArray[2])
        # max and min values of solution from training data
        self.uMax = uTrain.max()
        self.uMin = uTrain.min()
        # number of Nu and Nf points to generate and use for training
        self.NuTotal = NuTotal
        self.NfTotal = NfTotal
        # boolean indicating whether to regenerate training data every epoch
        self.trainingResample = trainingResample
        # boolean indicating whether solution points defining initial wave field should be from a range of time steps versus 2 time steps only
        self.NuRange = NuRange
        # residual loss term coefficient
        self.l1 = l1
        # solution loss term coefficient
        self.l2 = l2
        # initialise variables to track training progress
        self.trainTime = 0.
        self.history = np.empty((0,3), float)
        # device to use for model: cpu or gpu
        self.device = device
        # boolean whether to print update during training
        self.verbose = verbose
        # Adam optimiser
        self.AdamOptimiser = torch.optim.Adam(self.parameters())
        # L-BFGS optimiser
        self.LBFGS_optimiser = torch.optim.LBFGS(
            self.parameters(),
            max_iter=maxIterations,
            max_eval=maxEvaluations,
            line_search_fn='strong_wolfe'
            )

    # generate collocation points training tensor
    def createNf(self, tLim):
        # selected uniformly randomly across input domain: t - x - z
        # points scaled between -1 and 1
        Nf = np.array([
            rng.uniform(-1, 1, self.NfTotal),
            rng.uniform(-1, 1, self.NfTotal),
            rng.uniform(-1, 1, self.NfTotal),
        ]).T
        # scale time points according to largest input variable domain
        Nf[:,0] = Nf[:,0] * (self.maxArray[0]*1.e3 / self.inputMax)
        # convert to torch tensor and send to gpu if available
        return torch.from_numpy(Nf.astype(np.float32)).requires_grad_().to(self.device)

    # generate solution points training tensor
    def createNu(self):
        # check if solution points to be taken from range of time steps
        if self.NuRange:
            # create random array of indices
            Nu = np.array([
                rng.choice(self.uTrain.shape[0], self.NuTotal, replace=True),
                rng.choice(self.uTrain.shape[1], self.NuTotal, replace=True),
                rng.choice(self.uTrain.shape[2], self.NuTotal, replace=True),
                np.zeros(self.NuTotal)
                ]).astype(float).T
            # select solution values at each set of input indices
            Nu[:,3] = self.uTrain[
                Nu[:,0].astype(int),
                Nu[:,1].astype(int),
                Nu[:,2].astype(int)]
            # obtain true input values by multiplying indices by deltas
            Nu[:,:3] *= self.uTrainDeltas
        # otherwise solution points to be taken from 2 time snapshots only
        else:
            # take half of the points from each of the 2 time snapshots
            N = int(self.NuTotal*0.5)
            # create random array of indices for x and z only
            Nu1 = np.array([
                        np.zeros(N),
                        rng.choice(self.uTrain.shape[1], N, replace=True),
                        rng.choice(self.uTrain.shape[2], N, replace=True),
                        np.zeros(N),
                     ]).astype(float).T
            # select solution values at each set of input indices, for t=0
            Nu1[:,3] = self.uTrain[0][
                            Nu1[:,1].astype(int),
                            Nu1[:,2].astype(int)]
            # obtain true input values by multiplying indices by deltas
            Nu1[:,:3] *= self.uTrainDeltas
            # create random array of indices for x and z only, time index is the final snapshot
            Nu2 = np.array([
                        np.ones(N) * (self.uTrain.shape[0] - 1),
                        rng.choice(self.uTrain.shape[1], N, replace=True),
                        rng.choice(self.uTrain.shape[2], N, replace=True),
                        np.zeros(N),
                     ]).astype(float).T
            # select solution values at each set of input indices for last time snapshot
            Nu2[:,3] = self.uTrain[-1][
                                Nu2[:,1].astype(int),
                                Nu2[:,2].astype(int)]
            # obtain true input values by multiplying indices by deltas
            Nu2[:,:3] *= self.uTrainDeltas
            # combine 2 snapshot sets
            Nu = np.concatenate([Nu1, Nu2])
        # perform normalisation
        Nu[:,:-1] = 2 * Nu[:,:-1] / self.maxArray[:-1] - 1
        Nu[:,0] = Nu[:,0] * (self.maxArray[0]*1.e3 / self.inputMax)
        Nu[:,-1] = 2 * (Nu[:,-1] - self.uMin) / (self.uMax - self.uMin) - 1
        # convert to torch tensor and send to gpu if available
        # return tensors of inputs and outputs
        return torch.from_numpy(Nu[:,:-1].astype(np.float32)).requires_grad_().to(self.device),\
                    torch.from_numpy(Nu[:,-1].astype(np.float32)).to(self.device)

    # approximate solution via network forward pass
    def forward(self, X):
        input = X
        for layer in self.layers[:-1]:
            input = self.activationFunction(layer(input))
        # flatten to remove unneeded dimension
        return self.layers[-1](input).flatten()

    # compute residual based on governing PDE
    def residual(self, X, m):
        # access each input variable individually to allow appropriate differentiation
        t = X[:,0]
        x = X[:,1]
        z = X[:,2]
        # recombine inputs
        X = torch.stack((t,x,z)).T
        # approximate solution
        u = self.forward(X)
        # compute derivatives
        du_dNf = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dNf[:,0], t, torch.ones_like(u), create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dNf[:,1], x, torch.ones_like(u), create_graph=True)[0]
        d2u_dz2 = torch.autograd.grad(du_dNf[:,2], z, torch.ones_like(u), create_graph=True)[0]
        # compute residual
        return d2u_dt2 - m**2 * (d2u_dx2 + d2u_dz2)

    # get wave speed
    def medium(self, Nf):
        # define wave speed
        m = torch.ones(Nf.shape[0]).to(self.device)
        # normalise
        return m / m.max()

    # computes loss and stores training history
    def lossFunc(self, Nu_pred, Nu_targ, Nf_r):
        # increment count of training iterations
        self.iterationCount += 1
        # compute solution loss
        NuLoss = torch.mean(torch.square(Nu_pred - Nu_targ))
        # compute residual loss
        NfLoss = torch.mean(torch.square(Nf_r))
        # combine constituent loss components
        totalLoss = self.l2*NuLoss + self.l1*NfLoss
        # store iteration loss values
        self.history = np.append(self.history,
                                 np.array([
                                           [totalLoss.item()],
                                           [NuLoss.item()],
                                           [NfLoss.item()],
                                           ]).T,
                                 axis=0)
        # print training progress update (every (2**2)th epoch initially, then every 1000th)
        if self.verbose and ((np.log2(self.iterationCount) % 1 == 0 and self.iterationCount < 1000) or self.iterationCount % 1000 == 0):
            seconds = perf_counter() - self.startTime
            print(f'Epoch: {self.iterationCount} --- Elapsed time: {int(seconds/60)}m{int(seconds%60)}s --- Loss: {self.history[-1,0]}')
        return totalLoss

    # PINN training function using Adam optimiser
    def Adam_trainer(self, epochs, epochsNuOnly):
        # set network to training mode
        self.train()
        # initialise iteration counter and performance timer
        self.iterationCount = 0
        self.startTime = perf_counter()
        # define time limit for which collocation points will be generated
        # this could be used to implement a shifting time horizon during training process
        tLim = self.maxArray[0]
        # generate solution and collocation training sets
        Nf = self.createNf(tLim)
        Nu, Nu_targ = self.createNu()
        # compute wave speed values for collocation points
        m = self.medium(Nf)
        # iterate over training epochs
        for epoch in range(epochs):
            # zero accumulated gradients
            self.AdamOptimiser.zero_grad()
            # approximate solution
            Nu_pred = self.forward(Nu)
            # check if current epoch is greater than number of epochs used for solution point training only (curriculum learning)
            if epoch > epochsNuOnly:
                # compute residual
                Nf_r = self.residual(Nf, m)
            # otherwise create null value for residual loss for loss function functionality
            else:
                Nf_r = torch.tensor([0.]).to(self.device)
            # compute iteration loss value
            lossValue = self.lossFunc(Nu_pred, Nu_targ, Nf_r)
            # backpropagate loss
            lossValue.backward()
            # perform optimisation step
            self.AdamOptimiser.step()
            # check if set to regenerate training points every epoch
            if self.trainingResample:
                # generate solution and collocation training sets
                Nf = self.createNf(tLim)
                Nu, Nu_targ = self.createNu()
        # increment and store training time
        self.trainTime += (perf_counter() - self.startTime)
        # training complete, set network to evaluation mode
        self.eval()

    def LBFGS_trainer(self):
        # set network to training mode
        self.train()
        # initialise iteration counter and performance timer
        self.iterationCount = 0
        self.startTime = perf_counter()
        # define time limit for which collocation points will be generated
        # this could be used to implement a shifting time horizon during training process
        tLim = self.maxArray[0]
        # generate solution and collocation training sets
        Nf = self.createNf(tLim)
        Nu, Nu_targ = self.createNu()
        # compute wave speed values for collocation points
        m = self.medium(Nf)
        # define closure function for L-BFGS optimiser
        def closure():
            # zero accumulated gradients
            self.LBFGS_optimiser.zero_grad()
            # approximate solution
            Nu_pred = self.forward(Nu)
            # compute residual
            Nf_r = self.residual(Nf, m)
            # compute iteration loss value
            lossValue = self.lossFunc(Nu_pred, Nu_targ, Nf_r)
            # backpropagate loss
            lossValue.backward()
            return lossValue
        # define closure function for L-BFGS optimiser, regenerating solution and collocation training points every iteration
        def closureResample():
            # generate solution and collocation training sets
            Nf = self.createNf(tLim)
            Nu, Nu_targ = self.createNu()
            # zero accumulated gradients
            self.LBFGS_optimiser.zero_grad()
            # approximate solution
            Nu_pred = self.forward(Nu)
            # compute residual
            Nf_r = self.residual(Nf, m)
            # compute iteration loss value
            lossValue = self.lossFunc(Nu_pred, Nu_targ, Nf_r)
            # backpropagate loss
            lossValue.backward()
            return lossValue
        # check if training point regeneration being used
        if self.trainingResample:
            # run optimiser
            self.LBFGS_optimiser.step(closureResample)
        else:
            self.LBFGS_optimiser.step(closure)
        # increment and store training time
        self.trainTime += (perf_counter() - self.startTime)
        # training complete, set network to evaluation mode
        self.eval()

    # generate PINN output solution
    def generateOutput(self, FD_solution_nt):
        # generate input variable ranges
        t = np.arange(0, FD_solution_nt) * self.uTrainDeltas[0]
        x = np.arange(0, self.maxArray[1] + self.uTrainDeltas[1], self.uTrainDeltas[1])
        z = np.arange(0, self.maxArray[2] + self.uTrainDeltas[2], self.uTrainDeltas[2])
        # create meshgrids of input variables
        T, X, Z = np.meshgrid(t, x, z, indexing='ij')
        # create array of inputs from meshgrids
        N = np.stack((T.flatten(), X.flatten(), Z.flatten())).T
        # normalise inputs
        N = 2 * N / self.maxArray[:-1] - 1
        N[:,0] = N[:,0] * self.maxArray[0]
        # convert input array to tensor, approximate solution using network and convert output back to numpy array
        U = self.forward(torch.from_numpy(N.astype(np.float32)).to(self.device)).cpu().detach().numpy()
        # unnormalise output back to real values
        U = (U + 1.) * 0.5 * (self.uMax - self.uMin) + self.uMin
        # reshape output array
        return np.reshape(U, (t.shape[0], x.shape[0], z.shape[0]))

import math
import numpy as np
# from scipy.linalg import qr
import torch
from tqdm import tqdm

class GES:
    def __init__(self, T, P, n, k, f, update_X, std, alpha, beta, eta):
        self.T = T
        self.P = P
        self.n = n
        self.k = k
        self.f = f
        self.update_X = update_X
        self.std = std
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def run(self, X):
        def approximate_derivative(f_plus_arr, f_minus_arr, h=1e-8):
            return (f_plus_arr - f_minus_arr / (2 * h))

        self.P = self.k

        # for ele in X:
        f_plus_arr = torch.zeros((self.P, self.n))
        f_minus_arr = torch.zeros((self.P, self.n))
        noise_arr = torch.zeros((self.P, self.n))
        
        for t in tqdm(range(self.T)):
            # Get surrogate gradient
            if t == 0:
                U = torch.rand(self.n, self.k)
            else:
                surrogate_grad = approximate_derivative(f_plus_arr, f_minus_arr)
                # U = orth(surrogate_grad.T)
                try:
                    U, _ = torch.linalg.qr(surrogate_grad.T)
                except:
                    return float('nan')
            # Update k dim U
            
            for i in range(self.P):
                noise_n = torch.rand(self.n)
                noise_k = torch.rand(self.k)

                a = self.std * math.sqrt(self.alpha/self.n) * noise_n
                b = self.std * math.sqrt((1 - self.alpha)/self.k) * U @ noise_k
                noise =  a + b
                noise_arr[i] = noise
                f_plus_arr[i] = self.f(X + noise)
                f_minus_arr[i] = self.f(X - noise)
            g = self.beta / (2*(self.std**2)*self.P) * torch.sum(noise_arr * (f_plus_arr - f_minus_arr), dim=0)
            
            # X -= eta * g
            self.update_X(self.eta * g)
        return X
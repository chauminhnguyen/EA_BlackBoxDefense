from torch.nn import MSELoss, CrossEntropyLoss
import torch
from tqdm import tqdm
import math


class Surrogate:
    def __init__(self, denoiser, model):
        self.denoiser = denoiser
        self.model = model
        self.ce_criterion = CrossEntropyLoss(size_average=None, reduce=False, reduction='none').cuda()
        self.mse_criterion = MSELoss(size_average=None, reduce=None, reduction='none').cuda()

    def surrogate_cls(self, x, targets, batch_size=8):
        """
        Processes data in smaller batches to reduce memory consumption for classification.
        """
        x = x.view(-1, 3, 32, 32)
        total_loss = torch.tensor(0.0).to('cuda')
        num_samples = x.size(0)
        targets = targets.repeat(x.size(0))
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            x_batch = x[start_idx:end_idx]
            targets_batch = targets[start_idx:end_idx]
            
            # Forward pass for the batch
            x_batch = self.denoiser(x_batch)
            cls = self.model(x_batch)
            
            # Compute loss for the batch
            # print(x_batch.shape, cls.shape, targets_batch, targets_batch.repeat(x_batch.size(0)).shape)
            batch_loss = self.ce_criterion(cls, targets_batch)
            # print(batch_loss[0])
            total_loss += batch_loss[0]
        
        return total_loss / num_samples  # Average loss over the entire dataset

    def surrogate_recon(self, x, targets, batch_size=64):
        """
        Processes data in smaller batches to reduce memory consumption for reconstruction.
        """
        x = x.view(-1, 3, 32, 32)
        total_loss = 0.0
        num_samples = x.size(0)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size
            x_batch = x[start_idx:end_idx]
            targets_batch = targets[start_idx:end_idx]
            
            # Forward pass for the batch
            x_batch = self.denoiser(x_batch)
            cls = self.model(x_batch)
            
            # Compute loss for the batch
            batch_loss = self.mse_criterion(cls, targets_batch)
            total_loss += batch_loss
        
        return torch.tensor(total_loss / num_samples)  # Average loss over the entire dataset


# def update_X(model, loss):
#     loss.backward()
#     optimizer.step()


# k = 120
# P = k
# n = 3*64*64
# sur = Surrogate(denoiser, cls_model)
# ges = GES(P, n, k, sur.surrogate_cls, std=0.1, alpha=0.5, beta=2, eta=1e-7)

class GES:
    def __init__(self, P, n, k, f, std, alpha, beta, eta):
        self.P = P
        self.n = n
        self.k = k
        self.f = f
        self.std = std
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.U = None
        self.f_plus_arr = []
        self.f_minus_arr = []
        self.noise_arr = []

    def run(self, X, targets):
        def approximate_derivative(X, h=1e-8):
            sum_arr = []
            for i in range(self.P):
                noise = torch.rand(self.n, self.k).to('cuda')
                f_plus = self.f(X.T + noise, targets)
                f_minus = self.f(X.T - noise, targets)
                sum_arr.append(noise * (f_plus - f_minus))
            sum_arr = torch.vstack(sum_arr)
            # return torch.sum(sum_arr, dim=1)
            return sum_arr

        self.P = self.k

        # for ele in X:
        # f_plus_arr = torch.zeros((self.P, self.n))
        # f_minus_arr = torch.zeros((self.P, self.n))
        # noise_arr = torch.zeros((self.P, self.n))
        
        # for t in tqdm(range(self.T)):
        # Get surrogate gradient
        with torch.no_grad():
            if self.U is None:
                self.U = torch.rand(self.n, self.k)
            else:
                surrogate_grad = approximate_derivative(X)
                # U = orth(surrogate_grad.T)
                try:
                    self.U, _ = torch.linalg.qr(surrogate_grad.T)
                except:
                    return float('nan')
            # Update k dim U
            
            for i in range(self.P):
                noise_n = torch.rand(self.n)
                noise_k = torch.rand(self.k)

                a = self.std * math.sqrt(self.alpha/self.n) * noise_n
                b = self.std * math.sqrt((1 - self.alpha)/self.k) * self.U @ noise_k
                noise =  a + b
                self.noise_arr.append(noise)
                self.f_plus_arr.append(self.f(X + noise.to('cuda'), targets))
                self.f_minus_arr.append(self.f(X - noise.to('cuda'), targets))
            
            self.f_plus_arr = torch.vstack(self.f_plus_arr).to('cuda')
            self.f_minus_arr = torch.vstack(self.f_minus_arr).to('cuda')
            self.noise_arr = torch.vstack(self.noise_arr).to('cuda')

            g = self.beta / (2*(self.std**2)*self.P) * torch.sum(self.noise_arr * (self.f_plus_arr - self.f_minus_arr), dim=0)
            
            # X -= eta * g
            # self.update_X(self.eta * g)
            # return X
            return self.eta * g
            # return torch.sum(self.eta * g, dim=-1).mean()
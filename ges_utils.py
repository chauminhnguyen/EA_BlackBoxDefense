from torch.nn import MSELoss, CrossEntropyLoss
import torch
from tqdm import tqdm
import math
import torch.nn as nn


def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = 1
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))

        # correct_k = torch.flatten(correct[:1], start_dim=0).float().sum(0, keepdim=True)
        # res = correct_k.mul_(100.0 / batch_size)
        
        return correct[0].float()


class Surrogate:
    def __init__(self, denoiser, model, logger=None):
        self.denoiser = denoiser
        self.model = model
        self.ce_criterion = CrossEntropyLoss(size_average=None, reduce=False, reduction='none').cuda()
        self.mse_criterion = MSELoss(size_average=None, reduce=None, reduction='none').cuda()
        # self.acc_criterion = accuracy()
        self.logger = logger

    def surrogate_cls(self, x, targets, batch_size=8):
        """
        Processes data in smaller batches to reduce memory consumption for classification.
        """
        # print(x.shape)
        x = x.view(-1, 3, 32, 32)
        total_loss = torch.tensor(0.0).to('cuda')
        num_samples = x.size(0)
        # targets = targets.repeat(x.size(0))
        # for start_idx in range(0, num_samples, batch_size):
        batch_loss = []
        # for start_idx in range(0, num_samples):
        #     end_idx = start_idx + batch_size
        #     x_batch = x[start_idx:end_idx]
        #     targets_batch = targets[start_idx:end_idx]
            
        # Forward pass for the batch
        x_batch = self.denoiser(x)
        cls = self.model(x_batch)
        
        # Compute loss for the batch
        # print(cls.shape, targets_batch.shape, targets_batch)
        batch_loss.append(self.ce_criterion(cls, targets))
        # batch_loss.append(accuracy(cls, targets))

        batch_loss = torch.vstack(batch_loss)
        total_loss = sum(batch_loss.detach())
        if self.logger is not None:
            self.logger.update(total_loss.mean().item(), num_samples)
        
        # return total_loss / num_samples  # Average loss over the entire dataset
        return batch_loss.detach()

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
        self.f_plus_arr = torch.tensor(self.k, )
        self.f_minus_arr = []
        self.noise_arr = []

    def process_in_batches(self, X, targets, batch_size=1024):
        results = []
        num_chunks = (X.shape[0] + batch_size - 1) // batch_size  # Compute number of chunks
        
        for i in range(num_chunks):
            start = i * batch_size
            end = min((i + 1) * batch_size, X.shape[0])
            
            X_chunk = X[start:end]
            target_chunk = targets[start:end]
            
            with torch.no_grad():  # Prevent extra memory usage
                results.append(self.f(X_chunk, target_chunk).detach())
        
        return torch.cat(results, dim=0)

    def run(self, X, targets):
        def approximate_derivative(X, h=1e-8):
            sum_arr = []
            for i in range(self.P):
                # noise = torch.rand(self.n, self.k).to('cuda')
                noise = torch.rand(X.shape[0], self.n).to('cuda')
                f_plus = self.f(X + noise, targets)
                f_minus = self.f(X - noise, targets)
                # f_plus = self.process_in_batches(X + noise, targets, batch_size=32)
                # f_minus = self.process_in_batches(X - noise, targets, batch_size=32)
                # sum_arr = noise.T * (f_plus - f_minus)
                sum_arr.append(noise.T @ (f_plus - f_minus).T)
            sum_arr = torch.hstack(sum_arr)
            # return torch.sum(sum_arr, dim=1)
            return sum_arr

        self.P = self.k

        with torch.no_grad():
            if self.U is None:
                self.U = torch.rand(self.n, self.k).to('cuda')
            else:
                surrogate_grad = approximate_derivative(X)
                self.U, _ = torch.linalg.qr(surrogate_grad) 

            f_plus_arr = []
            f_minus_arr = []
            noise_arr = []
            for i in range(self.P):
                noise_n = torch.rand(self.n).to('cuda')
                noise_k = torch.rand(self.k).to('cuda')

                a = self.std * math.sqrt(self.alpha/self.n) * noise_n
                b = self.std * math.sqrt((1 - self.alpha)/self.k) * self.U @ noise_k
                noise =  a + b
                noise_arr.append(noise)
                print('=======================')
                print(noise)
                exit()
                f_plus_arr.append(self.f(X + noise, targets))
                f_minus_arr.append(self.f(X - noise, targets))
                # f_plus_arr.append(self.process_in_batches(X + noise, targets, batch_size=32))
                # f_minus_arr.append(self.process_in_batches(X - noise, targets, batch_size=32))
            
            self.f_plus_arr = torch.vstack(f_plus_arr).to('cuda')
            self.f_minus_arr = torch.vstack(f_minus_arr).to('cuda')
            self.noise_arr = torch.vstack(noise_arr).to('cuda')

            g = self.beta / (2*(self.std**2)*self.P) * torch.sum(self.noise_arr.T @ (self.f_plus_arr - self.f_minus_arr), dim=0)

            # print("f_plus min:", self.f_plus_arr.min().item(), "max:", self.f_plus_arr.max().item())
            # print("f_minus min:", self.f_minus_arr.min().item(), "max:", self.f_minus_arr.max().item())
            # print("Gradient estimate min:", g.min().item(), "max:", g.max().item())

            # X -= eta * g
            # self.update_X(self.eta * g)
            # return X
            # print(self.P, g.shape, self.noise_arr.shape, (self.f_plus_arr - self.f_minus_arr).shape)
            return self.eta * g
            # return torch.sum(self.eta * g, dim=-1).mean()
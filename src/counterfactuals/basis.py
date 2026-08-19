# # Defines the Phi matrix (B-Splines, Fourier, RBF, wavelet, Plynomial)
import torch
import torch.nn as nn
import numpy as np

class TemporalBasis(nn.Module):
    """
    Base class for Temporal Basis generators.
    
    Shape:
        Input: Time sequence length (T)
        Output: Basis Matrix \Phi of shape (T, K) where K is num_basis.
    """
    def __init__(self, sequence_length: int, num_basis: int):
        super().__init__()
        self.T = sequence_length
        self.K = num_basis
        # Create a time vector t in [-1, 1] for numerical stability
        self.register_buffer('t', torch.linspace(-1, 1, steps=sequence_length))

    def forward(self) -> torch.Tensor:
        """Returns the basis matrix Phi of shape (T, K)"""
        raise NotImplementedError

class PolynomialBasis(TemporalBasis):
    """
    Generates orthogonal Legendre Polynomials.
    Using standard powers (1, t, t^2) is unstable for optimization; 
    Legendre polynomials are orthogonal and stable in [-1, 1].
    """
    def forward(self) -> torch.Tensor:
        # Phi shape: (T, K)
        phi = torch.zeros(self.T, self.K, device=self.t.device)
        
        # P_0(t) = 1
        if self.K > 0:
            phi[:, 0] = 1.0
        # P_1(t) = t
        if self.K > 1:
            phi[:, 1] = self.t
            
        # Recurrence relation: (n+1)P_{n+1} = (2n+1)t P_n - n P_{n-1}
        for n in range(1, self.K - 1):
            term1 = (2 * n + 1) * self.t * phi[:, n]
            term2 = n * phi[:, n - 1]
            phi[:, n + 1] = (term1 - term2) / (n + 1)
            
        return phi

class FourierBasis(TemporalBasis):
    """
    Generates Sine and Cosine waves.
    Useful for capturing cyclic patterns or frequency-specific noise.
    """
    def forward(self) -> torch.Tensor:
        phi = torch.zeros(self.T, self.K, device=self.t.device)
        
        # Basis 0 is usually the DC component (constant)
        phi[:, 0] = 1.0
        
        # Fill remaining with sin/cos pairs
        # k=1 -> sin(pi*t), k=2 -> cos(pi*t), k=3 -> sin(2*pi*t)...
        for i in range(1, self.K):
            frequency = (i + 1) // 2
            if i % 2 == 1:
                phi[:, i] = torch.sin(np.pi * frequency * self.t)
            else:
                phi[:, i] = torch.cos(np.pi * frequency * self.t)
                
        return phi

class RBFBasis(TemporalBasis):
    """
    Radial Basis Functions (Gaussian Kernels).
    Places K Gaussian bells evenly across the time window.
    Good for "local" edits (e.g., a spike at t=20).
    """
    def __init__(self, sequence_length: int, num_basis: int, bandwidth: float = 0.2):
        super().__init__(sequence_length, num_basis)
        # Centers mu evenly spaced in [-1, 1]
        self.register_buffer('mu', torch.linspace(-1, 1, num_basis))
        self.sigma = bandwidth

    def forward(self) -> torch.Tensor:
        # t: (T, 1), mu: (1, K) -> Broadcasting to (T, K)
        t_col = self.t.unsqueeze(1)
        mu_row = self.mu.unsqueeze(0)
        
        # Gaussian formula: exp( - (t - mu)^2 / (2*sigma^2) )
        phi = torch.exp(- (t_col - mu_row)**2 / (2 * self.sigma**2))
        
        # Normalize so peaks are consistent
        return phi


class BSplineBasis(TemporalBasis):
    def __init__(self, sequence_length, num_basis, degree=3):
        super().__init__(sequence_length, num_basis)
        self.degree = degree
        num_knots = num_basis + degree + 1
        inner_knots = torch.linspace(-1, 1, num_knots - 2 * degree)
        left_padding = torch.full((degree,), -1.0)
        right_padding = torch.full((degree,), 1.0)
        self.register_buffer('knots', torch.cat([left_padding, inner_knots, right_padding]))

    def forward(self) -> torch.Tensor:
        t = self.t.unsqueeze(1)
        kv = self.knots
        basis_0 = []
        for i in range(self.K + self.degree):
            cond1 = t >= kv[i]
            cond2 = t < kv[i+1]
            if i == self.K + self.degree - 1: 
                cond2 = t <= kv[i+1]
            basis_0.append((cond1 & cond2).float())
        
        bases = torch.cat(basis_0, dim=1)
        
        for d in range(1, self.degree + 1):
            new_bases = []
            for i in range(self.K + self.degree - d):
                # Term 1
                denom1 = kv[i+d] - kv[i]
                if denom1 < 1e-6:
                    term1 = torch.zeros_like(t)
                else:
                    term1 = ((t - kv[i]) / denom1) * bases[:, i:i+1]
                
                # Term 2
                denom2 = kv[i+d+1] - kv[i+1]
                if denom2 < 1e-6:
                    term2 = torch.zeros_like(t)
                else:
                    term2 = ((kv[i+d+1] - t) / denom2) * bases[:, i+1:i+2]
                    
                new_bases.append(term1 + term2)
            bases = torch.cat(new_bases, dim=1)
            
        return bases


class WaveletBasis(TemporalBasis):
    """
    Ricker Wavelet (Mexican Hat).
    Captures localized frequency events (e.g., a momentary glitch).
    """
    def __init__(self, sequence_length: int, num_basis: int, width: float = 0.1):
        super().__init__(sequence_length, num_basis)
        # Centers mu evenly spaced
        self.register_buffer('mu', torch.linspace(-1, 1, num_basis))
        self.width = width # "a" parameter (scale)

    def forward(self) -> torch.Tensor:
        t_col = self.t.unsqueeze(1)
        mu_row = self.mu.unsqueeze(0)
        
        # A = 2 / (sqrt(3a) * pi^0.25)
        A = 2 / (np.sqrt(3 * self.width) * (np.pi ** 0.25))
        
        # x = (t - mu) / a
        vec = (t_col - mu_row) / self.width
        
        # Ricker = A * (1 - x^2) * exp(-0.5 * x^2)
        phi = A * (1 - vec**2) * torch.exp(-0.5 * vec**2)
        
        return phi

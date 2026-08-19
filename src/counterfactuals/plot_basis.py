import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Basis Classes (from basis.py) ---

class TemporalBasis(nn.Module):
    def __init__(self, sequence_length: int, num_basis: int):
        super().__init__()
        self.T = sequence_length
        self.K = num_basis
        self.register_buffer('t', torch.linspace(-1, 1, steps=sequence_length))

class PolynomialBasis(TemporalBasis):
    def forward(self) -> torch.Tensor:
        phi = torch.zeros(self.T, self.K, device=self.t.device)
        if self.K > 0: phi[:, 0] = 1.0
        if self.K > 1: phi[:, 1] = self.t
        for n in range(1, self.K - 1):
            term1 = (2 * n + 1) * self.t * phi[:, n]
            term2 = n * phi[:, n - 1]
            phi[:, n + 1] = (term1 - term2) / (n + 1)
        return phi

class FourierBasis(TemporalBasis):
    def forward(self) -> torch.Tensor:
        phi = torch.zeros(self.T, self.K, device=self.t.device)
        phi[:, 0] = 1.0
        for i in range(1, self.K):
            frequency = (i + 1) // 2
            if i % 2 == 1:
                phi[:, i] = torch.sin(np.pi * frequency * self.t)
            else:
                phi[:, i] = torch.cos(np.pi * frequency * self.t)
        return phi

class RBFBasis(TemporalBasis):
    def __init__(self, sequence_length, num_basis, bandwidth=0.2):
        super().__init__(sequence_length, num_basis)
        self.register_buffer('mu', torch.linspace(-1, 1, num_basis))
        self.sigma = bandwidth

    def forward(self) -> torch.Tensor:
        t_col = self.t.unsqueeze(1)
        mu_row = self.mu.unsqueeze(0)
        phi = torch.exp(- (t_col - mu_row)**2 / (2 * self.sigma**2))
        return phi

class WaveletBasis(TemporalBasis):
    def __init__(self, sequence_length, num_basis, width=0.1):
        super().__init__(sequence_length, num_basis)
        self.register_buffer('mu', torch.linspace(-1, 1, num_basis))
        self.width = width

    def forward(self) -> torch.Tensor:
        t_col = self.t.unsqueeze(1)
        mu_row = self.mu.unsqueeze(0)
        A = 2 / (np.sqrt(3 * self.width) * (np.pi ** 0.25))
        vec = (t_col - mu_row) / self.width
        phi = A * (1 - vec**2) * torch.exp(-0.5 * vec**2)
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
            if i == self.K + self.degree - 1: cond2 = t <= kv[i+1]
            basis_0.append((cond1 & cond2).float())
        
        bases = torch.cat(basis_0, dim=1)
        
        for d in range(1, self.degree + 1):
            new_bases = []
            for i in range(self.K + self.degree - d):
                # Term 1
                denom1 = kv[i+d] - kv[i]
                if denom1 < 1e-6:
                    # FIX: Use torch.zeros_like instead of 0.0
                    term1 = torch.zeros_like(bases[:, i:i+1])
                else:
                    term1 = ((t - kv[i]) / denom1) * bases[:, i:i+1]
                
                # Term 2
                denom2 = kv[i+d+1] - kv[i+1]
                if denom2 < 1e-6:
                    # FIX: Use torch.zeros_like instead of 0.0
                    term2 = torch.zeros_like(bases[:, i+1:i+2])
                else:
                    term2 = ((kv[i+d+1] - t) / denom2) * bases[:, i+1:i+2]
                    
                new_bases.append(term1 + term2)
            bases = torch.cat(new_bases, dim=1)
            
        return bases

# --- 2. Plotting Logic ---

def plot_all_bases(save_path: str = "bases.png"):
    """
    Plot all 5 basis families + CoMTE (identity / direct perturbation).

    CoMTE does not use a basis matrix — it perturbs the input directly
    (pixel-level). We visualise this as the identity matrix I(T,T)
    to show that every timestep has its own independent degree of freedom.
    """
    T = 100   # Time steps
    K = 5     # Number of basis functions
    
    bases = [
        ('Polynomial (Legendre)', PolynomialBasis(T, K)),
        ('Fourier (Sin/Cos)', FourierBasis(T, K)),
        ('RBF (Gaussian)', RBFBasis(T, K, bandwidth=0.3)),
        ('Wavelet (Mexican Hat)', WaveletBasis(T, K, width=0.15)),
        ('B-Spline (Cubic)', BSplineBasis(T, K, degree=3)),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()
    t_norm = np.linspace(0, 1, T)
    
    # Plot the 5 basis families
    for i, (name, basis_mod) in enumerate(bases):
        ax = axes_flat[i]
        phi = basis_mod().cpu().numpy()   # (T, K)
        for k in range(K):
            ax.plot(t_norm, phi[:, k], linewidth=1.2, alpha=0.8,
                    label=f'k={k}' if k < 6 else None)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel("Normalised Time")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, ncol=2)

    # Plot CoMTE: identity basis (each timestep is its own basis)
    ax_comte = axes_flat[5]
    K_show = min(K, T)
    # Show K "impulse" basis vectors (unit vectors at evenly spaced timesteps)
    impulse_locs = np.linspace(0, T - 1, K_show, dtype=int)
    for ki, loc in enumerate(impulse_locs):
        impulse = np.zeros(T)
        impulse[loc] = 1.0
        ax_comte.plot(t_norm, impulse, linewidth=1.5, alpha=0.85,
                      label=f't={loc}')
    ax_comte.set_title('CoMTE (Direct / Identity)', fontsize=10, fontweight='bold')
    ax_comte.set_xlabel("Normalised Time")
    ax_comte.set_ylabel("Perturbation weight")
    ax_comte.grid(True, alpha=0.3)
    ax_comte.legend(fontsize=7, ncol=2)
    ax_comte.annotate(
        'Each timestep × feature\nis independently editable',
        xy=(0.5, 0.5), xycoords='axes fraction',
        fontsize=8, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8),
    )

    fig.suptitle(f'Basis Functions — K={K} components, T={T} timesteps  (+CoMTE baseline)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_all_bases()
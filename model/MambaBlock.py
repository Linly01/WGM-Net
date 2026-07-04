import torch
import torch.nn as nn

class SimpleMamba(nn.Module):
    """
    轻量 Mamba block
    输入:  (B, T, N, D)
    输出:  (B, T, N, D)
    """
    def __init__(self, dim):
        super(SimpleMamba, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim
        )
        self.linear1 = nn.Linear(dim, dim*2)
        self.linear2 = nn.Linear(dim*2, dim)
        self.act = nn.GELU()

    def forward(self, x):
        B,T,N,D = x.shape
        x_res = x
        x = self.norm(x)
        x = x.permute(0,2,3,1).reshape(B*N, D, T)
        x = self.conv(x)
        x = x.reshape(B,N,D,T).permute(0,3,1,2)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        return x + x_res


def _import_mamba():
    """
    Best-effort import for standard Mamba-SSM.
    Returns the Mamba class or None if unavailable.
    """
    try:
        # Common import path in mamba-ssm v1.x
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
        return Mamba
    except Exception:
        pass
    try:
        # Some builds expose top-level symbol
        from mamba_ssm import Mamba  # type: ignore
        return Mamba
    except Exception:
        return None


class StandardMamba(nn.Module):
    """
    Standard Mamba-SSM block wrapper.

    Input:  (B, T, N, D)
    Output: (B, T, N, D)
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        Mamba = _import_mamba()
        if Mamba is None:
            raise ImportError(
                "mamba-ssm is not available. Install it (CUDA/WSL2 or a Windows wheel build) "
                "or switch back to SimpleMamba."
            )
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,D)
        B, T, N, D = x.shape
        x_res = x
        x = self.norm(x)
        x = x.reshape(B * N, T, D)  # (BN,T,D) for standard Mamba
        x = self.mamba(x)  # (BN,T,D)
        x = x.reshape(B, T, N, D)
        return x + x_res


def build_mamba_block(dim: int, use_standard: bool = True, **kwargs) -> nn.Module:
    """
    Factory to build either StandardMamba (preferred) or SimpleMamba.
    """
    if use_standard:
        # Do NOT silently fallback: if standard Mamba is requested,
        # surface the error so the caller knows the run is truly using Mamba-SSM.
        return StandardMamba(dim, **kwargs)
    return SimpleMamba(dim)
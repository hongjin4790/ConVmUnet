import copy
from timm.models.layers import DropPath
from functools import partial
from typing import Optional, Callable


from nets.conVmunet.ConvNext_V1_Origin import convnext_tiny
from nets.conVmunet.cross_scale_interaction import CIM


import math

from os.path import join as pjoin
from collections import OrderedDict
from torch.nn import Conv2d, Dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass


####################20240715 아직 수정못함 

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor): # d는 d_inner, k = 4, l = 64*64 4096, 
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L) # batch, 2, C(d_inner), H*W
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)  --> batch, 4, d_inner, L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape # 1, 64, 64, 96

        xz = self.in_proj(x)  # VSS block 에서 두갈래로 나눠지기 때문에 in_proj에서 channel을 2배하여  chunk로 나눠준다.
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)    B H W C -> 16, H, W, 96*2

        x = x.permute(0, 3, 1, 2).contiguous() # 16, 96*2, H, W
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

# SS2D 1Branch로 만든거
# class SS2D(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         d_state=16,
#         # d_state="auto", # 20240109
#         d_conv=3,
#         expand=2,
#         dt_rank="auto",
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init="random",
#         dt_scale=1.0,
#         dt_init_floor=1e-4,
#         dropout=0.,
#         conv_bias=True,
#         bias=False,
#         device=None,
#         dtype=None,
#         **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

#         self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs) # out channel의 *2를 제거
#         self.conv2d = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         self.act = nn.SiLU()

#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
#         del self.x_proj

#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
#         del self.dt_projs

#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
#         self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

#         # self.selective_scan = selective_scan_fn
#         self.forward_core = self.forward_corev0

#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None

#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank**-0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         dt_proj.bias._no_reinit = True

#         return dt_proj

#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log

#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D

#     def forward_corev0(self, x: torch.Tensor):
#         self.selective_scan = selective_scan_fn

#         B, C, H, W = x.shape
#         L = H * W
#         K = 4

#         x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
#         xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

#         xs = xs.float().view(B, -1, L) # (b, k * d, l)
#         dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
#         Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Ds = self.Ds.float().view(-1) # (k * d)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

#         out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)
#         assert out_y.dtype == torch.float

#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

#         return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

#     # an alternative to forward_corev1
#     def forward_corev1(self, x: torch.Tensor):
#         self.selective_scan = selective_scan_fn_v1

#         B, C, H, W = x.shape
#         L = H * W
#         K = 4

#         x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
#         xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

#         xs = xs.float().view(B, -1, L) # (b, k * d, l)
#         dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
#         Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
#         Ds = self.Ds.float().view(-1) # (k * d)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

#         out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#         ).view(B, K, -1, L)
#         assert out_y.dtype == torch.float

#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

#         return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

#     def forward(self, x: torch.Tensor, **kwargs):
#         B, H, W, C = x.shape


#         x = self.in_proj(x)

#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.act(self.conv2d(x)) # (b, d, h, w)
#         y1, y2, y3, y4 = self.forward_core(x)
#         assert y1.dtype == torch.float32
#         y = y1 + y2 + y3 + y4
#         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
#         y = self.out_norm(y)
#         out = self.out_proj(y)
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch
    """
    def __init__(self, in_channels=3):
        super(Embeddings, self).__init__()
        
        patch_size = (1,1)

        # self.hybrid_model = resnet50(pretrained=False)
        self.hybrid_model = convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT')

        in_channels = 384 # ConvNext out shape: 384 
        

        self.patch_embeddings = Conv2d(in_channels=in_channels, 
                                       out_channels=in_channels, 
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.dropout = Dropout(0.1)


    def forward(self, x):
        x, features = self.hybrid_model(x) # x.shape: (B, 384, 16, 16)
        x = self.patch_embeddings(x).permute(0, 2, 3, 1)  

        embeddings = self.dropout(x)
        return embeddings, features 

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

class VSSLayer(nn.Module):

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x

class VSSM(nn.Module):
    def __init__(self, depths = [2,2], dims = [384, 768],depths_decoder = [2, 2],dims_decoder= [768, 384], 
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super(VSSM, self).__init__()
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layer = nn.ModuleList()
        self.num_layers = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] #drop_path_rate, sum(depths)
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop = drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
            )
            self.layer.append(layer)
    

    def forward_features(self, x):
        x = self.pos_drop(x)
        for layer_block in self.layer:
            x = layer_block(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_features_up(x, skip_list)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embeddings = Embeddings() # CNN part
        self.vssm = VSSM() # ConvNeXt Out dim 384

    def forward(self, x):
        # x.shape: (B, 3, 256, 256)
        embedding_output, features = self.embeddings(x)
        # embedding_output.shape: (B, 384, 16, 16)
        # print("embedding_output: ", embedding_output.shape)
        encoded = self.vssm(embedding_output) 
        # encoded.shape: (B, 8, 8, 768)
        encoded = encoded.permute(0, 3, 1, 2)
        return encoded, features
    


class ChannelAttention(nn.Module): 
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module): # 1x1 conv
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)]) # smoothed conv ; 논문에서 f4과정

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):#[f1,f2,f3,f4]
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x) # hadamard product 과정 >> hadamard product는 원소별 곱셉이기 때문에 걍 곱하면됨

        return ans


class ConvNextVMUNet(nn.Module):
    def __init__(self, num_classes=1, mid_channel = 48, deep_supervision=True):
        super(ConvNextVMUNet, self).__init__()
        
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.encoder = Encoder()
        self.cim = CIM(1440, channels=[96, 192, 384, 768])

        # SDI
        self.ca_1 = ChannelAttention(2*mid_channel)
        self.sa_1 = SpatialAttention() 

        self.ca_2 = ChannelAttention(4*mid_channel)
        self.sa_2 = SpatialAttention()
        # TODO 320 or mid_channel * 8?
        self.ca_3 = ChannelAttention(8*mid_channel)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(16*mid_channel)
        self.sa_4 = SpatialAttention()
        
        self.Translayer_1 = BasicConv2d(2*mid_channel, mid_channel, 1)
        self.Translayer_2 = BasicConv2d(4*mid_channel, mid_channel, 1)
        self.Translayer_3 = BasicConv2d(8*mid_channel, mid_channel, 1)
        self.Translayer_4 = BasicConv2d(16*mid_channel, mid_channel, 1)  

        self.sdi_1 = SDI(mid_channel)
        self.sdi_2 = SDI(mid_channel)
        self.sdi_3 = SDI(mid_channel)
        self.sdi_4 = SDI(mid_channel)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(mid_channel, num_classes, 1, 1) for _ in range(4)])
        
        
    
        self.deconv2 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv6 = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)
        
            
    def forward(self, x):
        seg_outs = []
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)    
        x, features = self.encoder(x)

        fet = tuple([features[0], features[1], features[2], x])

        f1, f2, f3, f4 = self.cim(fet)
        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        # use sdi  
        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1) # f1 [2, 48, 64, 64]
        
        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2) # f2 [2, 48, 32, 32]

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3) # f3 [2, 48, 16, 16]

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4) # f4 [2, 48, 8, 8]
        
        
        f41 = self.sdi_4([f1, f2, f3, f4], f4) # [2, 48, 8, 8]
        f31 = self.sdi_3([f1, f2, f3, f4], f3) # [2, 48, 16, 16]
        f21 = self.sdi_2([f1, f2, f3, f4], f2) # [2, 48, 32, 32]
        f11 = self.sdi_1([f1, f2, f3, f4], f1) # [2, 48, 64, 64]
        
     
        seg_outs.append(self.seg_outs[0](f41)) # seg_outs[0] [2, 1, 8, 8]

        y = self.deconv2(f41) + f31 # y shape: [2, 48, 16, 16]
        seg_outs.append(self.seg_outs[1](y)) # seg_outs[1] [2, 1, 16, 16]

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y)) # seg_outs[2] [2, 1, 32, 32]

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y)) # seg_outs[3] [2, 1, 64, 64]
        
        for i, o in enumerate(seg_outs): # 4배 샘플링
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')

        if self.deep_supervision:
            
            temp = seg_outs[::-1]  # 0 [2, 1, 256, 256] 1 [2, 1, 128, 128]
            out_0 = temp[0]
            out_1 = temp[1]
            out_1 = self.deconv6(out_1)
            return torch.sigmoid(out_0 + out_1)  #  [2, 1, 256, 256]
        else:
            if self.num_classes == 1: return torch.sigmoid(seg_outs[-1])
            else: return seg_outs[-1]
        
     

if __name__ == '__main__':

    inputs = torch.randn(1, 3, 256, 256).cuda()
 

    enc = Encoder().cuda()
    encoded, fet = enc(inputs)
    print(fet[0].shape)
    print(fet[1].shape)
    print(fet[2].shape)
    print(encoded.shape)

    # torch.Size([1, 16, 16, 384])
    # torch.Size([1, 96, 64, 64])
    # torch.Size([1, 192, 32, 32])
    # torch.Size([1, 384, 16, 16])
    
    # dec = Decoder().cuda()
    # decoed = dec(encoded, fet)
    

    model = ConvNextVMUNet(num_classes=1).cuda()
    output = model(torch.randn(1, 3, 256, 256).cuda())
    print(output.shape)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    ############## 일단 인코더부분은 수정은 다 했고 테스트해봐야됨  20240626
    ############## 모델 수정 완료 
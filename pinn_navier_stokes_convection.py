import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path

"""
We will be solving the Navier-Stokes equations for incompressible flow with slip boundary conditions, coupled to a heat 
equation for convection, using physics-informed neural networks (PINNs). The governing equations are

Ω = x ∈ [0, L] ∪ y ∈ [0, D]                                   (2D spatial domain)
∇⋅ū = 0,                            x̄ ∈ Ω, t ∈ [0,T]          (incompressibility)
∂ū/∂t + (ū⋅∇)ū = ν∇²ū - ∇p + f_b,   x̄ ∈ Ω, t ∈ [0,T]          (momentum conservation)
∂T/∂t + (ū⋅∇)T = κ⋅∇²T,             x̄ ∈ Ω, t ∈ [0, T]         (temperature transport)
u = 0,                              x̄ ∈ ∂Ω_x, t ∈ [0,T]       (horizontal momentum can't leave system)
v = 0,                              x̄ ∈ ∂Ω_y, t ∈ [0,T]       (vertical momentum can't leave system)
∂T/∂n = 0                           x̄ ∈ ∂Ω, t ∈ [0,T]         (Neumann boundary conditions for heat)

The buoyancy force f_b is modeled as:
f_b = (0, g⋅κ⋅(T - T₀)), with T₀ = 0                          (Boussinesq body force)

We will solve for the x velocity, y velocity, pressure, and temperature (u, v, p, and T). We will write the solution as the trial functions

u(x, y, t; θ) = N_u(x, y, t; θ)⋅(L - x)⋅x,   
v(x, y, t; φ) = N_v(x, y, t; φ)⋅(D - y)⋅y, 
p(x, y, t; ψ) = N_p(x, y, t; ψ), 
T(x, y, t; ξ) = N_p(x, y, t; ξ), 

for a neural network, N(x, t, t; Λ) → [u, v, p, T]. The solution must satisfy ∂ū/∂t + (ū⋅∇)ū - ν∇²ū + ∇p - f = 0, incompressibility, 
∂T/∂t + (ū⋅∇)T - κ⋅∇²T = 0, and the initial condition, so we will define the loss as 

L_1 = ∫_Ω ∫_0^T [∂u/∂t + (ū⋅∇)u - ν∇²u + ∇p - f_x]² dt dx̄,

L_2 = ∫_Ω ∫_0^T [∂u/∂t + (ū⋅∇)v - ν∇²v + ∇p - f_y]² dt dx̄,

L_3 = ∫_Ω ∫_0^T [∇⋅ū]² dt dx̄,

L_4 = ∫_0^L ∫_0^T [∂T/∂t + (ū⋅∇)T - κ⋅∇²T]² dt dx,

L_5 = ∫_∂Ω [∂T/∂n]² dt dx,

which the neural network will be trained to minimize. We will validate using the finite difference method, taken from old code of mine, which
itself was validated by Poiseuille and Couette flow analytical solutions.
"""

# Hyperparameters -------------------------------------------------------------
ν, κ = 0.1, 0.01  # diffusion and buoyancy coefficients 
g = 200.0  # gravity 
L, D, T_max = 1.0, 1.0, 1.0  # domain size and timespan 
f_x = lambda x: torch.zeros_like(x) # horizontal body force
f_y = lambda y, T: torch.zeros_like(y) + g * κ * (T - 0.0) # vertical body force
u_initial = lambda x, y, t: 0.2 * t

N_x, N_y = 50, 50  # FD grid (for validation)
N_pts = 700  # random collocation pts (training)
NEURONS = 128  # hidden‑layer width
EPOCHS = 1000  # iterations

torch.manual_seed(0)
np.random.seed(0)

x = np.linspace(0.0, L, N_x, dtype=float)
y = np.linspace(0.0, D, N_y, dtype=float)
Δx = x[1] - x[0]
Δy = y[1] - y[0]
x_torch = torch.rand((N_pts, 1), requires_grad=True)
y_torch = torch.rand((N_pts, 1), requires_grad=True)
t_torch = torch.rand(N_pts, 1, requires_grad=True)

# Initial hotspot 
x_center = L * 0.5
y_center = D * 0.5
width = 0.1 

u_trial = lambda x, y, t: N(torch.cat([x, y, t], dim=1))[:, 0:1] * (L - x) * x
v_trial = lambda x, y, t: N(torch.cat([x, y, t], dim=1))[:, 1:2] * (D - y) * y
p_trial = lambda x, y, t: N(torch.cat([x, y, t], dim=1))[:, 2:3]
T_trial = lambda x, y, t: N(torch.cat([x, y, t], dim=1))[:, 3:4]

N_ic = 500
x_ic = torch.rand((N_ic, 1), requires_grad=True)
y_ic = torch.rand((N_ic, 1), requires_grad=True)

# Physics Informed Neural Network ---------------------------------------------
N = nn.Sequential(
    nn.Linear(3, NEURONS),
    nn.Tanh(),
    nn.Linear(NEURONS, NEURONS),
    nn.Sigmoid(),
    nn.Linear(NEURONS, NEURONS),
    nn.Sigmoid(),
    nn.Linear(NEURONS, NEURONS),
    nn.Tanh(),
    nn.Linear(NEURONS, 4)
)

# Loss & Optimizer ------------------------------------------------------------
def horizontal_loss(x, y, t):  # we want to satisfy ∂u/∂t + (ū⋅∇)u - ν∇²u + ∇p - f_x = 0
    u = u_trial(x, y, t)
    v = v_trial(x, y, t)
    p = p_trial(x, y, t)

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    
    u_res = u_t + (u * u_x + v * u_y) - ν * (u_xx + u_yy) + p_x - f_x(x)

    return torch.mean((u_res) ** 2)

def vertical_loss(x, y, t):  # we want to satisfy ∂u/∂t + (ū⋅∇)v - ν∇²v + ∇p - f_y = 0
    u = u_trial(x, y, t)
    v = v_trial(x, y, t)
    p = p_trial(x, y, t)
    T = T_trial(x, y, t)

    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(v), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    
    v_res = v_t + (u * v_x + v * v_y) - ν * (v_xx + v_yy) + p_y - f_y(y, T)

    return torch.mean((v_res) ** 2)

def incompressibility_loss(x, y, t):  # we want to satsify ∇⋅ū = 0
    u = u_trial(x, y, t)
    v = v_trial(x, y, t)

    u_x = torch.autograd.grad(u, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]

    return torch.mean((u_x + v_y) ** 2)

def heat_equation_loss(x, y, t):  # we want to satisfy ∂T/∂t + (ū⋅∇)T - κ⋅∇²T = 0
    T = T_trial(x, y, t)
    u = u_trial(x, y, t)
    v = v_trial(x, y, t)
    
    T_t = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0]

    # note that (ū⋅∇)T is NOT the same as (∇⋅ū)T
    return torch.mean((T_t + (u * T_x + v * T_y) - κ * (T_xx + T_yy)) ** 2)

def neumann_heat_loss(x, y, t):  # we want to satisfy ∂T/∂n = 0 for x̄ ∈ ∂Ω
    x_0 = torch.zeros_like(x, requires_grad=True)
    y_0 = torch.zeros_like(y, requires_grad=True)
    x_L = torch.ones_like(x, requires_grad=True) * L
    y_D = torch.ones_like(y, requires_grad=True) * D

    T_lef = T_trial(x_0, y, t)  # Left boundary: x = 0
    T_rig = T_trial(x_L, y, t)  # Right boundary: x = L
    T_bot = T_trial(x, y_0, t)  # Bottom boundary: y = 0
    T_top = T_trial(x, y_D, t)  # Top boundary: y = D

    T_lef_x = torch.autograd.grad(T_lef, x_0, torch.ones_like(T_lef), create_graph=True)[0]
    T_rig_x = torch.autograd.grad(T_rig, x_L, torch.ones_like(T_rig), create_graph=True)[0]
    T_bot_y = torch.autograd.grad(T_bot, y_0, torch.ones_like(T_bot), create_graph=True)[0]
    T_top_y = torch.autograd.grad(T_top, y_D, torch.ones_like(T_top), create_graph=True)[0]

    return torch.mean(T_lef_x ** 2 + T_rig_x ** 2 + T_bot_y ** 2 + T_top_y ** 2)

def initial_heat_loss():  # we want an initial hotspot in the middle of the domain
    t0 = torch.zeros_like(x_ic)
    T = T_trial(x_ic, y_ic, t0)

    T_true = torch.exp(-((x_ic - x_center)**2 + (y_ic - y_center)**2) / (2 * width**2))

    return torch.mean((T - T_true) ** 2)

def total_loss(x, y, t):
    loss = (
        horizontal_loss(x, y, t) + vertical_loss(x, y, t) + incompressibility_loss(x, y, t) + 
        heat_equation_loss(x, y, t) + neumann_heat_loss(x, y, t) + 100 * initial_heat_loss()
            )

    return loss


optimizer1 = torch.optim.Adam(N.parameters(), lr=1e-3)
optimizer2 = torch.optim.LBFGS(N.parameters())

# Training Loop ---------------------------------------------------------------
for epoch in range(EPOCHS): 
    # 1. & 2. Forward Pass & Loss
    loss = total_loss(x_torch, y_torch, t_torch)

    # 3. Optimizer Zero Grad
    optimizer1.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Gradient Descent
    optimizer1.step()

    print(epoch)

def closure():  # must have a closure function written like this for LBFG to work
    # 1. & 2. Forward Pass & Loss
    loss = total_loss(x_torch, y_torch, t_torch)  

    # 3. Optimizer Zero Grad
    optimizer2.zero_grad()

    # 4. Backpropagation
    loss.backward()
    return loss

for epoch in range(10):
    # 5. Gradient Descent
    optimizer2.step(closure)

    print(epoch)

# Finite Difference Validation ------------------------------------------------

# Will add later 

# Plot (this plotting section was written by ChatGPT) -------------------------
step = 4
arrow_mult = 1.0
t0 = 0.20
N_FRAMES = 100

# Set up evaluation grid (for plotting)
nx, ny = 120, 120
xs = torch.linspace(0, L, nx)
ys = torch.linspace(0, D, ny)
xg, yg = torch.meshgrid(xs, ys, indexing='ij')
xy = torch.stack([xg.flatten(), yg.flatten()], dim=1)
xg_np, yg_np = xg.numpy(), yg.numpy()

# ── 1. Replace the fields() helper ────────────────────────────────────────────
@torch.no_grad()
def fields(t_val):
    t_col = torch.full((nx * ny, 1), t_val)
    inp   = torch.cat([xy, t_col], dim=1)

    u = u_trial(inp[:, 0:1], inp[:, 1:2], inp[:, 2:3]).view(nx, ny)
    v = v_trial(inp[:, 0:1], inp[:, 1:2], inp[:, 2:3]).view(nx, ny)
    T = T_trial(inp[:, 0:1], inp[:, 1:2], inp[:, 2:3]).view(nx, ny)  # ← NEW

    return u.numpy(), v.numpy(), T.numpy()          # return T instead of speed

# Evaluate initial field and precompute mask
u0, v0, T0 = fields(t0)
mask = np.s_[::step, ::step]
x_masked, y_masked = xg_np[mask], yg_np[mask]

# Set up plot
fig, ax = plt.subplots(figsize=(6, 5))
pc = ax.pcolormesh(xg_np, yg_np, T0, shading='auto', cmap='viridis')
quiv = ax.quiver(x_masked, y_masked,
                 arrow_mult * u0[mask], arrow_mult * v0[mask],
                 scale_units='xy', scale=1.0, width=0.003, color='white')
cb = fig.colorbar(pc, ax=ax, label='Temperature')
title = ax.set_title(f't = {t0:.2f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Precompute speed range for consistent color scaling
_, _, s_final = fields(T_max)
T_max_temp = max(T0.max(), s_final.max())
pc.set_clim(0, T_max_temp)

# Animation function
def update(frame_idx):
    t_val = t0 + frame_idx / (N_FRAMES - 1) * (T_max - t0)
    u, v, s = fields(t_val)
    pc.set_array(s.ravel())
    quiv.set_UVC(arrow_mult * u[mask], arrow_mult * v[mask])
    title.set_text(f't = {t_val:.2f}')
    return pc, quiv, title

# Animate
ani = anim.FuncAnimation(fig, update, frames=N_FRAMES, blit=False, interval=50)
plt.show()

# Save ------------------------------------------------------------------------
model_path = Path("file/path/for/saved/model")
model_name = "navier_stokes_convection_model"

model_path.mkdir(parents=True, exist_ok=True)
model_save_path = model_path / model_name
torch.save(obj=N.state_dict(), f=model_save_path)


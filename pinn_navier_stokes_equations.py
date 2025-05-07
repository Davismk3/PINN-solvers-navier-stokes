import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path

"""
We will be solving the Navier-Stokes equations for incompressible flow with slip boundary conditions using physics-informed neural
networks (PINNs). The governing equations are

Ω = x ∈ [0, L] ∪ y ∈ [0, D]                                 (2D spatial domain)
∇⋅ū = 0,                          x̄ ∈ Ω, t ∈ [0,T]          (incompressibility)
∂ū/∂t + (ū⋅∇)ū = ν∇²ū - ∇p + f,   x̄ ∈ Ω, t ∈ [0,T]          (momentum conservation)
u = 0,                            x̄ ∈ ∂Ω_x, t ∈ [0,T]       (momentum can't leave system)
v = 0,                            x̄ ∈ ∂Ω_y, t ∈ [0,T]       (momentum can't leave system)
u = 0.2⋅t,                        y = 0, t ∈ [0,T]          (horizontal initial velocity has a time dependence)

We will solve for the x velovity, y velocity, and pressure (u, v, and p). We will write the solution as the trial functions

u(x, y, t; θ) = N_u(x, y, t; θ)⋅(L - x)⋅x,   
v(x, y, t; φ) = N_v(x, y, t; φ)⋅(D - y)⋅y, 
p(x, y, t; ψ) = N_p(x, y, t; ψ), 

for a neural network, N(x, t, t; Λ) → [u, v, p]. The solution must satisfy ∂ū/∂t + (ū⋅∇)ū - ν∇²ū + ∇p - f = 0, incompressibility, and the initial
condition, so we will define the loss as 

L_1 = ∫_Ω ∫_0^T [∂u/∂t + (ū⋅∇)u - ν∇²u + ∇p - f_x]² dt dx̄,

L_2 = ∫_Ω ∫_0^T [∂u/∂t + (ū⋅∇)v - ν∇²v + ∇p - f_y]² dt dx̄,

L_3 = ∫_Ω ∫_0^T [∇⋅ū]² dt dx̄,

L_4 = ∫_0^L ∫_0^T [u(x, 0, t; θ) - 1.0]² dt dx,

which the neural network will be trained to minimize. We will validate using the finite difference method, taken from old code of mine, which
itself was validated by Poiseuille and Couette flow analytical solutions.
"""

# Hyperparameters -------------------------------------------------------------
ν = 0.1  # diffusion coefficient 
L, D, T = 1.0, 1.0, 1.0  # domain size
f_x = lambda x: torch.zeros_like(x) # horizontal body force
f_y = lambda y: torch.zeros_like(y) # vertical body force
u_initial = lambda x, y, t: 0.2 * t

N_x, N_y = 100, 1500  # FD grid (for validation)
N_pts = 500  # random collocation pts (training)
NEURONS = 100  # hidden‑layer width
EPOCHS = 500  # iterations

torch.manual_seed(0)
np.random.seed(0)

x = np.linspace(0.0, L, N_x, dtype=float)
y = np.linspace(0.0, D, N_y, dtype=float)
Δx = x[1] - x[0]
Δy = y[1] - y[0]
x_torch = torch.rand((N_pts, 1), requires_grad=True)
y_torch = torch.rand((N_pts, 1), requires_grad=True)
t_torch = torch.rand(N_pts, 1, requires_grad=True)

u_trial = lambda x, y, t: N(torch.cat([x, y, t], dim=1))[:, 0:1] * (L - x) * x
v_trial = lambda x, y, t: N(torch.cat([x, y, t], dim=1))[:, 1:2] * (D - y) * y
p_trial = lambda x, y, t: N(torch.cat([x, y, t], dim=1))[:, 2:3]

# Physics Informed Neural Network ---------------------------------------------
N = nn.Sequential(
    nn.Linear(3, NEURONS),
    nn.Tanh(),
    nn.Linear(NEURONS, NEURONS),
    nn.Sigmoid(),
    nn.Linear(NEURONS, NEURONS),
    nn.Sigmoid(),
    nn.Linear(NEURONS, 3)
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
    
    u_res = u_t + (u_x + v_y) * u - ν * (u_xx + u_yy) + p_x - f_x(x)

    return torch.mean((u_res) ** 2)

def vertical_loss(x, y, t):  # we want to satisfy ∂u/∂t + (ū⋅∇)v - ν∇²v + ∇p - f_y = 0
    u = u_trial(x, y, t)
    v = v_trial(x, y, t)
    p = p_trial(x, y, t)

    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(v), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    
    v_res = v_t + (u_x + v_y) * v - ν * (v_xx + v_yy) + p_y - f_y(y)

    return torch.mean((v_res) ** 2)

def incompressibility_loss(x, y, t):  # we want to satsify ∇⋅ū = 0
    u = u_trial(x, y, t)
    v = v_trial(x, y, t)

    u_x = torch.autograd.grad(u, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]

    return torch.mean((u_x + v_y) ** 2)

def initial_loss(x, y, t):  # we want to satisfy u = 1.0 for y = 0, t ∈ [0,T]
    y_0 = torch.zeros_like(y)

    return torch.mean((u_trial(x, y_0, t) - u_initial(x, y, t)) ** 2)
    
def total_loss(x, y, t):
    return horizontal_loss(x, y, t) + vertical_loss(x, y, t) + incompressibility_loss(x, y, t) + initial_loss(x, y, t)

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

for epoch in range(EPOCHS):
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

# Evaluate u, v, speed fields at a fixed time
@torch.no_grad()
def fields(t_val):
    t_col = torch.full((nx * ny, 1), t_val)
    inp = torch.cat([xy, t_col], dim=1)
    u = u_trial(inp[:, 0:1], inp[:, 1:2], inp[:, 2:3]).view(nx, ny)
    v = v_trial(inp[:, 0:1], inp[:, 1:2], inp[:, 2:3]).view(nx, ny)
    speed = torch.sqrt(u**2 + v**2)
    return u.numpy(), v.numpy(), speed.numpy()

# Evaluate initial field and precompute mask
u0, v0, s0 = fields(t0)
mask = np.s_[::step, ::step]
x_masked, y_masked = xg_np[mask], yg_np[mask]

# Set up plot
fig, ax = plt.subplots(figsize=(6, 5))
pc = ax.pcolormesh(xg_np, yg_np, s0, shading='auto', cmap='viridis')
quiv = ax.quiver(x_masked, y_masked,
                 arrow_mult * u0[mask], arrow_mult * v0[mask],
                 scale_units='xy', scale=1.0, width=0.003, color='k')
cb = fig.colorbar(pc, ax=ax, label='|u|')
title = ax.set_title(f't = {t0:.2f}')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Precompute speed range for consistent color scaling
_, _, s_final = fields(T)
s_max = max(s0.max(), s_final.max())
pc.set_clim(0, s_max)

# Animation function
def update(frame_idx):
    t_val = t0 + frame_idx / (N_FRAMES - 1) * (T - t0)
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
model_name = "navier_stokes_spiral_model"

model_path.mkdir(parents=True, exist_ok=True)
model_save_path = model_path / model_name
torch.save(obj=N.state_dict(), f=model_save_path)


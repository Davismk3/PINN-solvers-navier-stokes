import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path

# Reconstruct Model -----------------------------------------------------------
model_path = Path("file/path/for/saved/model")
model_name = "navier_stokes_spiral_model"
model_save_path = model_path / model_name

ν = 0.1
L, D, T = 1.0, 1.0, 1.0
NEURONS = 100

loaded_N = nn.Sequential(
    nn.Linear(3, NEURONS),
    nn.Tanh(),
    nn.Linear(NEURONS, NEURONS),
    nn.Sigmoid(),
    nn.Linear(NEURONS, NEURONS),
    nn.Sigmoid(),
    nn.Linear(NEURONS, 3)
)

u_trial = lambda x, y, t: loaded_N(torch.cat([x, y, t], dim=1))[:, 0:1] * (L - x) * x
v_trial = lambda x, y, t: loaded_N(torch.cat([x, y, t], dim=1))[:, 1:2] * (D - y) * y
p_trial = lambda x, y, t: loaded_N(torch.cat([x, y, t], dim=1))[:, 2:3]

# Load ------------------------------------------------------------------------
loaded_N.load_state_dict(torch.load(f=model_save_path))
loaded_N.eval()

# Plot (Written Mostly By ChatGPT) --------------------------------------------
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
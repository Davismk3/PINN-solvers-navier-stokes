import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as anim
from pathlib import Path

# Reconstruct Model -----------------------------------------------------------
model_path = Path("/Users/michaeldavis/Desktop/Python/PDEs/PINNs/Saved_Models")
model_name = "navier_stokes_spiral_model_test"
model_save_path = model_path / model_name

L, D, T_max = 1.0, 1.0, 1.0  # domain size and timespan 
N_x, N_y = 10, 10  # FD grid (for validation)
N_pts = 500  # random collocation pts (training)
NEURONS = 128  # hidden‑layer width

x = np.linspace(0.0, L, N_x, dtype=float)
y = np.linspace(0.0, D, N_y, dtype=float)
Δx = x[1] - x[0]
Δy = y[1] - y[0]
x_torch = torch.rand((N_pts, 1), requires_grad=True)
y_torch = torch.rand((N_pts, 1), requires_grad=True)
t_torch = torch.rand(N_pts, 1, requires_grad=True)

loaded_N = nn.Sequential(
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

u_trial = lambda x, y, t: loaded_N(torch.cat([x, y, t], dim=1))[:, 0:1] * (L - x) * x
v_trial = lambda x, y, t: loaded_N(torch.cat([x, y, t], dim=1))[:, 1:2] * (D - y) * y
p_trial = lambda x, y, t: loaded_N(torch.cat([x, y, t], dim=1))[:, 2:3]
T_trial = lambda x, y, t: loaded_N(torch.cat([x, y, t], dim=1))[:, 3:4]

# Load ------------------------------------------------------------------------
loaded_N.load_state_dict(torch.load(f=model_save_path))
loaded_N.eval()

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
model_path = Path("/Users/michaeldavis/Desktop/Python/PDEs/PINNs/Saved_Models")
model_name = "navier_stokes_spiral_model_test"

model_path.mkdir(parents=True, exist_ok=True)
model_save_path = model_path / model_name
torch.save(obj=loaded_N.state_dict(), f=model_save_path)


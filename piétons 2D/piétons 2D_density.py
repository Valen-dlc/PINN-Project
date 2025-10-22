# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 14:05:25 2025

@author: Didier
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Paramètres de la rue ---
L = 100.0   # longueur (m)
W = 7       # largeur (m)

# --- Paramètres des piétons ---
N = 100
v_desired = 1.3
relax_time = 0.5
A = 2.0
B = 0.5
wall_repulsion = 1
dt = 0.5
steps = 90

# --- Initialisation ---
positions = np.zeros((N, 2))
positions[:, 0] = np.random.uniform(0, L/4, N)
positions[:, 1] = np.random.uniform(0, W, N)
velocities = np.zeros((N, 2))

# Tableau pour stocker les positions et vitesses
All_positions = np.zeros((steps, N, 2))
All_velocities = np.zeros((steps, N, 2))

# --- Force sociale ---
def social_forces(positions, velocities):
    Nb = positions.shape[0]
    forces = np.zeros_like(positions)

    v0 = np.zeros_like(velocities)
    v0[:, 0] = v_desired
    forces += (v0 - velocities) / relax_time

    # Répulsion entre piétons
    for i in range(Nb):
        for j in range(i+1, Nb):
            d = positions[i] - positions[j]
            dist = np.linalg.norm(d)
            if dist > 1e-5:
                f = A * np.exp(-dist / B) * (d / dist)
                forces[i] += f
                forces[j] -= f

    # Répulsion avec les murs
    for i in range(Nb):
        y = positions[i, 1]
        forces[i, 1] += wall_repulsion / (y + 1e-3)
        forces[i, 1] -= wall_repulsion / (W - y + 1e-3)

    return forces

# --- Mise à jour ---
frame_idx = 0  # compteur global

def update(frame):
    global positions, velocities, frame_idx

    f = social_forces(positions, velocities)
    velocities += f * dt
    positions += velocities * dt

    # Enregistrement des positions à ce step
    if frame_idx < steps:
        # on garde la taille constante en remplissant avec NaN pour les piétons sortis
        All_positions[frame_idx, :positions.shape[0], :] = positions
        All_positions[frame_idx, positions.shape[0]:, :] = np.nan
        All_velocities[frame_idx, :positions.shape[0], :] = velocities
        All_velocities[frame_idx, positions.shape[0]:, :] = np.nan
        frame_idx += 1

    # Retirer les piétons sortis
    mask = positions[:, 0] < L
    positions = positions[mask]
    velocities = velocities[mask]

    scat.set_offsets(positions)
    return scat,

# --- Animation ---
fig, ax = plt.subplots(figsize=(12, 3))
scat = ax.scatter(positions[:, 0], positions[:, 1], c="blue")
ax.set_xlim(0, L)
ax.set_ylim(0, W)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Simulation de piétons (animation)")

ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=True, repeat=False)

plt.show()

# À la fin, All_positions contient toutes les positions sauvegardées
fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.set_xlim(0, L)
ax2.set_ylim(0, W)
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_title("Trajectoires des piétons")

# 🔹 Trace tous les piétons (ou un sous-échantillon si c'est trop dense)
subset = np.linspace(0, N-1, min(N, 20), dtype=int)  # max 20 piétons pour lisibilité
for i in subset:
    traj = All_positions[:, i, :]
    ax2.plot(traj[:, 0], traj[:, 1], alpha=0.3)

plt.show()


# ==== paramètres du cadrillage ====
nx, ny = 10, 2       # nb de cases en x et y (ajuste si besoin)
normalize =False    # True: densité normalisée par nb de piétons actifs à chaque frame

# ==== bords des cellules ====
x_edges = np.linspace(0, L, nx+1)
y_edges = np.linspace(0, W, ny+1)
x_cent = 0.5*(x_edges[:-1] + x_edges[1:])
y_cent = 0.5*(y_edges[:-1] + y_edges[1:])

steps = All_positions.shape[0]
D = np.zeros((steps, ny, nx), dtype=float)   # densité par frame

# ==== calcule la densité par frame ====
for t in range(steps):
    pos_t = All_positions[t]                 # (N, 2)
    mask = np.isfinite(pos_t[:,0]) & np.isfinite(pos_t[:,1])
    x = pos_t[mask, 0]
    y = pos_t[mask, 1]
    H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])  # (ny, nx)

    if normalize:
        H = H / max(1, mask.sum())          # normalisation par nb de piétons présents
    D[t] = H


# ==== 2) animation frame par frame ====
fig, ax = plt.subplots(figsize=(12, 3))
quad = ax.pcolormesh(x_edges, y_edges, D[0], shading='auto')
cbar = plt.colorbar(quad, ax=ax, label=("densité normalisée" if normalize else "comptes"))
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_xlim(0, L); ax.set_ylim(0, W)
ax.set_title("Densité par pas de temps")

# échelle de couleur fixe pour éviter le pompage (prend des percentiles robustes)
vmin = np.percentile(D, 5)
vmax = np.percentile(D, 95)
quad.set_clim(vmin, vmax)

def update(k):
    quad.set_array(D[k].ravel())  # pcolormesh attend un array aplati (ny*nx,)
    ax.set_title(f"Densité au pas t={k+1}/{steps}")
    return (quad,)

ani = FuncAnimation(fig, update, frames=steps, interval=60, blit=False, repeat=False)
plt.tight_layout(); plt.show()

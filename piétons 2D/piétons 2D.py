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
W = 7    # largeur (m)

# --- Paramètres des piétons ---
N = 15                    # nombre de piétons
v_desired = 1.3             # vitesse désirée (m/s)
relax_time = 0.5            # temps de relaxation vers v_desired
A = 2.0                     # intensité répulsion piéton-piéton
B = 0.5                 # distance caractéristique répulsion
wall_repulsion = 1     # intensité répulsion murs
dt = 0.5               # pas de temps
steps = 200              # nombre de pas simulés

# --- Initialisation ---
positions = np.zeros((N, 2))
positions[:, 0] = np.random.uniform(0, L/4, N)   # départ côté gauche
positions[:, 1] = np.random.uniform(0, W, N)     # position latérale
velocities = np.zeros((N, 2))

# --- Force sociale ---
def social_forces(positions, velocities):
    N = positions.shape[0]
    forces = np.zeros_like(positions)

    # Force désirée vers la sortie (droite)
    v0 = np.zeros_like(velocities)
    v0[:, 0] = v_desired
    forces += (v0 - velocities) / relax_time

    # Répulsion entre piétons
    for i in range(N):
        for j in range(i+1, N):
            d = positions[i] - positions[j]
            dist = np.linalg.norm(d)
            if dist > 1e-5:
                f = A * np.exp(-dist / B) * (d / dist)
                forces[i] += f
                forces[j] -= f

    # Répulsion avec les murs
    for i in range(N):
        y = positions[i, 1]
        forces[i, 1] += wall_repulsion / (y + 1e-3)        # mur bas
        forces[i, 1] -= wall_repulsion / (W - y + 1e-3)    # mur haut

    return forces

# --- Mise à jour ---
def update(frame):
    global positions, velocities, ani
    if positions.shape[0] == 0:
        # Plus de piétons à afficher, arrêter l'animation
        ani.event_source.stop()
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    f = social_forces(positions, velocities)
    velocities += f * dt
    positions += velocities * dt

    # retirer les piétons sortis
    mask = positions[:, 0] < L
    positions = positions[mask]
    velocities = velocities[mask]

    if positions.shape[0] == 0:
        ani.event_source.stop()
        scat.set_offsets(np.empty((0, 2)))
        return scat,

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


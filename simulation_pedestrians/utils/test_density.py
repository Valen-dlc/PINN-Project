import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from simulation_pedestrians.run_simulation import run_pedestrian_simulation

def compute_density(all_positions, L=100.0, W=7.0, nx=100, ny=14, normalize=False):
    """
    Calcule la densité spatiale des piétons pour chaque frame.
    all_positions : array (steps, N, 2)
    Retourne D : array (steps, ny, nx)
    """
    steps, N, _ = all_positions.shape
    
    # Bords du maillage
    x_edges = np.linspace(0, L, nx + 1)
    y_edges = np.linspace(0, W, ny + 1)
    
    # Matrice densité
    D = np.zeros((steps, ny, nx))
    
    # Boucle sur les frames 
    for t in range(steps):
        pos_t = all_positions[t]
        mask = np.isfinite(pos_t[:, 0]) & np.isfinite(pos_t[:, 1])
        if not np.any(mask): 
            continue
        x, y = pos_t[mask, 0], pos_t[mask, 1]
        H, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
        if normalize:
            H /= max(1, mask.sum())
        D[t] = H
        
    return D, x_edges, y_edges


def plot_trajectories(all_positions, L=100.0, W=7.0, max_traj=20):
    """Affiche quelques trajectoires."""
    N = all_positions.shape[1]
    subset = np.linspace(0, N-1, min(N, max_traj), dtype=int)
    plt.figure(figsize=(12, 3))
    for i in subset:
        traj = all_positions[:, i, :]
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.3)
    plt.xlim(0, L)
    plt.ylim(0, W)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectoires des piétons")
    plt.show()


def animate_density(D, x_edges, y_edges, L=100.0, W=7.0, normalize=False):
    """Anime la densité sur le temps."""
    steps = D.shape[0]
    fig, ax = plt.subplots(figsize=(12, 3))

    D = np.maximum(D, 0)
    vmin, vmax = 0, np.max(D)

    quad = ax.pcolormesh(x_edges, y_edges, D[0], shading='auto')
    quad.set_clim(vmin, vmax)

    plt.colorbar(quad, ax=ax, label="Densité" + (" normalisée" if normalize else ""))
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    
    def update(t):
        quad.set_array(D[t].ravel())
        ax.set_title(f"Densité - frame {t+1}/{steps}")
        return [quad]
    
    ani = FuncAnimation(fig, update, frames=steps, interval=50)
    plt.show()
    return ani


def smooth_density_gaussian(D, x_edges, y_edges, sigma_m, preserve_sum=True, truncate=4.0):
    """
    Lisse la densité D avec un noyau gaussien.
    D : array (steps, ny, nx)  -- counts ou densité par cellule
    x_edges, y_edges : bords des cellules (comme dans ton code)
    sigma_m : écart-type du noyau en mètres (rayon physique)
    preserve_sum : si True, on rescales chaque frame pour conserver la somme initiale (total de piétons)
    truncate : facteur pour limiter le noyau (idem paramètre truncate de gaussian_filter)
    
    Retour : D_smooth (même forme que D)
    """
    # tailles des cellules (assume grille régulière)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    # sigma en nombre de cellules (axe x->nx, y->ny)
    sigma_x = float(sigma_m) / dx
    sigma_y = float(sigma_m) / dy

    # gaussian_filter peut prendre sigma par axe ; D.shape = (t, ny, nx)
    # on ne filtre pas l'axe t => sigma_axis = 0 pour t
    sigma_axis = (0.0, sigma_y, sigma_x)

    # appliquer filtre (vectorisé : filtre appliqué sur axes 1 & 2 pour tous t)
    D_smooth = gaussian_filter(D, sigma=sigma_axis, mode='reflect', truncate=truncate)

    if preserve_sum:
        # évite division par 0 ; rescale par frame
        orig_sums = D.sum(axis=(1,2))
        new_sums  = D_smooth.sum(axis=(1,2))
        # pour frames où new_sum > 0, on applique un facteur de correction
        mask = new_sums > 0
        factors = np.ones_like(new_sums)
        factors[mask] = orig_sums[mask] / new_sums[mask]
        # appliquer le facteur par broadcasting
        D_smooth = D_smooth * factors[:, None, None]

    return D_smooth




if __name__ == "__main__":
    positions, velocities = run_pedestrian_simulation(N = 300, showPlot=True)

    D, x_edges, y_edges = compute_density(positions, normalize=False, nx = 200 , ny = 28)

    # lisser avec sigma = 0.5 m
    sigma_m = 1
    D_smooth = smooth_density_gaussian(D, x_edges, y_edges, sigma_m, preserve_sum=True)


    ani = animate_density(D_smooth, x_edges=x_edges, y_edges= y_edges)

    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#voir si sa marche 
from simulation_pedestrians.utils.update import update2

def run_pedestrian_simulation(L: float = 150.0,
                              W: int = 7,
                              N: int = 15,
                              v_desired: float = 1.3,
                              relax_time: float = 0.5,
                              A: float = 2.0,
                              B: float = 0.5,
                              wall_repulsion: int = 1,
                              dt: float = 0.5,
                              steps: int = 200):

    # --- Initialisation des états ---
    positions = np.zeros((N, 2))
    positions[:, 0] = np.random.uniform(0, L / 4, N)   # départ côté gauche
    positions[:, 1] = np.random.uniform(0, W, N)       # position latérale
    velocities = np.zeros((N, 2))

    All_positions = np.full((steps, N, 2), np.nan)     # NaN par défaut pour tracer proprement
    All_velocities = np.full((steps, N, 2), np.nan)

    # --- Figure / Axes ---
    fig, ax = plt.subplots(figsize=(12, 3))
    scat = ax.scatter(positions[:, 0], positions[:, 1])
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Simulation de piétons (N = {N})")

    # --- Init pour blit=True : retourne les artistes à dessiner ---
    def init():
        scat.set_offsets(positions)
        return (scat,)

    # --- Animation ---
    ani = FuncAnimation(
        fig, update2, frames=steps, init_func=init,
        interval=50, blit=True, repeat=False,
        fargs=(positions, velocities, v_desired, relax_time, A, B,
               wall_repulsion, W, L, dt, scat, steps, All_positions, All_velocities)
    )

    plt.show()

    # --- Tracé des trajectoires enregistrées ---
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, W)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Trajectoires des piétons")

    subset = np.linspace(0, N - 1, min(N, 20), dtype=int)  # max 20 traj. pour la lisibilité
    for i in subset:
        traj = All_positions[:, i, :]
        ax2.plot(traj[:, 0], traj[:, 1], alpha=0.3)

    plt.show()

    return ani

if __name__ == "__main__":
    run_pedestrian_simulation()   






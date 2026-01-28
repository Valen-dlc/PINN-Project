import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad

# Paramètre de la simulation (identique à BlackBox)
L, W = 10.0, 5.0
v_desired, relax_time = 1.3, 0.5
A, B, wall_repulsion = 20, 0.5, 5
dt, T_sim = 0.02, 3.0
steps = int(T_sim / dt)

# Simulation
def social_forces(positions, velocities):
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    v0 = np.zeros_like(velocities); v0[:, 0] = v_desired
    forces += (v0 - velocities) / relax_time
    for i in range(N):
        for j in range(i + 1, N):
            d = positions[i] - positions[j]
            dist = np.linalg.norm(d) + 1e-5
            f = A * np.exp(-dist / B) * (d / dist)
            forces[i] += f; forces[j] -= f
        y, x = positions[i, 1], positions[i, 0]
        forces[i, 1] += wall_repulsion / (y + 1e-3) - wall_repulsion / (W - y + 1e-3)
        forces[i, 0] += wall_repulsion / (x + 1e-3) - wall_repulsion / (L - x + 1e-3)
    return forces

def run_simulation_with_history(N, centered=False):
    pos = np.zeros((N, 2))
    if centered:
        pos[:, 0] = np.random.uniform(2.0, 4.0, N) # Un peu plus à gauche pour voir le mouvement
    else:
        pos[:, 0] = np.random.uniform(0.1, L - 0.1, N)
    pos[:, 1] = np.random.uniform(0.1, W - 0.1, N)
    vel = np.zeros((N, 2))
    history_x = np.zeros((steps, N))
    for s in range(steps):
        f = social_forces(pos, vel)
        vel += f * dt; pos += vel * dt
        history_x[s, :] = pos[:, 0]
    return history_x, pos, vel

# Récupération des données pour le diagramme fondamental
print("Génération des données du diagramme fondamental...")
densities, fluxes = [], []
for _ in range(100):
    N_test = np.random.randint(1, 100)
    _, final_pos, final_vel = run_simulation_with_history(N_test, centered=False)
    rho = N_test / (L * W)
    flux = rho * np.mean(final_vel[:, 0])
    densities.append(rho); fluxes.append(flux)

rho_train = torch.tensor(densities, dtype=torch.float32).view(-1, 1)
phi_train = torch.tensor(fluxes, dtype=torch.float32).view(-1, 1)

# =============================================
# 2. Réseaux de neurones (PhiNet & PINN)
# =============================================
class PhiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh(), nn.Linear(100, 1))
    def forward(self, x, t):
        return self.net(torch.cat([x/L, t/T_sim], dim=1))


phi_net = PhiNet()
pinn = PINN()

# =============================================
# 3. Entraînement
# =============================================

# Phinet (comme blackBox)
optimizer_phi = torch.optim.Adam(phi_net.parameters(), lr=0.01)
for _ in range(1000):
    optimizer_phi.zero_grad()
    pred = phi_net(rho_train)
    # Contrainte : le flux à densité nulle doit être nul
    loss = nn.MSELoss()(pred, phi_train) + 0.1 * torch.pow(phi_net(torch.zeros(1,1)), 2)
    loss.backward(); optimizer_phi.step()

# =============================================
# Entraînement PINN 
# =============================================
# PINN avec minimalisation de l'EDP 

optimizer_pinn = torch.optim.Adam(pinn.parameters(), lr=1e-3)
MSE = nn.MSELoss()

def initial_condition(x):
    return 0.4 * torch.exp(-0.5 * ((x - 2.0) / 0.3)**2)

rho_in = 0.05  # densité d'entrée (faible trafic)

print("Entraînement du PINN (EDP conservative)...")
for epoch in range(20001):
    optimizer_pinn.zero_grad()

    x_c = (torch.rand(300, 1) * L).requires_grad_(True)
    t_c = (torch.rand(300, 1) * T_sim).requires_grad_(True)

    rho = pinn(x_c, t_c)

    drho_dt = grad(rho, t_c, torch.ones_like(rho), create_graph=True)[0]
    drho_dx = grad(rho, x_c, torch.ones_like(rho), create_graph=True)[0]

    dphi_drho = grad(
        phi_net(rho),
        rho,
        torch.ones_like(rho),
        create_graph=True
    )[0]

    loss_phys = torch.mean((drho_dt + dphi_drho * drho_dx) ** 2)

    x_i = torch.rand(400, 1) * L
    loss_init = MSE(
        pinn(x_i, torch.zeros_like(x_i)),
        initial_condition(x_i)
    )

    t_in = torch.rand(200, 1) * T_sim
    x_in = torch.zeros_like(t_in)
    loss_in = MSE(
        pinn(x_in, t_in),
        rho_in * torch.ones_like(t_in)
    )

    total_loss = 5 * loss_phys + loss_init + 2 * loss_in
    total_loss.backward()
    optimizer_pinn.step()

    if epoch % 2000 == 0:
        print(f"Epoch {epoch} | Phys {loss_phys.item():.3e} | Init {loss_init.item():.3e}")


# Affichage des résultats

N_val = 15
trajs_x, _, _ = run_simulation_with_history(N_val, centered=False)
t_axis = np.linspace(0, T_sim, steps)
# Calcul des courbes de prédiction
rho_plot_range = torch.linspace(0, max(densities), 100).view(-1, 1)
with torch.no_grad():
    phi_plot_pred = phi_net(rho_plot_range).numpy()
    x_grid = torch.linspace(0, L, 100)
    t_grid = torch.linspace(0, T_sim, 100)
    X, T = torch.meshgrid(x_grid, t_grid, indexing='ij')
    rho_heatmap = pinn(X.flatten().view(-1, 1), T.flatten().view(-1, 1)).reshape(X.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Diagramme Fondamental
ax1.scatter(densities, fluxes, color="black", alpha=0.3, label="Données Simulateur")
ax1.plot(rho_plot_range.numpy(), phi_plot_pred, 'r-', linewidth=3, label="PhiNet Appris")
ax1.set_xlabel("Densité ρ (piétons/m²)"); ax1.set_ylabel("Flux Φ (piétons/m/s)")
ax1.set_title("1. Loi de Flux Apprise (PhiNet)")
ax1.legend(); ax1.grid(True)

# Panel 2: Heatmap PINN + Trajectoires
im = ax2.contourf(X.numpy(), T.numpy(), rho_heatmap.numpy(), levels=50, cmap='viridis')
plt.colorbar(im, ax=ax2, label='Densité ρ (PINN)')
#for i in range(N_val):
#    ax2.plot(trajs_x[:, i], t_axis, color='white', linewidth=0.8, alpha=0.6)
#ax2.plot([], [], color='white', label='Trajectoires microscopiques') # Pour la légende
ax2.set_xlabel("Position x (m)"); ax2.set_ylabel("Temps t (s)")
ax2.set_title("2. Propagation PINN ")
ax2.legend(); ax2.set_xlim(0, L); ax2.set_ylim(0, T_sim)

plt.tight_layout()
plt.show()


plt.show()

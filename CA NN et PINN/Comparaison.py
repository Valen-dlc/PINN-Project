import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.quasirandom import SobolEngine

# =============================================
# 1. Génération des données (Modèle Piéton BlackBox)
# =============================================

# Paramètres extraits de BlackBox.py
L, W = 10.0, 5.0
v_desired, relax_time = 1.3, 0.5
A, B, wall_repulsion = 20, 0.5, 5
dt, T_sim = 0.02, 3.0
steps = int(T_sim / dt)

# --- 2. Moteur de simulation (Social Force Model) ---
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
        # Pour la heatmap : on concentre les piétons au milieu au début
        pos[:, 0] = np.random.uniform(3.0, 5.0, N)
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

# --- 3. Génération des données pour PhiNet (Diagramme Fondamental) ---
print("Génération des données du diagramme fondamental...")
densities, fluxes = [], []
for _ in range(80):
    N_test = np.random.randint(1, 80)
    _, final_pos, final_vel = run_simulation_with_history(N_test, centered=False)
    rho = N_test / (L * W)
    flux = rho * np.mean(final_vel[:, 0])
    densities.append(rho); fluxes.append(flux)

rho_train = torch.tensor(densities, dtype=torch.float32).view(-1, 1)
phi_train = torch.tensor(fluxes, dtype=torch.float32).view(-1, 1)

# --- 4. Entraînement de PhiNet (Loi de flux) ---
class PhiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

phi_model = PhiNet()
optimizer_phi = torch.optim.Adam(phi_model.parameters(), lr=0.01)

print("Entraînement de PhiNet...")
for _ in range(1500):
    optimizer_phi.zero_grad()
    loss = nn.MSELoss()(phi_model(rho_train), phi_train)
    loss.backward(); optimizer_phi.step()

# --- 5. Entraînement du PINN (Propagation) ---
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x, t):
        return self.net(torch.cat([x/L, t/T_sim], dim=1))

pinn = PINN()
optimizer_pinn = torch.optim.Adam(pinn.parameters(), lr=0.001)

# Condition initiale (Bosse de densité)
def initial_condition(x):
    return torch.exp(-0.5 * ((x - 4.0) / 0.8)**2) * 1.0

x_col = (torch.rand(1200, 1) * L).requires_grad_(True)
t_col = (torch.rand(1200, 1) * T_sim).requires_grad_(True)

print("Entraînement du PINN...")
for epoch in range(4001):
    optimizer_pinn.zero_grad()
    rho_p = pinn(x_col, t_col)
    
    # Dérivées pour la loi de conservation : d_rho/dt + d_phi/dx = 0
    drho_dt = grad(rho_p, t_col, torch.ones_like(rho_p), create_graph=True)[0]
    phi_p = phi_model(rho_p)
    dphi_dx = grad(phi_p, x_col, torch.ones_like(phi_p), create_graph=True)[0]
    
    loss_physics = torch.mean((drho_dt + dphi_dx)**2)
    loss_init = torch.mean((pinn(x_col, torch.zeros_like(x_col)) - initial_condition(x_col))**2)
    
    total_loss = 10 * loss_physics + loss_init
    total_loss.backward(); optimizer_pinn.step()

# --- 6. Affichage Final ---
# Simulation spécifique pour les trajectoires de validation
N_val = 15
trajs_x, _, _ = run_simulation_with_history(N_val, centered=True)
t_axis = np.linspace(0, T_sim, steps)

# Calcul des courbes de prédiction
rho_plot_range = torch.linspace(0, max(densities), 100).view(-1, 1)
with torch.no_grad():
    phi_plot_pred = phi_model(rho_plot_range).numpy()
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
for i in range(N_val):
    ax2.plot(trajs_x[:, i], t_axis, color='white', linewidth=0.8, alpha=0.6)
ax2.plot([], [], color='white', label='Trajectoires microscopiques') # Pour la légende
ax2.set_xlabel("Position x (m)"); ax2.set_ylabel("Temps t (s)")
ax2.set_title("2. Propagation PINN vs Trajectoires Individuelles")
ax2.legend(); ax2.set_xlim(0, L); ax2.set_ylim(0, T_sim)

plt.tight_layout()
plt.show()
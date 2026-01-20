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

# --- Moteur de simulation (Code Prof) ---
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

def run_simulation(N):
    pos = np.zeros((N, 2))
    pos[:, 0] = np.random.uniform(0.1, L - 0.1, N)
    pos[:, 1] = np.random.uniform(0.1, W - 0.1, N)
    vel = np.zeros((N, 2))
    for _ in range(steps):
        f = social_forces(pos, vel)
        vel += f * dt; pos += vel * dt
    return pos, vel

# --- Collecte des données ---
density_flux_pairs = []
print("Génération des données (Simulation du prof)...")
for k in range(100): # N_tirages
    N = np.random.randint(1, 80)
    pos, vel = run_simulation(N)
    density = N / (L * W)
    flux = density * np.mean(vel[:, 0])
    density_flux_pairs.append((density, flux))

densities, fluxes = zip(*density_flux_pairs)
rho_train = torch.tensor(densities, dtype=torch.float32).view(-1, 1)
phi_train = torch.tensor(fluxes, dtype=torch.float32).view(-1, 1)

# =============================================
# 2. Apprentissage de la loi de flux phi(rho)
# =============================================

class PhiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

model = PhiNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Entraînement du NN...")
for epoch in range(5001):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(rho_train), phi_train)
    loss.backward(); optimizer.step()
# =============================================
# 3. Résolution de l'équation de conservation (PINN)
# =============================================

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        # Normalisation des entrées pour aider la convergence
        X = torch.cat([x/L, t/T_sim], dim=1)
        return self.net(X)

pinn = PINN()
optimizer_pinn = torch.optim.Adam(pinn.parameters(), lr=0.001)

# Condition initiale : une "foule" au milieu du couloir
def condition_initiale(x):
    # Pic de densité de 1.0 piéton/m² au centre (L/2 = 5m)
    return torch.exp(-0.5 * ((x - 5.0) / 1.5)**2) * 1.0

# Préparation des points de collocation (Sobol)
n_points = 500
sobol = SobolEngine(dimension=2, scramble=True)
points = sobol.draw(n_points)
x_phys = (points[:, 0:1] * L).requires_grad_(True)
t_phys = (points[:, 1:2] * T_sim).requires_grad_(True)

x_init = (torch.rand(200, 1) * L).requires_grad_(True)
t_init = torch.zeros_like(x_init)

def pinn_loss(rho, x, t, phi_net):
    # d_rho / d_t
    d_rho_dt = grad(rho, t, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    # d_phi(rho) / d_x
    phi = phi_net(rho)
    d_phi_dx = grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    return torch.mean((d_rho_dt + d_phi_dx)**2)

print("Entraînement du PINN...")
for epoch in range(10001):
    optimizer_pinn.zero_grad()
    
    # Perte Physique
    rho_p = pinn(x_phys, t_phys)
    loss_phys = pinn_loss(rho_p, x_phys, t_phys, model)
    
    # Perte Condition Initiale
    rho_i = pinn(x_init, t_init)
    loss_init = torch.mean((rho_i - condition_initiale(x_init))**2)
    
    total_loss = 10 * loss_phys + loss_init
    total_loss.backward()
    optimizer_pinn.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")

# =============================================
# 4. Visualisation
# =============================================

with torch.no_grad():
    # 1. Diagramme Fondamental
    rho_range = torch.linspace(0, max(densities), 100).view(-1, 1)
    phi_range = model(rho_range)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(densities, fluxes, color='black', alpha=0.5, label='Simulateur')
    plt.plot(rho_range, phi_range, 'r-', lw=2, label='Appris (PhiNet)')
    plt.xlabel('Densité (piétons/m²)'); plt.ylabel('Flux (piétons/m/s)')
    plt.legend(); plt.grid(True)

    # 2. Évolution de la densité
    x_grid = torch.linspace(0, L, 100).view(-1, 1)
    t_grid = torch.linspace(0, T_sim, 100).view(-1, 1)
    X, T = torch.meshgrid(x_grid.squeeze(), t_grid.squeeze(), indexing='ij')
    
    rho_final = pinn(X.flatten().view(-1, 1), T.flatten().view(-1, 1)).reshape(X.shape)
    
    plt.subplot(1, 2, 2)
    cp = plt.contourf(X.numpy(), T.numpy(), rho_final.numpy(), levels=50, cmap='viridis')
    plt.colorbar(cp, label='Densité rho')
    plt.xlabel('Position x (m)'); plt.ylabel('Temps t (s)')
    plt.title('Propagation de la foule (PINN)')
    plt.tight_layout(); plt.show()
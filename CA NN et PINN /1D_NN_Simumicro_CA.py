import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import grad

# Configuration
torch.manual_seed(0)
np.random.seed(0)
device = "cpu"

# ============================================================
# 1. SIMULATION AGENT-BASED (PARAMÈTRES PIÉTONS)
# ============================================================

L = 100.0          # Rue de 100 mètres
v_max_free = 1.34  # Vitesse marche normale (m/s)
T_sim = 60.0       # Temps total (s)
dt = 0.1
N_max = 150        # Densité max plus élevée pour piétons
rho_max_val = N_max / L # env 1.5 piétons/m

def update(positions, vitesses, N):
    rho = N / L
    # Loi de comportement piéton linéaire simple
    v_limit = v_max_free * (1 - (rho / rho_max_val))
    v_limit = max(v_limit, 0.05) 
    
    for i in range(N):
        # Distance au piéton devant (périodique)
        dx = (positions[(i+1)%N] - positions[i]) % L
        if dx < 1.0: # Un piéton prend environ 1m d'espace vital
            vitesses[i] = min(vitesses[i], dx / 2)
        else:
            vitesses[i] = min(vitesses[i] + 0.1, v_limit)
            
    positions += vitesses * dt
    positions %= L
    return positions, vitesses

# Listes pour stocker les données (ce qui manquait dans ton script)
rho_all, v_all, t_all, x_all = [], [], [], []

for N in range(2, N_max + 1, 2):
    positions = np.linspace(0, L, N, endpoint=False)
    vitesses = np.random.uniform(0, v_max_free, N)
    
    # On laisse la simulation se stabiliser avant de prendre des mesures
    for t_step in range(int(T_sim / dt)):
        positions, vitesses = update(positions, vitesses, N)
        if t_step > (T_sim / dt) * 0.8: # On garde les 20% derniers pas de temps
            for i in range(N):
                rho_all.append(N/L)
                v_all.append(vitesses[i])
                t_all.append(t_step * dt)
                x_all.append(positions[i])

# Conversion en tenseurs
rho_tensor = torch.tensor(rho_all, dtype=torch.float32).view(-1, 1)
v_tensor = torch.tensor(v_all, dtype=torch.float32).view(-1, 1)
# Pour le PINN
t_pinn_data = torch.tensor(t_all, dtype=torch.float32).view(-1, 1)
x_pinn_data = torch.tensor(x_all, dtype=torch.float32).view(-1, 1)

# ============================================================
# 2. SUPERVISED NN : rho -> v (APPRENTISSAGE DE LA LOI)
# ============================================================

class VelocityNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32,32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, rho):
        return self.net(rho)

vel_nn = VelocityNN().to(device)
opt_nn = torch.optim.Adam(vel_nn.parameters(), lr=1e-3)

print("Entraînement de VelocityNN...")
for epoch in range(2001):
    opt_nn.zero_grad()
    pred = vel_nn(rho_tensor)
    loss = nn.MSELoss()(pred, v_tensor)
    loss.backward()
    opt_nn.step()
    if epoch % 500 == 0:
        print(f"NN Epoch {epoch} | Loss = {loss.item():.6f}")

# ============================================================
# 3. PINN POUR rho(x,t) (RÉSOLUTION DE L'ÉQUATION)
# ============================================================

class DensityPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x, t):
        # Normalisation interne des entrées (x/L, t/T) pour aider la convergence
        return self.net(torch.cat([x/L, t/T_sim], dim=1))

rho_pinn = DensityPINN().to(device)
opt_pinn = torch.optim.Adam(rho_pinn.parameters(), lr=1e-3)

# Points de collocation pour la PDE
N_c = 2000
x_c = (torch.rand(N_c, 1) * L).requires_grad_(True)
t_c = (torch.rand(N_c, 1) * T_sim).requires_grad_(True)

print("\nEntraînement du PINN...")
for epoch in range(1001):
    opt_pinn.zero_grad()

    # 1. Perte PDE: d_rho/dt + d(rho*v)/dx = 0
    rho_pred = rho_pinn(x_c, t_c)
    v_pred = vel_nn(rho_pred) # Utilisation de la loi apprise
    flux = rho_pred * v_pred

    rho_t = grad(rho_pred, t_c, torch.ones_like(rho_pred), create_graph=True)[0]
    flux_x = grad(flux, x_c, torch.ones_like(flux), create_graph=True)[0]
    loss_pde = torch.mean((rho_t + flux_x)**2)

    # 2. Condition Limite (Flux nul ou densité faible à l'entrée x=0)
    x_bc = torch.zeros(200, 1)
    t_bc = torch.rand(200, 1) * T_sim
    rho_bc = rho_pinn(x_bc, t_bc)
    loss_bc = torch.mean(rho_bc**2)

    loss = 100 * loss_pde + loss_bc # On pondère la PDE
    loss.backward()
    opt_pinn.step()

    if epoch % 500 == 0:
        print(f"PINN Epoch {epoch} | Loss = {loss.item():.6f}")

# ============================================================
# 4. VISUALISATIONS
# ============================================================

# Comparaison Loi apprise vs Données
rho_plot = torch.linspace(0.0, rho_tensor.max(), 100).view(-1, 1)
v_nn_plot = vel_nn(rho_plot).detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].scatter(rho_all, v_all, s=1, alpha=0.1, label="Données Agents")
ax[0].plot(rho_plot.numpy(), v_nn_plot, 'r', label="Vitesse apprise $v(\\rho)$")
ax[0].set_xlabel("Densité $\\rho$ (ped/m)")
ax[0].set_ylabel("Vitesse $v$ (m/s)")
ax[0].legend()

ax[1].plot(rho_plot.numpy(), rho_plot.numpy() * v_nn_plot, 'g', linewidth=2)
ax[1].set_xlabel("Densité $\\rho$ (ped/m)")
ax[1].set_ylabel("Flux $q = \\rho \\cdot v$")
ax[1].set_title("Diagramme Fondamental Piéton")

plt.tight_layout()
plt.show()

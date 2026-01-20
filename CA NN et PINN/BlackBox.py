import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Paramètres du Professeur ---
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

# --- Ton Réseau de Neurones (S'inspirant de ton code) ---
class PhiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

model = PhiNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Entraînement du NN...")
for epoch in range(2001):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(rho_train), phi_train)
    loss.backward(); optimizer.step()

# --- Affichage identique au prof ---
rho_test = torch.linspace(0, max(densities), 100).view(-1, 1)
phi_pred = model(rho_test).detach().numpy()

# plt.figure(figsize=(8, 6))
#plt.scatter(densities, fluxes, color="black", label="Données Empiriques (simulation)")
#plt.plot(rho_test.numpy(), phi_pred, 'r-', linewidth=2, label="Diagramme fondamental appris")
#plt.xlabel("Densité moyenne (piétons/m²)")
#plt.ylabel("Flux moyen (piétons/(m·s))")
#plt.title("Comparaison : Données Empiriques vs Modèle Appris")
#plt.legend(); plt.grid(True); plt.show()
plt.figure(figsize=(8, 6))
plt.scatter(densities, fluxes, color="black", label="Données Empiriques (simulation)")
plt.xlabel("Densité moyenne (piétons/m²)")
plt.ylabel("Flux moyen (piétons/(m·s))")
plt.title("Comparaison : Données Empiriques issus de 100 simulations aléatoires")
plt.legend(); plt.grid(True); plt.show()
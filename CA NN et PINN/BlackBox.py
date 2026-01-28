import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Paramètres de la simulation 
L, W = 10.0, 5.0
v_desired, relax_time = 1.3, 0.5
A, B, wall_repulsion = 20, 0.5, 5
dt, T_sim = 0.02, 3.0
steps = int(T_sim / dt)

# Moteur de simulation basé sur le modèle des forces sociales
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

# Collecte des données d'entraînement (densité,flux)
density_flux_pairs = []
print("Génération des données (Simulation)...")
for k in range(100): # N_tirages
    N = np.random.randint(1, 120)
    pos, vel = run_simulation(N)
    density = N / (L * W)
    flux = density * np.mean(vel[:, 0])
    density_flux_pairs.append((density, flux))

densities, fluxes = zip(*density_flux_pairs)
rho_train = torch.tensor(densities, dtype=torch.float32).view(-1, 1) #On reshape pour Pytorch
phi_train = torch.tensor(fluxes, dtype=torch.float32).view(-1, 1)

# Réseau de Neurones simple (reconstruction de la fonction phi(rho))
class PhiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    def forward(self, x): return self.net(x)

model = PhiNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Entraînement avec suivi de la loss
loss_history = []  # Liste pour stocker l'évolution de l'erreur
print("Entraînement du NN...")

for epoch in range(2001):
    optimizer.zero_grad()
    outputs = model(rho_train)
    loss = criterion(outputs, phi_train)
    loss.backward()
    optimizer.step()
    
   
    loss_history.append(loss.item()) # Stockage de la loss à chaque itération
    
    if epoch % 500 == 0:
        print(f"Époch {epoch} : Loss = {loss.item():.6f}")

# --- Affichage des résultats ---

# 1. Courbe de Loss (Apprentissage)
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.yscale('log') # Echelle logarithmique pour mieux visualiser la convergence.
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Évolution de l'erreur pendant l'entraînement")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

# 2. Comparaison Données vs Modèle
rho_test = torch.linspace(0, max(densities), 100).view(-1, 1)
phi_pred = model(rho_test).detach().numpy()

plt.figure(figsize=(8, 6))
plt.scatter(densities, fluxes, color="black", alpha=0.5, label="Données de simulation")
plt.plot(rho_test.numpy(), phi_pred, 'r-', linewidth=3, label="Modèle (Neural Net)")
plt.xlabel("Densité moyenne (piétons/m²)")
plt.ylabel("Flux moyen (piétons/(m·s))")
plt.title("Diagramme Fondamental Appris")
plt.legend()
plt.grid(True)
plt.show()
plt.scatter(densities, fluxes, color="black", label="Données Empiriques (simulation)")
plt.xlabel("Densité moyenne (piétons/m²)")
plt.ylabel("Flux moyen (piétons/(m·s))")
plt.title("Comparaison : Données Empiriques issus de 100 simulations aléatoires")

plt.legend(); plt.grid(True); plt.show()

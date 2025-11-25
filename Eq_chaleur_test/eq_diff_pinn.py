import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


# Paramètres physiques
L = 1.0  # Longueur de la barre (m)
lambda_ = 237  # Conductivité thermique (W/m·K)
c = 900  # Capacité thermique massique (J/kg·K)
rho = 2700  # Masse volumique (kg/m³)
alpha = lambda_ / (c * rho)  # Diffusivité thermique (m²/s)
alpha = 0.01

# Domaine numérique
Nx = 200
Nt = 200
dx = L / (Nx-1)
# dt = 0.8 * dx**2 / (2 * alpha)
tf = 10
dt = Nt/tf

# Réseau de neurones
class FCN(nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.layers = nn.ModuleList()
		for i in range(len(layers)-1):
			self.layers.append(nn.Linear(layers[i], layers[i+1]))
		self.activation = nn.Tanh()
	def forward(self, x, t):
		X = torch.cat([x, t/tf], dim=1)
		for i, layer in enumerate(self.layers[:-1]):
			X = self.activation(layer(X))
		return self.layers[-1](X)

# Fonctions utilitaires
def initial_condition(x):
	return torch.where(x >= 0.5, 1.0, 0.0)

# Points d'entraînement
N_f = 200  # points pour la physique
N_ic = Nx+1  # points pour la condition initiale
N_bc = 2  # points pour les bords

# Grille pour affichage
x_np = np.linspace(0, L, Nx+1)
t0 = 0.0
t1 = tf/4
t2 = tf/2
t3 = tf

# PINN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
layers = [2, 64, 64, 64, 1]
model = FCN(layers).to(device)


# Optimiseur et scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# Réduit le lr d'un facteur 0.1 après 80% des epochs
epochs = 5000
# step_size = int(epochs * 0.8)
# scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

# Points pour la physique (collocation)
xf = torch.FloatTensor(np.random.uniform(0, L, (N_f, 1))).to(device)
tf_ = torch.FloatTensor(np.random.uniform(0, tf, (N_f, 1))).to(device)

# Points pour la condition initiale
x_ic = torch.FloatTensor(x_np.reshape(-1,1)).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
T_ic = initial_condition(x_ic).to(device)

# Points pour les bords
t_bc = torch.FloatTensor(np.linspace(0, tf, N_bc).reshape(-1,1)).to(device)
x0_bc = torch.zeros_like(t_bc).to(device)
x1_bc = torch.full_like(t_bc, L).to(device)

# Entraînement
def pinn_loss():
	# Condition initiale
	T_pred_ic = model(x_ic, t_ic)
	loss_ic = nn.MSELoss()(T_pred_ic, T_ic)
	# Bords
	T0 = model(x0_bc, t_bc)
	T1 = model(x1_bc, t_bc)
	loss_bc = nn.MSELoss()(T0, torch.zeros_like(T0)) + nn.MSELoss()(T1, torch.full_like(T1, 1.0))
	# Physique (équation de la chaleur)
	xf_ = xf.clone().detach().requires_grad_(True)
	tf__ = tf_.clone().detach().requires_grad_(True)
	T_f = model(xf_, tf__)
	T_f_x = torch.autograd.grad(T_f, xf_, grad_outputs=torch.ones_like(T_f), create_graph=True)[0]
	T_f_xx = torch.autograd.grad(T_f_x, xf_, grad_outputs=torch.ones_like(T_f_x), create_graph=True)[0]
	T_f_t = torch.autograd.grad(T_f, tf__, grad_outputs=torch.ones_like(T_f), create_graph=True)[0]
	eq = T_f_t - alpha * T_f_xx
	loss_f = torch.mean(eq**2)
	return loss_ic + loss_bc + loss_f



# Suivi de la loss
loss_history = []
for epoch in range(epochs):
	optimizer.zero_grad()
	loss = pinn_loss()
	loss.backward()
	optimizer.step()
	# scheduler.step()
	loss_history.append(loss.item())
	if epoch % 10 == 0:
		lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {epoch}, Loss: {loss.item():.6f}, lr: {lr:.2e}")

# Sauvegarde automatique des poids après entraînement
torch.save(model.state_dict(), "pinn_weights.pth")
print("Poids du modèle sauvegardés dans pinn_weights.pth")

# Affichage de la courbe de loss
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss totale")
plt.title("Évolution de la loss (PINN)")
plt.grid()

# Prédiction pour affichage
def predict_temps(t_val):
	x_t = torch.FloatTensor(x_np.reshape(-1,1)).to(device)
	t_t = torch.full_like(x_t, t_val).to(device)
	with torch.no_grad():
		T_pred = model(x_t, t_t).cpu().numpy().squeeze()
	return T_pred

results = {
	"t0": predict_temps(t0),
	"tf/4": predict_temps(t1),
	"tf/2": predict_temps(t2),
	"tf": predict_temps(t3)
}

# Affichage
plt.figure()
plt.plot(x_np, results["t0"], label="t = 0 s")
plt.plot(x_np, results["tf/4"], label=f"t = {t1:.1f} s")
plt.plot(x_np, results["tf/2"], label=f"t = {t2:.1f} s")
plt.plot(x_np, results["tf"], label=f"t = {t3:.1f} s")
plt.xlabel("Position x (m)")
plt.ylabel("Température (°C)")
plt.title("Évolution de la température (PINN)")
plt.legend(fancybox=False, edgecolor="black")
plt.grid()
plt.show()

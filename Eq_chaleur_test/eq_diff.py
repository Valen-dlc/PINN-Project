import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
L = 1.0  # Longueur totale de la barre (m)
lambda_ = 237  # Conductivité thermique de l'aluminium (W/m·K)
c = 900  # Capacité thermique massique de l'aluminium (J/kg·K)
rho = 2700  # Masse volumique de l'aluminium (kg/m³)

# Calcul du coefficient alpha
alpha = lambda_ / (c * rho)  # Diffusivité thermique (m²/s)

alpha = 0.01

# Paramètres numériques
Nx = 200  # Nombre de points de discrétisation spatiale
Nt = 5000 # Nombre de pas de temps
dx = L / (Nx-1)  # Pas spatial
dt = 0.8 * dx**2 / (2 * alpha)  # Pas temporel respectant la condition CFL


x = np.linspace(0, L, Nx+1)


# Initialisation des conditions initiales
T = np.zeros(Nx+1)  # Température initiale (0°C partout)
T[x>=0.5] = 40  # La moitié droite est à 40°C

# Ajout des conditions aux bords
T[0] = 0  # Condition de Dirichlet à x = 0
T[-1] = 40  # Condition de Dirichlet à x = 1

# Préparation pour la méthode des différences finies
T_new = np.zeros_like(T)

# Ajout d'une liste pour stocker les températures à différents instants
results = {"t0": T.copy(), "tf/4": None, "tf/2": None, "tf": None}

# Boucle temporelle avec conditions aux bords
for n in range(Nt):
    for i in range(1, Nx):  # On ignore les bords (i=0 et i=Nx)
        T_new[i] = T[i] + alpha * dt / dx**2 * (T[i-1] - 2*T[i] + T[i+1])
    T_new[0] = 0  # Condition de Dirichlet à x = 0
    T_new[-1] = 40  # Condition de Dirichlet à x = 1
    T[:] = T_new  # Mise à jour de la température

    # Enregistrement des résultats aux instants spécifiés
    if n == int(Nt / 4):
        results["tf/4"] = T.copy()
    elif n == int(Nt / 2):
        results["tf/2"] = T.copy()
    elif n == Nt - 1:
        results["tf"] = T.copy()

# Affichage des résultats
plt.plot(x, results["t0"], label="t = 0 s")
plt.plot(x, results["tf/4"], label=f"t = {Nt * dt / 4:.1f} s")
plt.plot(x, results["tf/2"], label=f"t = {Nt * dt / 2:.1f} s")
plt.plot(x, results["tf"], label=f"t = {Nt * dt:.1f} s")

plt.xlabel("Position x (m)")
plt.ylabel("Température (°C)")
plt.title("Évolution de la température dans la barre")
plt.legend(fancybox=False, edgecolor="black")
plt.grid()
plt.show()


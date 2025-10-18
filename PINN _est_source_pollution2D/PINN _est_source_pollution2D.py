#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial import KDTree


#% In[87]:


# Simulation par différences finies spatiales (méthode des lignes) avec conditions aux limites de Dirichlet
# Simulation 2D de la diffusion d'un nuage de pollution avec source gaussienne
# Projet ASI 2024-2025
def calculer_vecteur_coordonnees_2D(Lx, Ly, Nx, Ny):
    """
    Calcule un vecteur des coordonnées cartésiennes 2D pour une grille de dimensions Nx x Ny
    dans un rectangle de dimensions Lx x Ly.
    
    Args:
        Lx (float): Longueur du rectangle sur l'axe x.
        Ly (float): Longueur du rectangle sur l'axe y.
        Nx (int): Nombre de points sur la grille le long de l'axe x.
        Ny (int): Nombre de points sur la grille le long de l'axe y.
    
    Returns:
        coordonnees (numpy.ndarray): Un vecteur de dimension (Nx * Ny, 2) contenant les
                                     coordonnées cartésiennes (x, y) pour chaque point de la grille.
    """
    # Créer les points de grille dans les directions x et y
    x = np.linspace(0, Lx, Nx)  # Coordonnées x
    y = np.linspace(0, Ly, Ny)  # Coordonnées y
    
    # Créer une grille 2D des coordonnées
    X, Y = np.meshgrid(x, y)  # Grilles 2D
    
    # Aplatir les matrices X et Y pour obtenir les coordonnées sous forme de vecteurs
    X_flat = X.ravel()  # Coordonnées x aplaties
    Y_flat = Y.ravel()  # Coordonnées y aplaties
    
    # Combiner les coordonnées x et y dans un vecteur de taille (Nx * Ny, 2)
    coordonnees = np.vstack((X_flat, Y_flat)).T
    
    return coordonnees

# Paramètres physiques
K = 20  # Coefficients de diffusion (m^2/s) (ne pas changer)
Ux = 2  # Vitesses du vent en m/s  (uniformes et constantes A MODIFIER)
Uy = -1

# Tailles du maillage en x, y (11 <= Nx, Ny <= 41)
Nx = 101
Ny = Nx
# Dimensions du domaine (en m)
Lx = 2000
Ly = 2000
hx = Lx / (Nx - 1)
hy = Ly / (Ny - 1)

# Vecteur des coordonnées cartésiennes des points de la grille
X = calculer_vecteur_coordonnees_2D(Lx, Ly, Nx, Ny)

# Amplitude de la source (unité non précisée : dépend de la nature de la source !)
S = 10

# Localisation de la source (en m)
xs = 500
ys = 1000
sigma = 50  # dispersion de la source

# Pas de temps (période d'échantillonnage)
T = 120  # en secondes (10 <= T <= 120)

N = 5  # nombre de pas de temps : N*T = durée de la simulation

tmax = T*N
temps = np.linspace(0, tmax, N+1)

# Conditions Initiales (pas de pollution)
x0 = np.zeros(Nx * Ny)


#%% 

def simu():
    # Simulation temporelle par Différences Finies

    # Utilisation d'ODE pour la simulation
    sol = solve_ivp(
        lambda t, x: dynPDEd(t, x, S, Ux, Uy, K, hx, hy, Nx, Ny, xs, ys, sigma),
        [0, N * T], x0, t_eval=temps
    )

    xe = sol.y[:, -1] #♥ solution en fin de simulation
    
    # Tracé de la solution en fin de simulation par interpolation RBF sur la grille
    
    Zi = graph_interpol(3, Lx, Ly, 100, 'SPH', 1, 0, 0, xe, X, 15, 'Pollutant concentration')
    
    return sol.t, sol.y


#%% Fonction pour calculer la dérivée temporelle de la distribution de pollution
# Méthode des différences finies

def dynPDEd(t, x, S, Ux, Uy, K, hx, hy, Nx, Ny, xs, ys, sigma):
    """
    Fonction de dynamique pour l'équation aux dérivées partielles d'advection-diffusion avec source gaussienne.
    
    Args:
        t (float): Temps.
        x (numpy.ndarray): Vecteur des valeurs de la solution sur la grille (dimension Nx*Ny).
        S (float): Amplitude de la source.
        Ux (float): Vitesse de l'advection dans la direction x.
        Uy (float): Vitesse de l'advection dans la direction y.
        K (float): Coefficient de diffusion.
        hx (float): Pas de la grille dans la direction x.
        hy (float): Pas de la grille dans la direction y.
        Nx (int): Nombre de points de grille dans la direction x.
        Ny (int): Nombre de points de grille dans la direction y.
        xs (float): Position x de la source.
        ys (float): Position y de la source.
        sigma (float): Dispersion de la source gaussienne.
    
    Returns:
        numpy.ndarray: Vecteur F des valeurs de la dynamique sur la grille (dimension Nx*Ny).
    """
    F = np.zeros(Nx * Ny)
    
    for i in range(Nx):
        for j in range(Ny):
            zij = x[i + j * Nx]
            
            # Conditions aux limites pour zim1j, zip1j, zijm1, zijp1 (points fantômes)
            if i == 0:
                zim1j = 0  # Condition de Dirichlet à gauche
            else:
                zim1j = x[(i - 1) + j * Nx]
                
            if i == Nx - 1:
                zip1j = 0  # Condition de Dirichlet à droite
            else:
                zip1j = x[(i + 1) + j * Nx]
                
            if j == 0:
                zijm1 = 0  # Condition de Dirichlet en bas
            else:
                zijm1 = x[i + (j - 1) * Nx]
                
            if j == Ny - 1:
                zijp1 = 0  # Condition de Dirichlet en haut
            else:
                zijp1 = x[i + (j + 1) * Nx]
            
            # Termes de la source sous forme gaussienne
            ci = (i) * hx
            cj = (j) * hy
            deltaij = np.exp(-((ci - xs)**2 + (cj - ys)**2) / (2 * sigma**2))
            
            # Schéma Upwind pour l'advection
            if Ux > 0:
                F1 = -Ux * (zij - zim1j) / hx
            else:
                F1 = -Ux * (zip1j - zij) / hx
                
            if Uy > 0:
                F2 = -Uy * (zij - zijm1) / hy
            else:
                F2 = -Uy * (zijp1 - zij) / hy
                
            # Diffusion et source gaussienne
            F[i + j * Nx] = (F1 + F2 +
                             K * (zip1j - 2 * zij + zim1j) / hx**2 +
                             K * (zijp1 - 2 * zij + zijm1) / hy**2 +
                             deltaij * S)
    
    return F# def dynPDEd(t, x, S, Ux, Uy, K, hx, hy, Nx, Ny, xs, ys, sigma):
            
def base_e(x, xi, r, epsil, beta, fun='SPH') :
    # Calcul de base RBF
    # UTILISER ICI r=1, epsil=0, beta=0 POUR SPH
    R = np.sum((x - xi) ** 2, axis=1)
    #Spline Polyharmonic
    if fun == 'SPH':
        phi = R**(r/2)
    #RBF gaussienne
    elif fun == 'Gaussian':
        phi = np.exp(-epsil**2*R)
    #MQ or IMQ
    elif fun == 'IMQ':
        phi = (1+epsil**2*R)**beta
    return phi

def noyau(x,r,epsil,beta,fun):
    return np.array([base_e(x_i,x,r,epsil,beta,fun) for x_i in x])
            

# Fonction pour visualiser les résultats après interpolation RBF

def graph_interpol(fn, Lx, Ly, Mg, fun, r, epsil, beta, Z, X, N, titre):

    # Interpolation de solutions basée sur N plus proches voisins des points de la grille (stencil)
    # par algorithme KD tree
    # N = taille des stencils d'approximation (Nombre de voisins)
    # Z = données à interpoler sur une grille de taille MgxMg
    # X = vecteur des coordonnées cartésiennes des points de la grille
    # Mg = nombre de points de grille du graphe (taille de la grille =Mg * Mg)

    Cx = np.linspace(0, Lx, Mg)
    Cy = np.linspace(0, Ly, Mg)
    Zi = np.zeros((Mg, Mg))

    # Initialize kD-tree search
    Mdl = KDTree(X)

    for i in range(Mg):
        for j in range(Mg):
            x = np.array([Cx[i], Cy[j]])
            # kD-Tree search for each point
            d, Ind = Mdl.query(x, k=N)
            # Calcul des opérateurs pour chaque point
            stencil = X[Ind, :]
            K = noyau(stencil, r, epsil, beta, fun) # SANS POLY
            phi = base_e(x, stencil, r, epsil, beta, fun)
            Zi[i, j] = np.dot(phi, np.linalg.inv(K)) @ Z[Ind]

    plt.figure(fn)
    s = plt.imshow(Zi.T, extent=[0, Lx, 0, Ly], origin='lower')
    plt.colorbar(s)
    plt.xlabel(r'$x (m)$')
    plt.ylabel(r'$y (m)$')
    plt.title(titre)
    plt.grid(False)
    plt.show()
    
    return Zi

# Fonction RBF pour interpolation linéaire d'ordre N en 2D

def interpol(fun, r, epsil, beta, Z, X, N, theta):
    # Interpolation de solutions basée sur N plus proches voisins des points de la grille (stencil)
    # par algorithme KD tree
    # N = taille des stencils d'approximation (Nombre de voisins)
    # Z = données à interpoler sur une grille de taille MgxMg
    # X = vecteur des coordonnées cartésiennes des points de la grille
    # theta = coordonnées cartésiennes du point dont la concentration doit être interpolée à partir de X
    
    # sorties : Z_theta = valeur au point de coordonnées \theta, C_theta = opérateur de sortie correspondant
    # tel que Z_theta = C_theta Z

    # Initialize kD-tree search
    Mdl = KDTree(X)

    x = np.array([theta[0], theta[1]])
    # kD-Tree search for each point
    d, Ind = Mdl.query(x, k=N)
    # Calcul du stencil
    stencil = X[Ind, :]
    K = noyau(stencil, r, epsil, beta, fun) 
    phi = base_e(x, stencil, r, epsil, beta, fun) # Calcul des N bases du stencil
    C = np.dot(phi, np.linalg.inv(K))
    Z_theta = C @ Z[Ind]
    C_theta =np.zeros((len(Z)))
    C_theta[Ind] = C
    
    return Z_theta, C_theta.reshape(1,len(Z))

def source_def(Nx, Ny, hx, hy, xs, ys, sigma, S):
    source = np.zeros(Nx * Ny)
    for i in range(Nx):
        for j in range(Ny):                       
            # Termes de la source sous forme gaussienne
            ci = (i) * hx
            cj = (j) * hy
            deltaij = np.exp(-((ci - xs)**2 + (cj - ys)**2) / (2 * sigma**2))
                            
            # Diffusion et source gaussienne
            source[i + j * Nx] = deltaij * S
    
    return source

Nn = 50
# Neural network to estimate En (pollutant concentration)
class ZnNetwork(nn.Module):
    def __init__(self):
        super(ZnNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, Nn),
            nn.Tanh(),
            nn.Linear(Nn, Nn),
            nn.Tanh(),
            nn.Linear(Nn, Nn),
            nn.Tanh(),
            nn.Linear(Nn, 1)
        )
    
    def forward(self, t, x, y):
        inputs = torch.cat([t/tmax, x/Lx, y/Ly], dim=1)
        return self.fc(inputs)

class SnNetwork(nn.Module):
    def __init__(self):
        super(SnNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, Nn),
            nn.Tanh(),
            nn.Linear(Nn, Nn),
            nn.Tanh(),
            nn.Linear(Nn, Nn),
            nn.Tanh(),
            nn.Linear(Nn, 1)
        )
    
    def forward(self, x, y):
        inputs = torch.cat([x/Lx, y/Ly], dim=1)
        return self.fc(inputs)

# Loss function for the governing differential equation (PDE)
def physics_loss(zn_model, sn_model, x, y, t, u_exact, params):
    # Enable gradient tracking
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    # Constants and parameters
    Kr, Mr, Kb, Mb, Ki, Mi, Km, Mm, alpha_r, beta = params
    Vx, Vy, K = 1, 1, 30  # advection and diffusion constants

    # Predict concentration z using the Zn model
    z_pred = zn_model(t, x, y)

    # Gradients with respect to x, y, t
    z_t = torch.autograd.grad(z_pred, t, grad_outputs=torch.ones_like(z_pred), create_graph=True)[0]
    z_x = torch.autograd.grad(z_pred, x, grad_outputs=torch.ones_like(z_pred), create_graph=True)[0]
    z_y = torch.autograd.grad(z_pred, y, grad_outputs=torch.ones_like(z_pred), create_graph=True)[0]

    # Second-order gradients with respect to x and y
    z_xx = torch.autograd.grad(z_x, x, grad_outputs=torch.ones_like(z_x), create_graph=True)[0]
    z_yy = torch.autograd.grad(z_y, y, grad_outputs=torch.ones_like(z_y), create_graph=True)[0]

    # Source S predicted by the Sn model
    s_pred = sn_model(x, y)

    # Interior loss: Advection-diffusion PDE
    pde_interior = (
        z_t + Vx * z_x + Vy * z_y 
        - K * (z_xx + z_yy) 
        - s_pred
    )
    interior_loss = Kr * torch.mean(pde_interior**2) 

    # Boundary losses 
    bc_left = zn_model(t, torch.zeros_like(x), y)
    boundary_loss_left = Kb * torch.mean(bc_left**2)
    
    bc_right = zn_model(t, torch.ones_like(x)*Lx, y)    
    boundary_loss_right = Kb * torch.mean(bc_right**2)
    
    bc_bottom = zn_model(t, x, torch.zeros_like(y)) 
    boundary_loss_bottom = Kb * torch.mean(bc_bottom**2)
    
    bc_top = zn_model(t, x, torch.ones_like(y)*Ly) 
    boundary_loss_top = Kb * torch.mean(bc_top**2)
    

    # Initial condition loss 
    z0_pred = zn_model(torch.zeros_like(t), x, y)
    initial_loss = Ki  * torch.mean((z0_pred)**2)

    # Measurement data loss 
    measurement_loss = 0
    Nt = len(t_simu)
    for k in range(Nt):
        measurement_pred = zn_model((torch.ones_like(theta_c[:,0])*t_simu[k]).reshape(-1,1), theta_c[:,0].reshape(-1,1), theta_c[:,1].reshape(-1,1))
        measurement_loss += Km * torch.mean((measurement_pred - y_mes[:,k].reshape(-1,1))**2)/Nt

    # Source regularization avec pénalité S>0
    zer = torch.zeros_like(s_pred)
    source_loss = alpha_r  * torch.mean(torch.min(zer,s_pred)**2)

    # Total loss
    total_loss = (
        interior_loss 
        + boundary_loss_left
        + boundary_loss_right
        + boundary_loss_bottom
        + boundary_loss_top
        + initial_loss 
        + measurement_loss 
        + source_loss 
    )

    return total_loss

# Import your model classes
# from your_model import ZnNetwork, SnNetwork, physics_loss

# Function to generate synthetic test data
def generate_synthetic_data(num_samples=256, tmax=600, Lx=2000, Ly=2000):
    # Initialiser le moteur Sobol pour 3 dimensions (pour générer t, x et y)
    sobol_engine = torch.quasirandom.SobolEngine(dimension=3, scramble=False)
    # Générer n points
    n = num_samples
    # nombre de points que vous souhaitez générer
    points = sobol_engine.draw(n)
    x=points[:, 0] * Lx
    y=points[:, 1] * Ly
    t=points[:, 2] * tmax

    return t, x, y

# Updated training algorithm
# Constants (example values; replace with actual ones)
Mp = 1024 
Kr, Mr = 1, Mp
Kb, Mb = 1, Mp
Ki, Mi = 0, Mp # on retrouve les conditions intiales et la source !
Km, Mm = 1, Mp
alpha_r, beta = 10, 0

params = (Kr, Mr, Kb, Mb, Ki, Mi, Km, Mm, alpha_r, beta)

#%% Initializing the models
zn_model = ZnNetwork()
sn_model = SnNetwork()

# Generate training data
t_train, x_train, y_train = generate_synthetic_data(Mp, tmax, Lx, Ly)
t_simu, u_exact = simu()
source =  source_def(Nx, Ny, hx, hy, xs, ys, sigma, S)

#%%
Si = graph_interpol(
    fn=4,            # Figure number
    Lx=Lx,    # Maximum dimension in x
    Ly=Ly,    # Maximum dimension in y
    Mg=100,          # Number of grid points
    fun='SPH',       # Kernel function (adjust if necessary)
    r=1,             # Parameter r
    epsil=0,         # Parameter epsilon
    beta=0,          # Parameter beta
    Z=source,        # Predicted Z values from the model
    X=X,             # Cartesian coordinates (x, y)
    N=15,            # Number of neighbors for KDTree
    titre='Source'  # Graph title
)
#%%

t_train = t_train.reshape(-1,1)
x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
u_exact = torch.tensor(u_exact, dtype=torch.float32)

# Générer Ns coordonnées de capteurs - Suite de Sobol'
Ns = 512
theta_c = torch.zeros((Ns,2), dtype=torch.float32 )
# Sobol'
sobol_engine = torch.quasirandom.SobolEngine(dimension=2, scramble=False)
points = sobol_engine.draw(Ns)
theta_c[:,0] = points[:,0] * Lx
theta_c[:,1] = points[:,1] * Ly

Nt = len(t_simu)
y_mes = torch.zeros((Ns,Nt), dtype=torch.float32)
for t in range(Nt):
    for l in range(Ns):
        y_mes[l,t], C_theta = interpol('SPH', 1, 0, 0, u_exact[:,t].numpy(), X, 15, theta_c[l,:])

# Capteurs sur grille DF
# theta_c  = torch.tensor(X, dtype=torch.float32)
# y_mes =u_exact
#%% Combine the parameters of both networks
params_network = list(zn_model.parameters()) + list(sn_model.parameters())

# Optimizer
optimizer = optim.Adam(params_network, lr=0.01)

# Iterative training algorithm
num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    total_loss = physics_loss(zn_model, sn_model, x_train, y_train, t_train, u_exact, params)

    # Backpropagation
    total_loss.backward()
#    torch.nn.utils.clip_grad_norm_(params_network, max_norm=1.0)  # Gradient clipping
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Total Loss: {total_loss.item()}")

#%% Définition de l'optimiseur LBFGS
optimizer = torch.optim.LBFGS(params_network, lr=0.1, max_iter=3000, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn='strong_wolfe')

def closure():
    optimizer.zero_grad()
    total_loss = physics_loss(zn_model, sn_model, x_train, y_train, t_train, u_exact, params)
    total_loss.backward()
    return total_loss

optimizer.step(closure)
print(f'Loss: {closure():.4f}') 

# Combine coordinates (x, y) for the interpolation graph
Xe = torch.cat([x_train, y_train], dim=1).detach().numpy() 

#%% Generate interpolated graph

z_pred = zn_model((torch.ones_like(x_train)*tmax).reshape(-1,1), x_train, y_train).detach().numpy()
s_pred = sn_model(x_train, y_train).detach().numpy()

Zi_train = graph_interpol(
    fn=1,            # Figure number
    Lx=Lx,    # Maximum dimension in x
    Ly=Ly,    # Maximum dimension in y
    Mg=100,          # Number of grid points
    fun='SPH',       # Kernel function (adjust if necessary)
    r=1,             # Parameter r
    epsil=0,         # Parameter epsilon
    beta=0,          # Parameter beta
    Z=z_pred,        # Predicted Z values from the model
    X=Xe,             # Cartesian coordinates (x, y)
    N=15,            # Number of neighbors for KDTree
    titre='Estimated pollutant'  # Graph title
)

#%% Generate interpolated graph
Si_train = graph_interpol(
    fn=2,            # Figure number
    Lx=Lx,    # Maximum dimension in x
    Ly=Ly,    # Maximum dimension in y
    Mg=100,          # Number of grid points
    fun='SPH',       # Kernel function (adjust if necessary)
    r=1,             # Parameter r
    epsil=0,         # Parameter epsilon
    beta=0,          # Parameter beta
    Z=s_pred,        # Predicted Z values from the model
    X=Xe,             # Cartesian coordinates (x, y)
    N=15,            # Number of neighbors for KDTree
    titre='Estimated source'  # Graph title
)


import numpy as np
import matplotlib.pyplot as plt



# ============================================================
# 1) CALCUL DU FLUX À PARTIR DE (densité, vitesses)
# ============================================================
def compute_flux(D, U, V):
    """
    Calcule le flux macroscopique :
      φx = ρ * U
      φy = ρ * V
      ||φ|| = sqrt(φx² + φy²)

    D : densité (T, ny, nx)
    U : vitesse moyenne en x (T, ny, nx)
    V : vitesse moyenne en y (T, ny, nx)

    Retourne :
      phi_x, phi_y, phi_norm
    """
    phi_x = D * U
    phi_y = D * V
    phi_norm = np.sqrt(phi_x**2 + phi_y**2)

    return phi_x, phi_y, phi_norm



# ============================================================
# 2) DIAGRAMME FONDAMENTAL (scatter flux vs densité)
# ============================================================
def plot_fundamental_diagram(D, phi_norm, title="Diagramme fondamental (flux–densité)"):
    """
    Trace le diagramme fondamental Φ = f(ρ).

    D : densité (T, ny, nx)
    phi_norm : norme du flux (T, ny, nx)
    """
    # Aplatit toutes les cellules de tous les instants
    rho = D.flatten()
    phi = phi_norm.flatten()

    # Filtrage des NaN + densités strictement positives
    mask = np.isfinite(rho) & np.isfinite(phi) & (rho > 0)
    rho = rho[mask]
    phi = phi[mask]

    plt.figure(figsize=(8,5))
    plt.scatter(rho, phi, s=5, alpha=0.25)
    plt.xlabel("Densité ρ (piétons / m²)")
    plt.ylabel("Flux ||φ|| (piétons / m·s)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return rho, phi



# ============================================================
# 3) PIPELINE COMPLET (tout calcul + scatter)
# ============================================================
def compute_and_plot_fundamental(D, All_positions, All_velocities,
                                 compute_velocity_field, 
                                 nx=100, ny=14, L=100.0, W=7.0):
    """
    Pipeline complet :

    1. calcule les vitesses moyennes U,V (par grille)
    2. calcule les flux φ
    3. trace le diagramme fondamental ρ → Φ

    Paramètres :
      D : densité (T, ny, nx)
      All_positions : positions microscopiques (T, N, 2)
      All_velocities : vitesses microscopiques (T, N, 2)
      compute_velocity_field : fonction utilisateur (T,N,2)->(U,V)
    
    Retourne :
      rho, phi, U, V, phi_norm
    """

    # --- champ de vitesses moyennes
    U, V = compute_velocity_field(All_positions, All_velocities,
                                  L=L, W=W, nx=nx, ny=ny)

    # --- flux macroscopique
    phi_x, phi_y, phi_norm = compute_flux(D, U, V)

    # --- diagramme fondamental
    rho, phi = plot_fundamental_diagram(D, phi_norm)

    return rho, phi, U, V, phi_norm

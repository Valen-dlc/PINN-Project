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
def plot_fundamental_diagram(D, phi_x, title="Diagramme fondamental (flux–densité)"):
    """
    Trace le diagramme fondamental Φ = f(ρ).

    D : densité (T, ny, nx)
    phi_norm : norme du flux (T, ny, nx)
    """
    # Aplatit toutes les cellules de tous les instants
    rho = D.flatten()
    phi = phi_x.flatten()

    # Filtrage des NaN + densités strictement positives
    mask = np.isfinite(rho) & np.isfinite(phi) & (rho > 0.01)
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


def compute_global_fundamental_point(All_positions, All_velocities,
                                     L=100.0, W=7.0):
    """
    Calcule un seul point du diagramme fondamental :
      - densité moyenne globale sur toute la rue
      - flux moyen global (direction x), via φ = ρ * U_moy

    All_positions : array-like de shape (T, N_t, 2)
    All_velocities : array-like de shape (T, N_t, 2)
    L : longueur de la rue
    W : largeur de la rue

    Retourne :
      rho_global, phi_global
    """

    T = len(All_positions)

    densities = []
    fluxes = []

    area = L * W

    for t in range(T):
        pos_t = All_positions[t]
        vel_t = All_velocities[t]

        # nombre de piétons à t (on ne compte que ceux avec une vitesse finie)
        vx_t = vel_t[:, 0]
        mask = np.isfinite(vx_t)
        N_t = np.sum(mask)

        if N_t == 0:
            # aucun piéton "valide" à cet instant -> on saute ce t
            continue

        # ----- 1) Densité globale à l'instant t -----
        rho_t = N_t / area
        densities.append(rho_t)

        # ----- 2) Vitesse moyenne longitudinale (x) -----
        u_mean_t = np.mean(vx_t[mask])

        # Flux global : φ(t) = ρ(t) * U_moy(t)
        phi_t = rho_t * u_mean_t
        fluxes.append(phi_t)

    if len(densities) == 0:
        raise ValueError("Aucune densité valide n'a pu être calculée (densities vide).")

    if len(fluxes) == 0:
        raise ValueError("Aucun flux valide n'a pu être calculé (fluxes vide).")

    # ----- 3) Moyennes temporelles -----
    rho_global = np.mean(densities)
    phi_global = np.mean(fluxes)

    return rho_global, phi_global



def compute_local_fundamental_point(All_positions,
                                    All_velocities,
                                    calcul_density,
                                    compute_velocity_field,
                                    L=100.0,
                                    W=7.0,
                                    nx=100,
                                    ny=14,
                                    rho_min=0.1):
    """
    Équivalent LOCAL de compute_global_fundamental_point,
    mais physiquement correct (flux longitudinal local).

    Retourne :
      rho, phi  (vecteurs prêts à être scatter)
    """

    # 1) Densité locale
    D = calcul_density(
        All_positions,
        L=L,
        W=W,
        nx=nx,
        ny=ny
    )

    # 2) Vitesse moyenne locale
    U, V = compute_velocity_field(
        All_positions,
        All_velocities,
        L=L,
        W=W,
        nx=nx,
        ny=ny
    )

    # 3) Flux longitudinal local
    phi_x = D * U

    # 4) Aplatissement + filtrage
    rho = D.flatten()
    phi = np.abs(phi_x).flatten()

    mask = np.isfinite(rho) & np.isfinite(phi) & (rho > rho_min)
    rho = rho[mask]
    phi = phi[mask]

    return rho, phi
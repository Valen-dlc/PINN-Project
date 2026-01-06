import numpy as np
import matplotlib.pyplot as plt
from pinn_model import SimplePINN
from pinn_mass_model import MassFlowPINN

def generate_realistic_data():
    """
    Génère des données où la vitesse dépend de la densité 
    (Modèle de Greenshields) pour voir le diagramme fondamental. 
    Génération de données =! de simulation piétons.
    """
    # 1. Grille espace-temps (Rue de 10m de long, 2m de large)
    x = np.linspace(0, 10, 30)
    y = np.linspace(0, 2, 10)
    t = np.linspace(0, 2, 10)
    X, Y, T = np.meshgrid(x, y, t)
    
    x_f, y_f, t_f = X.flatten(), Y.flatten(), T.flatten()
    xyt = np.stack([x_f, y_f, t_f], axis=1)

    # 2. Simulation d'une densité variable (un groupe qui s'étale)
    # rho varie entre 0 et 5 personnes/m²
    rho = 5.0 * np.exp(-((x_f - (3 + 2*t_f))**2 + (y_f - 1)**2) / 4.0)
    
    # 3. RELATION FONDAMENTALE (Greenshields)
    v_max = 1.34  # Vitesse libre (m/s)
    rho_max = 6.0 # Densité de blocage
    
    # La vitesse diminue quand la densité augmente
    vx = v_max * (1 - rho / rho_max)
    vy = np.zeros_like(vx) # On reste dans l'axe de la rue

    return xyt, rho, vx, vy

def run_pinn_analysis():
    # --- 1. Préparation ---¨
    # on extrait les données réalistes de la "fausse" simulation pour la loss physique
    xyt, rho, vx, vy = generate_realistic_data() 
    
    
    # --- 2. Entraînement du PINN ---
    # n_hidden nombre de neuronne dans la couche caché élevé pour capturer la non-linéarité du diagramme
    model = MassFlowPINN(n_hidden=300, alpha_physics=100.0)
    print("Entraînement du PINN en cours...")
    model.fit(xyt, rho, vx, vy)

    # --- 3. Prédiction ---
    phi_x_pred, phi_y_pred = model.predict(xyt)
    flux_total = np.sqrt(phi_x_pred**2 + phi_y_pred**2)

    # --- 4. Visualisation ---
    fig, (ax1) = plt.subplots(1, figsize=(16, 6))

    # Graphe 1: LE DIAGRAMME FONDAMENTAL
    # On trace le Flux prédit par le PINN en fonction de la Densité d'entrée
    ax1.scatter(rho, flux_total, color='teal', alpha=0.3, s=5, label='Prédictions PINN')
    
    # On ajoute la courbe théorique pour comparer
    r_axis = np.linspace(0, 6, 100)
    f_theo = r_axis * (1.34 * (1 - r_axis/6.0))
    ax1.plot(r_axis, f_theo, 'r--', linewidth=2, label='Théorie (Greenshields)')

    ax1.set_title("Diagramme Fondamental (Flux vs Densité) d'après une simulation par modèle de greenshields")
    ax1.set_xlabel("Densité φ (personnes/m²)")
    ax1.set_ylabel("Flux total ||Φ|| (personnes/m/s)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pinn_analysis()



def run_example():
    # 1. Préparation des données
    phi_max = np.pi
    x_train = np.linspace(0, phi_max, 50)
    y_train = np.sin(x_train)

    # 2. Entraînement
    model = SimplePINN(n_hidden=200, alpha=50.0) #on utilise le deuxième PINN quit "fit" la courbe d'un sinus
    model.fit(x_train, y_train)

    # 3. Prédiction sur une grille fine pour l'affichage
    x_test = np.linspace(0, phi_max, 500)
    y_pred = model.predict(x_test)
    y_true = np.sin(x_test)
    
    # Calcul de l'erreur locale (résidus)
    error_local = np.abs(y_true - y_pred)
    mae = model.get_mae(x_test, y_true)

    # 4. Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Graphique 1 : Fitting ---
    ax1.plot(x_test, y_true, 'b-', label='Sinus Réel (Cible)', alpha=0.5)
    ax1.plot(x_test, y_pred, 'r--', label='Prédiction PINN', linewidth=2)
    ax1.scatter(x_train, y_train, color='black', s=15, label='Points d\'entraînement')
    ax1.set_title(f"Fitting PINN (MAE: {mae:.6e})")
    ax1.set_xlabel("Angle φ (radians)")
    ax1.set_ylabel("y")
    ax1.set_xticks([0, np.pi/2, np.pi])
    ax1.set_xticklabels(['0', 'π/2', 'π'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Graphique 2 : Erreur de prédiction ---
    ax2.fill_between(x_test, error_local, color='green', alpha=0.2)
    ax2.plot(x_test, error_local, 'g-', label='Erreur Absolue |y_true - y_pred|')
    ax2.set_title("Distribution de l'erreur sur l'intervalle")
    ax2.set_xlabel("Angle φ (radians)")
    ax2.set_ylabel("Erreur")
    ax2.set_xticks([0, np.pi/2, np.pi])
    ax2.set_xticklabels(['0', 'π/2', 'π'])
    ax2.set_yscale('log') # Échelle logarithmique pour mieux voir les petites erreurs
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_example()

import numpy as np
from scipy.linalg import lstsq

class MassFlowPINN:
    def __init__(self, n_hidden=150, alpha_physics=20.0):
        self.n_hid = n_hidden
        self.alpha = alpha_physics
        self.weights_out = None # Contiendra les poids concaténés [Wx, Wy]

    def _get_basis_and_derivs(self, xyt):
        # xyt shape: (N, 3) -> [x, y, t]
        np.random.seed(42)
        W1 = np.random.randn(3, self.n_hid) * 0.5
        b1 = np.random.randn(1, self.n_hid) * 0.1
        
        Z = np.dot(xyt, W1) + b1
        phi_h = np.tanh(Z)
        sech2 = 1 - phi_h**2
        
        # Dérivées de la couche cachée par rapport à x et y
        dHdx = W1[0, :] * sech2
        dHdy = W1[1, :] * sech2
        return phi_h, dHdx, dHdy

    def fit(self, xyt, rho, vx, vy):
        N = len(xyt)
        H, dHdx, dHdy = self._get_basis_and_derivs(xyt)
        
        # On cherche w_x et w_y tels que :
        # 1. Données : H*w_x = rho*vx  ET  H*w_y = rho*vy
        # 2. Physique : dHdx*w_x + dHdy*w_y = -d_rho/dt
        
        # --- Bloc Données ---
        # Matrice diagonale par bloc pour résoudre w_x et w_y simultanément
        A_data = np.block([
            [H,                np.zeros_like(H)],
            [np.zeros_like(H), H               ]
        ])
        b_data = np.hstack([(rho * vx).flatten(), (rho * vy).flatten()])

        # --- Bloc Physique (Conservation : div(Phi) = -d_rho/dt) ---
        # On approxime d_rho/dt temporellement
        drho_dt = np.gradient(rho.flatten(), xyt[:, 2]) 
        
        A_phys = self.alpha * np.block([[dHdx, dHdy]])
        b_phys = self.alpha * (-drho_dt)

        # --- Bloc Boundary (Flux nul aux bords de la rue) ---
        # On force Phi_x = 0 à x=0 et x=L, et Phi_y = 0 aux trottoirs
        A_total = np.vstack([A_data, A_phys])
        b_total = np.hstack([b_data, b_phys])

        self.weights_out, _, _, _ = lstsq(A_total, b_total)

    def predict(self, xyt):
        H, _, _ = self._get_basis_and_derivs(xyt)
        n_w = self.n_hid
        wx = self.weights_out[:n_w]
        wy = self.weights_out[n_w:]
        return H @ wx, H @ wy
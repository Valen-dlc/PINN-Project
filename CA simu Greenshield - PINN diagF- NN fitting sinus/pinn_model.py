import numpy as np
from scipy.linalg import lstsq

class SimplePINN:
    def __init__(self, n_hidden=20, alpha=1.0):
        """
        Initialise un PINN à une couche cachée.
        n_hidden : nombre de neurones dans la couche cachée.
        alpha : poids accordé à la contrainte physique (y'' + y = 0).
        """
        self.n_in = 1
        self.n_hid = n_hidden
        self.alpha = alpha
        self.weights_out = None
        
        # Initialisation fixe des poids de la couche cachée
        np.random.seed(42)
        self.W1 = np.random.randn(self.n_in, self.n_hid) * 2
        self.b1 = np.random.randn(1, self.n_hid) * 2

    def _get_basis(self, x):
        """Calcule la couche cachée et sa dérivée seconde."""
        X = x.reshape(-1, self.n_in)
        Z = np.dot(X, self.W1) + self.b1
        phi = np.tanh(Z)
        
        # Physique : d²phi/dx² pour tanh
        sech2 = 1 - phi**2
        d2phi = -2 * (self.W1**2) * phi * sech2
        return phi, d2phi

    def fit(self, x_train, y_train):
        """Entraîne le réseau via les moindres carrés."""
        Phi, D2Phi = self._get_basis(x_train)
        
        # Construction du système A*w = b
        A_data = Phi
        b_data = y_train
        
        # Physique : d²y/dx² + y = 0  => (D2Phi + Phi) * w = 0
        A_physics = self.alpha * (D2Phi + Phi)
        b_physics = np.zeros(len(x_train))
        
        A = np.vstack([A_data, A_physics])
        b = np.hstack([b_data, b_physics])
        
        # Résolution
        self.weights_out, _, _, _ = lstsq(A, b)

    def predict(self, x):
        """Prédit les valeurs pour un set de points x."""
        if self.weights_out is None:
            raise ValueError("Le modèle doit être entraîné avec .fit() avant de prédire.")
        
        phi, _ = self._get_basis(x)
        return phi @ self.weights_out
    
    def get_mae(self, x, y_true):
        """Calcule l'erreur moyenne absolue (MAE)."""
        y_pred = self.predict(x)
        return np.mean(np.abs(y_true - y_pred))
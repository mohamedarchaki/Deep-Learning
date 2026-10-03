import numpy as np

class AdamOptimizer:
    def __init__(self, weights, biases, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Compteur de pas de temps

        # Initialisation des premiers et seconds moments pour W et b
        self.m_w = [np.zeros_like(w) for w in weights]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, d_weights, d_biases):
        """
        Met a jour les parametres du reseau selon l'algorithme Adam.
        """
        self.t += 1

        for i in range(len(weights)):
            # Mise a jour des moyennes mobiles pour les poids W
            self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * d_weights[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1.0 - self.beta2) * np.square(d_weights[i])

            # Correction de biais pour W
            m_w_corrected = self.m_w[i] / (1.0 - (self.beta1 ** self.t))
            v_w_corrected = self.v_w[i] / (1.0 - (self.beta2 ** self.t))

            weights[i] -= self.lr * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)

            # Mise a jour des moyennes mobiles pour les biais b
            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * d_biases[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * np.square(d_biases[i])

            # Correction de biais pour b
            m_b_corrected = self.m_b[i] / (1.0 - (self.beta1 ** self.t))
            v_b_corrected = self.v_b[i] / (1.0 - (self.beta2 ** self.t))

            biases[i] -= self.lr * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
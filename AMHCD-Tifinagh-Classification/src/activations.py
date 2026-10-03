import numpy as np

def relu(x):
    """
    Activation ReLU : max(0, x)
    Formule mathématique : g(z) = max(0, z)
    """
    assert isinstance(x, np.ndarray), "Input to ReLU must be a numpy array"
    
    # Implémentation du TODO : max(0, x)
    result = np.maximum(0, x)
    
    assert np.all(result >= 0), "ReLU output must be non-negative"
    return result


def relu_derivative(x):
    """
    Dérivée de ReLU : 1 si x > 0, sinon 0
    Formule : ReLU'(z) = 1 si z > 0, 0 sinon
    """
    assert isinstance(x, np.ndarray), "Input to ReLU derivative must be a numpy array"
    
    # Implémentation du TODO : condition booléenne convertie en entiers/flottants
    result = (x > 0).astype(float)
    
    assert np.all((result == 0) | (result == 1)), "ReLU derivative must be 0 or 1"
    return result


def softmax(x):
    """
    Activation Softmax : exp(x) / sum(exp(x))
    Formule : exp(z_i) / sum_j exp(z_j)
    """
    assert isinstance(x, np.ndarray), "Input to softmax must be a numpy array"
    
    # Stabilité numérique : soustraire le max pour éviter l'overflow
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shift_x)
    
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    assert np.all((result >= 0) & (result <= 1)), "Softmax output must be in [0, 1]"
    assert np.allclose(np.sum(result, axis=1), 1), "Softmax output must sum to 1 per sample"
    return result
# Le modèle "Ferry & Rush" : tout ce qu'il faut pour PRÉDIRE (utilisé par l'app Flask).
# L'entraînement (données, Fortuna, évaluation) est dans train.py.

import numpy as np

# Les 4 routes et leurs interrupteurs (is_from_B, goes_to_D) :
#   is_from_B = 1 si la route part de B, goes_to_D = 1 si elle va vers D
ROUTES = {
    "A->C->D": (0, 1),
    "A->C->E": (0, 0),
    "B->C->D": (1, 1),
    "B->C->E": (1, 0),
}

# Les 7 paramètres de theta, dans l'ordre où la formule les déballe :
#   E     : durée d'une route vers E (min)
#   g     : gain de B quand on attrape le ferry (min, négatif)
#   phi   : minute de départ qui attrape tout juste le ferry
#   d_low : durée vers D sans bouchon, le fond de la cuvette (min)
#   d_amp : bouchon maximal ajouté aux heures de pointe (min)
#   c     : centre de la période calme (heures décimales)
#   w     : largeur de la période calme (heures)
PARAM_NAMES = ["E", "g", "phi", "d_low", "d_amp", "c", "w"]


def predict_duration(theta, t, m, is_from_B, goes_to_D):
    # Durée prédite (min). Marche pour un départ comme pour tous les trajets d'un coup
    E, g, phi, d_low, d_amp, c, w = theta

    # tronçon C->D : cuvette des heures de pointe (0 bouchon à c, d_amp loin de c)
    rush = d_low + d_amp * (1 - np.exp(-((t - c) / w) ** 4))

    # tronçon B->C : attente du prochain ferry (0 à 60 min)
    wait = (phi - m) % 60

    return (1 - goes_to_D) * E + goes_to_D * rush + is_from_B * (g + wait)

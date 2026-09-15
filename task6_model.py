import json
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent / "data" / "traffic.jsonl"

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

# Bornes de chaque paramètre, dans le même ordre que PARAM_NAMES
# Observées dans les données, ou choisies artbitrairement
#                  E    g  phi  d_low  d_amp   c    w
LOW  = np.array([ 80, -40,   0,    50,     0,  9, 0.5])
HIGH = np.array([120,   0,  60,   100,    80, 15, 6.0])


def to_minutes(hhmm):
    # "13:17" -> 797 (minutes depuis minuit)
    d = datetime.strptime(hhmm, "%H:%M")
    return d.hour * 60 + d.minute


def load_data():
    # Renvoie 5 tableaux numpy (un élément par trajet) :
    #   t    : heure de départ en heures décimales (13:17 -> 13.28)
    #   m    : minute de départ dans l'heure (13:17 -> 17)
    #   is_from_B : 1 si la route part de B, sinon 0
    #   goes_to_D : 1 si la route va vers D, sinon 0
    #   y    : durée réelle du trajet en minutes
    t, m, is_from_B, goes_to_D, y = [], [], [], [], []

    with open(DATA_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            trip = json.loads(line)

            dep = to_minutes(trip["depature"]) 
            arr = to_minutes(trip["arrival"])

            t.append(dep / 60)
            m.append(dep % 60)
            is_from_B.append(trip["road"].startswith("B"))
            goes_to_D.append(trip["road"].endswith("D"))
            y.append(arr - dep)

    return (np.array(t, dtype=float), np.array(m, dtype=float),
            np.array(is_from_B, dtype=float), np.array(goes_to_D, dtype=float),
            np.array(y, dtype=float))


def predict_duration(theta, t, m, is_from_B, goes_to_D):
    # Durée prédite (min). Marche pour un départ comme pour tous les trajets d'un coup
    E, g, phi, d_low, d_amp, c, w = theta

    # tronçon C->D : cuvette des heures de pointe (0 bouchon à c, d_amp loin de c)
    rush = d_low + d_amp * (1 - np.exp(-((t - c) / w) ** 4))

    # tronçon B->C : attente du prochain ferry (0 à 60 min)
    wait = (phi - m) % 60

    return (1 - goes_to_D) * E + goes_to_D * rush + is_from_B * (g + wait)


if __name__ == "__main__":
    print("to_minutes('13:17') =", to_minutes("13:17"))

    t, m, is_from_B, goes_to_D, y = load_data()
    print("trajets      :", len(y))
    print("t min / max  :", t.min(), "/", round(t.max(), 2))
    print("y min / max  :", y.min(), "/", y.max(), "| moyenne :", round(y.mean(), 2))
    print("départs de B :", int(is_from_B.sum()))
    print("vers D       :", int(goes_to_D.sum()))
    print("paramètres   :", PARAM_NAMES)

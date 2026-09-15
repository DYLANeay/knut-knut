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


def get_loss(theta, t, m, is_from_B, goes_to_D, y):
    # Score d'un theta : erreur quadratique moyenne (MSE), plus c'est bas, mieux c'est
    y_hat = predict_duration(theta, t, m, is_from_B, goes_to_D)
    return np.mean((y_hat - y) ** 2)


def sample_theta(rng):
    # Un theta tiré au hasard : chaque paramètre dans ses propres bornes [LOW, HIGH]
    return rng.uniform(LOW, HIGH)


# data = (t, m, is_from_B, goes_to_D, y), tel que renvoyé par load_data()
# Les 3 versions renvoient (best_loss, best_theta) et ont le même budget n_evals.

def fortuna_classic(data, n_evals, rng):
    # Version 1 : chaque essai est tiré au hasard, on garde le meilleur
    best_theta = sample_theta(rng)
    best_loss = get_loss(best_theta, *data)

    for _ in range(n_evals - 1):
        theta = sample_theta(rng)
        loss = get_loss(theta, *data)
        if loss < best_loss:
            best_loss, best_theta = loss, theta

    return best_loss, best_theta


def fortuna_local(data, n_evals, rng, step_start=0.2, step_end=0.002):
    # Version 2 (tâche 4) : on cherche autour du meilleur theta, avec un pas qui rétrécit.
    # Le pas est une fraction de la largeur de chaque intervalle (HIGH - LOW),
    # pour que phi (largeur 60) et w (largeur 5.5) bougent chacun à leur échelle.
    best_theta = sample_theta(rng)
    best_loss = get_loss(best_theta, *data)

    for i in range(n_evals - 1):
        step = step_start + (step_end - step_start) * i / n_evals
        theta = best_theta + rng.normal(0, 1, size=len(LOW)) * step * (HIGH - LOW)
        theta = np.clip(theta, LOW, HIGH)
        loss = get_loss(theta, *data)
        if loss < best_loss:
            best_loss, best_theta = loss, theta

    return best_loss, best_theta


def fortuna_restarts(data, n_evals, rng, n_restarts=10):
    # Version 3 : plusieurs recherches locales courtes depuis des départs différents,
    # on garde la meilleure. Même budget total : n_evals est partagé entre les redémarrages.
    best_loss, best_theta = np.inf, None

    for _ in range(n_restarts):
        loss, theta = fortuna_local(data, n_evals // n_restarts, rng)
        if loss < best_loss:
            best_loss, best_theta = loss, theta

    return best_loss, best_theta


def compare_fortunas(data, n_runs=10, n_evals=30000, seed=0, success_loss=20):
    # Lance chaque version n_runs fois avec le même budget et affiche un résumé
    rng = np.random.default_rng(seed)
    versions = {
        "classique": fortuna_classic,
        "local": fortuna_local,
        "redémarrages": fortuna_restarts,
    }

    for name, fortuna in versions.items():
        losses, thetas = [], []
        for _ in range(n_runs):
            loss, theta = fortuna(data, n_evals, rng)
            losses.append(loss)
            thetas.append(theta)
        losses = np.array(losses)

        print(f"{name:13s} | moyenne {losses.mean():6.1f} | médiane {np.median(losses):6.1f} "
              f"| min {losses.min():6.1f} | max {losses.max():6.1f} "
              f"| réussis {(losses < success_loss).sum()}/{n_runs}")
        # phi des runs ratés : pour vérifier l'explication de la borne 0/60
        failed = [round(th[2], 1) for th, l in zip(thetas, losses) if l >= success_loss]
        if failed and name != "classique":
            print(f"{'':13s}   phi des runs ratés : {failed}")


if __name__ == "__main__":
    print("to_minutes('13:17') =", to_minutes("13:17"))

    data = load_data()
    t, m, is_from_B, goes_to_D, y = data
    print("trajets      :", len(y))
    print("t min / max  :", t.min(), "/", round(t.max(), 2))
    print("y min / max  :", y.min(), "/", y.max(), "| moyenne :", round(y.mean(), 2))
    print("départs de B :", int(is_from_B.sum()))
    print("vers D       :", int(goes_to_D.sum()))
    print("paramètres   :", PARAM_NAMES)

    print("\nComparaison des 3 versions de Fortuna :")
    compare_fortunas(data)

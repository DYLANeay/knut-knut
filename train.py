# Entraînement du modèle "Ferry & Rush" : données, Fortuna, comparaison.
# La formule du modèle elle-même est dans route_model.py.

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from route_model import PARAM_NAMES, ROUTES, predict_duration

DATA_PATH = Path(__file__).parent / "data" / "traffic.jsonl"

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
        loss = get_loss(theta, *data) #* = opérateur de unpacking
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


def split_data(data, test_ratio=0.2, seed=0):
    # Répartit les VRAIS trajets en deux groupes, au hasard (aucune ligne créée ni modifiée) :
    #   train : utilisé par Fortuna pour trouver theta
    #   test  : caché pendant l'entraînement, sert à mesurer l'erreur sur des trajets "nouveaux"
    n = len(data[-1])
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    train = tuple(a[train_idx] for a in data)
    test = tuple(a[test_idx] for a in data)
    return train, test


def errors(y_hat, y):
    # MAE : erreur moyenne (min) ; RMSE : erreur typique (min), pénalise plus les grosses erreurs
    residuals = y_hat - y
    return np.mean(np.abs(residuals)), np.sqrt(np.mean(residuals ** 2))


def evaluate(theta, data):
    # Erreurs du modèle sur un groupe de trajets : au total, puis route par route
    t, m, is_from_B, goes_to_D, y = data
    y_hat = predict_duration(theta, t, m, is_from_B, goes_to_D)

    results = {"toutes": errors(y_hat, y)}
    for road, (b, d) in ROUTES.items():
        mask = (is_from_B == b) & (goes_to_D == d)
        results[road] = errors(y_hat[mask], y[mask])
    return results


def evaluate_route_means(train, test):
    # Point de comparaison sans modèle : prédire la durée moyenne de la route (calculée sur train)
    _, _, b_train, d_train, y_train = train
    _, _, b_test, d_test, y_test = test

    y_hat = np.zeros_like(y_test)
    for b, d in ROUTES.values():
        mean_train = y_train[(b_train == b) & (d_train == d)].mean()
        y_hat[(b_test == b) & (d_test == d)] = mean_train
    return errors(y_hat, y_test)


def print_evaluation(train, test, theta):
    train_res, test_res = evaluate(theta, train), evaluate(theta, test)

    print(f"{'':10s} | {'MAE train':>9s} | {'MAE test':>8s} | {'RMSE train':>10s} | {'RMSE test':>9s}")
    for name in train_res:
        (mae_tr, rmse_tr), (mae_te, rmse_te) = train_res[name], test_res[name]
        print(f"{name:10s} | {mae_tr:9.2f} | {mae_te:8.2f} | {rmse_tr:10.2f} | {rmse_te:9.2f}")

    mae_base, rmse_base = evaluate_route_means(train, test)
    print(f"\nsans modèle (moyenne de chaque route) : MAE test {mae_base:.2f} | RMSE test {rmse_base:.2f}")


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

    # Étape 6 : entraîner sur train uniquement, mesurer sur test
    train, test = split_data(data)
    print(f"\nSéparation : {len(train[-1])} trajets train, {len(test[-1])} trajets test")

    loss, theta = fortuna_restarts(train, 30000, np.random.default_rng(0))
    print(f"loss train : {loss:.2f}")
    print("theta      :", {name: round(float(v), 2) for name, v in zip(PARAM_NAMES, theta)})
    print()
    print_evaluation(train, test, theta)

    print("\nComparaison des 3 versions de Fortuna :")
    compare_fortunas(data)

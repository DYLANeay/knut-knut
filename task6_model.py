import json
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent / "data" / "traffic.jsonl"


def to_minutes(hhmm):
    # "13:17" -> 797 (minutes depuis minuit)
    d = datetime.strptime(hhmm, "%H:%M")
    return d.hour * 60 + d.minute


def load_data():
    # Renvoie 5 tableaux numpy (un élément par trajet) :
    #   t    : heure de départ en heures décimales (13:17 -> 13.28)
    #   m    : minute de départ dans l'heure (13:17 -> 17)
    #   is_B : 1 si la route part de B, sinon 0
    #   is_D : 1 si la route va vers D, sinon 0
    #   y    : durée réelle du trajet en minutes
    t, m, is_B, is_D, y = [], [], [], [], []

    with open(DATA_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            trip = json.loads(line)

            dep = to_minutes(trip["depature"])  # "depature" : faute de frappe dans les données
            arr = to_minutes(trip["arrival"])

            t.append(dep / 60)
            m.append(dep % 60)
            is_B.append(trip["road"].startswith("B"))
            is_D.append(trip["road"].endswith("D"))
            y.append(arr - dep)

    return (np.array(t, dtype=float), np.array(m, dtype=float),
            np.array(is_B, dtype=float), np.array(is_D, dtype=float),
            np.array(y, dtype=float))


if __name__ == "__main__":
    print("to_minutes('13:17') =", to_minutes("13:17"))

    t, m, is_B, is_D, y = load_data()
    print("trajets      :", len(y))
    print("t min / max  :", t.min(), "/", round(t.max(), 2))
    print("y min / max  :", y.min(), "/", y.max(), "| moyenne :", round(y.mean(), 2))
    print("départs de B :", int(is_B.sum()))
    print("vers D       :", int(is_D.sum()))

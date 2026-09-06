from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    razdeli_znacilke_in_cilj,
    zgradi_predobdelavo
)


KORENSKA_MAPA = Path(__file__).resolve().parents[1]

POT_PODATKOV = (
    KORENSKA_MAPA
    / "data"
    / "cars_features.csv"
)

POT_MODELA = (
    KORENSKA_MAPA
    / "models"
    / "linear_regression_model.joblib"
)


def main() -> None:

    print("Nalaganje podatkov...")

    podatki = pd.read_csv(
        POT_PODATKOV
    )

    vhodni_podatki, cilj = (
        razdeli_znacilke_in_cilj(
            podatki
        )
    )

    (
        vhod_trening,
        vhod_test,
        cilj_trening,
        cilj_test
    ) = train_test_split(
        vhodni_podatki,
        cilj,
        test_size=0.20,
        random_state=42
    )

    print(
        "Število trening primerov:",
        len(vhod_trening)
    )

    print(
        "Število testnih primerov:",
        len(vhod_test)
    )

    model = Pipeline(
        steps=[
            (
                "predobdelava",
                zgradi_predobdelavo()
            ),
            (
                "regresor",
                LinearRegression()
            )
        ]
    )

    print(
        "Treniranje linearne regresije..."
    )

    model.fit(
        vhod_trening,
        cilj_trening
    )

    POT_MODELA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        POT_MODELA
    )

    print(
        "Model shranjen v:",
        POT_MODELA
    )


if __name__ == "__main__":
    main()

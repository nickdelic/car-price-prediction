from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

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

POT_FINALNEGA_MODELA = (
    KORENSKA_MAPA
    / "models"
    / "car_price_model.joblib"
)


def main() -> None:

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

    modeli = {
        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            )
    }

    rezultati = []

    najboljsi_mae = float("inf")
    najboljsi_model = None
    ime_najboljsega_modela = None

    for ime_modela, regresor in modeli.items():

        print(
            "\nTreniranje:",
            ime_modela
        )

        model = Pipeline(
            steps=[
                (
                    "predobdelava",
                    zgradi_predobdelavo()
                ),
                (
                    "regresor",
                    regresor
                )
            ]
        )

        model.fit(
            vhod_trening,
            cilj_trening
        )

        napovedi = model.predict(
            vhod_test
        )

        mae = mean_absolute_error(
            cilj_test,
            napovedi
        )

        mse = mean_squared_error(
            cilj_test,
            napovedi
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            cilj_test,
            napovedi
        )

        rezultati.append({
            "model": ime_modela,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        })

        print(
            f"MAE: ${mae:,.2f}"
        )

        print(
            f"RMSE: ${rmse:,.2f}"
        )

        print(
            f"R²: {r2:.4f}"
        )

        if mae < najboljsi_mae:

            najboljsi_mae = mae
            najboljsi_model = model
            ime_najboljsega_modela = (
                ime_modela
            )

    tabela_rezultatov = (
        pd.DataFrame(
            rezultati
        )
        .sort_values(
            "mae",
            ascending=True
        )
        .reset_index(drop=True)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PRIMERJAVA MODELOV"
    )

    print(
        "=" * 70
    )

    print(
        tabela_rezultatov
    )

    print(
        "\nNajboljši model:",
        ime_najboljsega_modela
    )

    print(
        f"Najboljši MAE: "
        f"${najboljsi_mae:,.2f}"
    )

    POT_FINALNEGA_MODELA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        najboljsi_model,
        POT_FINALNEGA_MODELA
    )

    print(
        "\nFinalni model shranjen v:",
        POT_FINALNEGA_MODELA
    )


if __name__ == "__main__":
    main()

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    razdeli_znacilke_in_cilj
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

    model = joblib.load(
        POT_MODELA
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

    print("\nREGRESIJSKE METRIKE")
    print("=" * 50)

    print(
        f"MAE:  ${mae:,.2f}"
    )

    print(
        f"MSE:  {mse:,.2f}"
    )

    print(
        f"RMSE: ${rmse:,.2f}"
    )

    print(
        f"R²:    {r2:.4f}"
    )

    analiza = pd.DataFrame({
        "dejanska_cena": cilj_test.values,
        "napovedana_cena": napovedi
    })

    analiza["napaka"] = (
        analiza["dejanska_cena"]
        - analiza["napovedana_cena"]
    )

    analiza["absolutna_napaka"] = (
        analiza["napaka"].abs()
    )

    print("\n10 naključnih primerov:")

    print(
        analiza.sample(
            10,
            random_state=42
        )
    )

    print(
        "\n10 največjih napak:"
    )

    print(
        analiza
        .sort_values(
            "absolutna_napaka",
            ascending=False
        )
        .head(10)
    )


if __name__ == "__main__":
    main()

from datetime import datetime
from pathlib import Path

import pandas as pd


KORENSKA_MAPA = Path(__file__).resolve().parents[1]

POT_OCISCENIH_PODATKOV = (
    KORENSKA_MAPA / "data" / "cars_cleaned.csv"
)

POT_PODATKOV_Z_ZNACILKAMI = (
    KORENSKA_MAPA
    / "data"
    / "cars_features.csv"
)


def dodaj_starost_avtomobila(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    trenutno_leto = datetime.now().year

    podatki["car_age"] = (
        trenutno_leto
        - podatki["year"]
    )

    return podatki


def dodaj_kilometrino_na_leto(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    starost_za_izracun = (
        podatki["car_age"]
        .clip(lower=1)
    )

    podatki["mileage_per_year"] = (
        podatki["mileage_kilometers"]
        / starost_za_izracun
    )

    return podatki


def dodaj_prostornino_v_litrih(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    podatki["engine_volume_liters"] = (
        podatki["volume_cm3"]
        / 1000
    )

    return podatki


def dodaj_znamko_in_model(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    znamka = (
        podatki["make"]
        .fillna("unknown")
        .astype(str)
    )

    model = (
        podatki["model"]
        .fillna("unknown")
        .astype(str)
    )

    podatki["brand_model"] = (
        znamka
        + " "
        + model
    )

    return podatki


def zgradi_znacilke(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    novi_podatki = (
        podatki
        .pipe(
            dodaj_starost_avtomobila
        )
        .pipe(
            dodaj_kilometrino_na_leto
        )
        .pipe(
            dodaj_prostornino_v_litrih
        )
        .pipe(
            dodaj_znamko_in_model
        )
        .reset_index(drop=True)
    )

    return novi_podatki


def main() -> None:

    print(
        "Nalaganje očiščenih podatkov..."
    )

    podatki = pd.read_csv(
        POT_OCISCENIH_PODATKOV
    )

    print(
        "Izdelava novih značilk..."
    )

    podatki_z_znacilkami = (
        zgradi_znacilke(
            podatki
        )
    )

    podatki_z_znacilkami.to_csv(
        POT_PODATKOV_Z_ZNACILKAMI,
        index=False
    )

    print(
        "Podatki z značilkami shranjeni v:",
        POT_PODATKOV_Z_ZNACILKAMI
    )


if __name__ == "__main__":
    main()

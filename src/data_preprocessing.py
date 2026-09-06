import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


CILJNA_KOLONA = "price_usd"


NUMERICNE_ZNACILKE = [
    "mileage_kilometers",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters"
]


KATEGORIJSKE_ZNACILKE = [
    "make",
    "brand_model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment"
]


def pridobi_vse_znacilke() -> list[str]:

    return (
        NUMERICNE_ZNACILKE
        + KATEGORIJSKE_ZNACILKE
    )


def razdeli_znacilke_in_cilj(
    podatki: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:

    vhodni_podatki = podatki[
        pridobi_vse_znacilke()
    ].copy()

    cilj = podatki[
        CILJNA_KOLONA
    ].copy()

    return vhodni_podatki, cilj


def zgradi_numericni_transformer() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


def zgradi_kategorijski_transformer() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )


def zgradi_predobdelavo() -> ColumnTransformer:

    predobdelava = ColumnTransformer(
        transformers=[
            (
                "numericne",
                zgradi_numericni_transformer(),
                NUMERICNE_ZNACILKE
            ),
            (
                "kategorijske",
                zgradi_kategorijski_transformer(),
                KATEGORIJSKE_ZNACILKE
            )
        ],
        remainder="drop"
    )

    return predobdelava

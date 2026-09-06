import re
from datetime import datetime
from pathlib import Path

import pandas as pd


KORENSKA_MAPA = Path(__file__).resolve().parents[1]

POT_SUROVIH_PODATKOV = (
    KORENSKA_MAPA / "data" / "cars.csv"
)

POT_OCISCENIH_PODATKOV = (
    KORENSKA_MAPA / "data" / "cars_cleaned.csv"
)


MANJKAJOCE_VREDNOSTI = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE"
}


def standardiziraj_imena_stolpcev(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    nova_imena = []

    for ime in podatki.columns:

        novo_ime = str(ime).strip()

        # priceUSD -> price_USD
        novo_ime = re.sub(
            r"([a-z0-9])([A-Z])",
            r"\1_\2",
            novo_ime
        )

        novo_ime = novo_ime.lower()

        novo_ime = novo_ime.replace(
            "(",
            "_"
        )

        novo_ime = novo_ime.replace(
            ")",
            ""
        )

        novo_ime = novo_ime.replace(
            "-",
            "_"
        )

        novo_ime = novo_ime.replace(
            "/",
            "_"
        )

        novo_ime = re.sub(
            r"\s+",
            "_",
            novo_ime
        )

        novo_ime = re.sub(
            r"[^a-z0-9_]",
            "",
            novo_ime
        )

        novo_ime = re.sub(
            r"_+",
            "_",
            novo_ime
        )

        novo_ime = novo_ime.strip("_")

        nova_imena.append(
            novo_ime
        )

    podatki.columns = nova_imena

    return podatki


def odstrani_odvecne_presledke(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    tekstovni_stolpci = (
        podatki
        .select_dtypes(
            include=["object", "string"]
        )
        .columns
    )

    for stolpec in tekstovni_stolpci:

        podatki[stolpec] = (
            podatki[stolpec]
            .astype("string")
            .str.strip()
        )

    return podatki


def standardiziraj_manjkajoce_vrednosti(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    podatki = podatki.replace(
        list(MANJKAJOCE_VREDNOSTI),
        pd.NA
    )

    return podatki


def pretvori_numericne_stolpce(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    numericni_stolpci = [
        "price_usd",
        "year",
        "mileage_kilometers",
        "volume_cm3"
    ]

    for stolpec in numericni_stolpci:

        if stolpec in podatki.columns:

            vrednosti = (
                podatki[stolpec]
                .astype("string")
                .str.replace(
                    r"[^\d.\-]",
                    "",
                    regex=True
                )
            )

            podatki[stolpec] = pd.to_numeric(
                vrednosti,
                errors="coerce"
            )

    return podatki


def standardiziraj_kategorije(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    kategorijski_stolpci = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment"
    ]

    for stolpec in kategorijski_stolpci:

        if stolpec in podatki.columns:

            podatki[stolpec] = (
                podatki[stolpec]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    return podatki


def odstrani_neveljavne_vrstice(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    podatki = podatki.copy()

    trenutno_leto = datetime.now().year

    # Brez ciljne vrednosti modela ni mogoče trenirati.
    podatki = podatki.dropna(
        subset=["price_usd", "year"]
    )

    # Cena mora biti pozitivna.
    podatki = podatki[
        podatki["price_usd"] > 0
    ]

    # Leto proizvodnje mora biti realno.
    podatki = podatki[
        podatki["year"].between(
            1900,
            trenutno_leto
        )
    ]

    # Negativna kilometrina je neveljavna.
    if "mileage_kilometers" in podatki.columns:

        podatki.loc[
            podatki["mileage_kilometers"] < 0,
            "mileage_kilometers"
        ] = pd.NA

    # Neveljavne prostornine motorja obravnavamo
    # kot manjkajočo vrednost.
    if "volume_cm3" in podatki.columns:

        podatki.loc[
            podatki["volume_cm3"] <= 0,
            "volume_cm3"
        ] = pd.NA

    podatki = podatki.drop_duplicates()

    return podatki


def ocisti_podatke(
    podatki: pd.DataFrame
) -> pd.DataFrame:

    ocisceni_podatki = (
        podatki
        .pipe(
            standardiziraj_imena_stolpcev
        )
        .pipe(
            odstrani_odvecne_presledke
        )
        .pipe(
            standardiziraj_manjkajoce_vrednosti
        )
        .pipe(
            pretvori_numericne_stolpce
        )
        .pipe(
            standardiziraj_kategorije
        )
        .pipe(
            odstrani_neveljavne_vrstice
        )
        .reset_index(drop=True)
    )

    return ocisceni_podatki


def main() -> None:

    print("Nalaganje surovih podatkov...")

    podatki = pd.read_csv(
        POT_SUROVIH_PODATKOV
    )

    print(
        "Število vrstic pred čiščenjem:",
        len(podatki)
    )

    ocisceni_podatki = ocisti_podatke(
        podatki
    )

    print(
        "Število vrstic po čiščenju:",
        len(ocisceni_podatki)
    )

    ocisceni_podatki.to_csv(
        POT_OCISCENIH_PODATKOV,
        index=False
    )

    print(
        "Očiščeni podatki shranjeni v:",
        POT_OCISCENIH_PODATKOV
    )


if __name__ == "__main__":
    main()

# car-price-prediction

# Predikcija cene rabljenih avtomobilov

Projekt uporablja regresijske algoritme strojnega učenja za napovedovanje cene rabljenega avtomobila v USD.

## Podatki

Uporabljen je podatkovni nabor `cars.csv`.

Ciljna spremenljivka je:

`priceUSD`

## Struktura projekta

- `data/cars.csv` - surovi podatki
- `notebooks/01_eda.ipynb` - raziskovalna analiza podatkov
- `src/data_cleaning.py` - čiščenje podatkov
- `src/feature_engineering.py` - izdelava novih značilk
- `src/data_preprocessing.py` - priprava podatkov za modele
- `src/model_training.py` - trening osnovnega modela
- `src/model_evaluation.py` - evaluacija modela
- `src/model_comparison.py` - primerjava regresijskih algoritmov
- `models/car_price_model.joblib` - finalni model

## Inženiring značilk

Izdelane so bile naslednje nove značilke:

- `car_age`
- `mileage_per_year`
- `engine_volume_liters`
- `brand_model`

Te značilke bolje predstavijo starost avtomobila, intenzivnost uporabe, velikost motorja ter kombinacijo znamke in modela.

## Pretprocesiranje

Numerične značilke:

- manjkajoče vrednosti se dopolnijo z mediano,
- vrednosti se standardizirajo s StandardScalerjem.

Kategorijske značilke:

- manjkajoče vrednosti se dopolnijo z najpogostejšo vrednostjo,
- kategorije se pretvorijo z OneHotEncoderjem.

## Modeli

Primerjani so bili:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Vsi modeli uporabljajo isti trening in testni nabor.

## Evaluacija

Uporabljene metrike:

- MAE
- MSE
- RMSE
- R²

Kot glavna metrika za izbor modela je uporabljen MAE, ker neposredno pove povprečno napako v dolarjih.

## Rezultati

DOPOLNI PO IZVEDBI:

Najboljši model:

`IME MODELA`

MAE:

`XXX USD`

RMSE:

`XXX USD`

R²:

`X.XX`

Finalni model je bil izbran zato, ker je dosegel najnižji MAE.

## Zagon projekta

### 1. Namestitev knjižnic

```bash
pip install -r requirements.txt

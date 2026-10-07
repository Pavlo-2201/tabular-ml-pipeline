# tabular-ml-pipeline: прогноз оттока клиентов

Бинарная классификация оттока клиентов телеком-оператора (`Churn`). Полный цикл: воспроизводимая
загрузка данных, EDA, базовые модели, RandomForest и CatBoost, подбор параметров на
кросс-валидации, выбор порога, однократная оценка на test, анализ ошибок, обучение и прогноз
из командной строки.

## Итоги

Модели сравнивались 5-fold CV на train+val, лучшая выбиралась по PR-AUC:

| Модель | CV PR-AUC | CV ROC-AUC |
|---|---|---|
| Dummy (prior) | 0.2654 ± 0.0001 | 0.5000 ± 0.0000 |
| LogisticRegression | 0.6612 ± 0.0176 | 0.8459 ± 0.0083 |
| RandomForest | 0.6670 ± 0.0214 | 0.8470 ± 0.0095 |
| **CatBoost** (выбрана) | **0.6738 ± 0.0236** | **0.8502 ± 0.0089** |

Порог 0.35 выбран по максимуму F1 на out-of-fold предсказаниях. Test (1409 клиентов, 374 ушедших)
использован один раз:

| ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| 0.8447 | 0.6661 | 0.6300 | 0.5604 | 0.7193 |

- Модель находит 72% уходящих клиентов; из отмеченных моделью уходит 56%.
- Метрики test близки к CV, переобучения на выборе параметров не видно.
- Отрыв CatBoost меньше разброса между fold, но устойчив: на тех же fold он лучше
  LogisticRegression во всех пяти и лучше RandomForest в четырёх из пяти.
- Сильнее всего на отток влияют стаж, тип контракта и интернет-услуги. Ушедших клиентов
  с долгосрочным контрактом модель почти не находит: в данных нет признаков, которые их отличают.

## Данные

IBM Telco Customer Churn из [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d),
лицензия Apache-2.0: 7043 клиента, 19 признаков, доля оттока 26.5%. Файл скачивается по
фиксированному коммиту и сверяется по SHA-256; в Git данные не хранятся.

## Быстрый старт (PowerShell)

Python 3.13, CPU, GPU не нужен.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m churn.download_data            # скачать данные и проверить SHA-256
python train.py                          # подбор, порог, оценка на test, модель (~4 мин на 4 CPU)
python predict.py --input examples/new_customers.csv --output predictions.csv
pytest -q
```

## Структура

```
churn/                  код: данные, разбиения, предобработка, модели, CV, порог, инференс
configs/                конфигурации обучения (train.json, smoke.json)
train.py, predict.py    обучение и прогноз
notebooks/01_eda.ipynb  EDA
results/, reports/      метрики, предсказания, графики
tests/                  pytest
```

## Стек

Python 3.13, pandas, scikit-learn, CatBoost, matplotlib, Jupyter, pytest.

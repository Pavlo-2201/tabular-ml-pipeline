# tabular-ml-pipeline: прогноз оттока клиентов

Учебный проект: бинарная классификация оттока (`Churn`) на датасете IBM Telco Customer Churn.

## Данные

- Источник: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d),
  файл `data/Telco-Customer-Churn.csv`, коммит `d5371f5d83a446ad5673cbcca3b814b926491f8a`.
- Лицензия репозитория: Apache-2.0. Это официальный пример IBM, источник доступен (проверено 2026-01-12).
- SHA-256 файла: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- 7043 клиента, 21 столбец, доля оттока 26.5%.

Данные в Git не хранятся, их скачивает скрипт.

## Установка и запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m churn.download_data   # скачивание и проверка SHA-256
python -m churn.check_data      # проверка схемы, распределение цели -> results/data_summary.json
python -m churn.baselines       # разбиения, Dummy и LogisticRegression -> results/baselines.json
pytest -q
```

EDA: `notebooks/01_eda.ipynb` (выполненный, с выводами). Графики сохраняются в `reports/figures/`,
краткая сводка в `results/eda_summary.json`. Перезапуск без Jupyter-интерфейса:

```bash
python -c "import nbformat; from nbclient import NotebookClient; nb=nbformat.read('notebooks/01_eda.ipynb', 4); NotebookClient(nb, resources={'metadata': {'path': 'notebooks'}}).execute(); nbformat.write(nb, 'notebooks/01_eda.ipynb')"
```

## Решения этапа 1

- Файл скачивается по фиксированному коммиту, а не по ветке `master`, и сверяется по SHA-256:
  если источник изменится, скрипт остановится, а не тихо обучится на других данных.
- Схема проверяется явно: состав и типы столбцов, уникальность `customerID`, значения цели.
  `TotalCharges` в сыром файле строковый; разбор этого столбца отложен до EDA.

## Решения этапа 2 (EDA)

- NaN в сыром файле нет. `TotalCharges` строковый из-за 11 пустых значений, и все они у клиентов
  с `tenure == 0` (новые клиенты). Им ставится `TotalCharges = 0` построчным правилом в
  `churn.data.clean_data`; правило не использует статистик выборки, поэтому утечки нет.
- `TotalCharges` почти равен `tenure × MonthlyCharges` (корреляция с `tenure` 0.83). Признак
  оставлен, мультиколлинеарность для линейной модели гасится регуляризацией.
- `customerID` уникален и исключён из признаков (`churn.data.FEATURES`), он нужен только для
  отслеживания разбиений.
- Сильнее всего с оттоком связаны стаж, тип контракта, интернет-услуги (`OnlineSecurity`,
  `TechSupport`), способ оплаты; `gender` и `PhoneService` почти не влияют.
- Отток 26.5%: дисбаланс умеренный, поэтому дальше смотрим PR-AUC и подбираем порог.

## Решения этапа 3 (разбиения и базовые модели)

- Разбиение 60/20/20 (train/validation/test), стратифицировано по `Churn`, seed 42. Сохраняется
  списком `customerID` в `results/splits.csv`, и все скрипты берут его из файла: test один и тот же.
- Предобработка собрана в `Pipeline` + `ColumnTransformer` (`churn/features.py`): медианная
  импутация и `StandardScaler` для чисел, `most_frequent` + `OneHotEncoder(handle_unknown="ignore")`
  для категорий. Так статистики считаются только на train, а неизвестная категория на инференсе
  не ломает модель. Тесты проверяют, что медианы и средние совпадают со статистиками train.
- `DummyClassifier(strategy="prior")` задаёт нижнюю границу качества.

Результаты на validation (порог 0.5, test не использовался):

| Модель | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| Dummy | 0.5000 | 0.2654 | 0.0000 |
| LogisticRegression | 0.8361 | 0.6423 | 0.5900 |

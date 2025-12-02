"""
run_optuna_all_models.py

Запускает per-dataset Optuna-тюнинг для трёх моделей:
  - DSB
  - CTGAN
  - TabDDPM

и сохраняет лучшие параметры и метрики по каждому датасету в один файл:
  - optuna_best_params_all_models.pkl
  - optuna_best_params_all_models.json
"""

import json
import pickle
from pathlib import Path

from tqdm.auto import tqdm

from optuna_tune_tabular_generators import tune_model_per_dataset


def main():
    # Модели, которые тюним
    # MODELS = ["DSB", "CTGAN", "TabDDPM"]
    MODELS = [ "DSB"]

    # Количество trial'ов Optuna на каждый датасет/модель
    N_TRIALS = 30

    # Если None — используем все реальные датасеты,
    # определённые в prepare_all_datasets внутри optuna_tune_tabular_generators.
    # Можно ограничить, например: ["adult_numeric", "default_credit_card_numeric"]
    # TARGET_DATASETS = ["california_housing", "king_county_housing"]  
    TARGET_DATASETS = ['california_housing',"king_county_housing"]  

    # Optuna storage (для возобновляемости экспериментов и просмотра истории)
    # Можно оставить None, тогда всё будет в памяти
    STORAGE = "sqlite:///optuna_tabular_dsb.db"

    # Куда сохраняем итоговые best-параметры и метрики
    out_pkl = Path("optuna_best_params_all_models.pkl")
    out_json = Path("optuna_best_params_all_models.json")

    all_results = {}

    # tqdm по моделям — видно, на какой модели сейчас идёт тюнинг
    for model_name in tqdm(MODELS, desc="Models", position=0):
        print(f"\n##### TUNING MODEL: {model_name} #####")

        # Внутри tune_model_per_dataset есть лог:
        #   === Tuning {MODEL} on dataset '{ds_name}' ===
        # так что по консоли всегда видно и модель, и датасет.

        per_model_results = tune_model_per_dataset(
            model_name=model_name,
            n_trials=N_TRIALS,
            target_datasets=TARGET_DATASETS,
            use_pca=False,
            test_size=0.2,
            random_state=42,
            storage=STORAGE,
            direction="minimize",
            save_path=None,  # здесь не сохраняем по модели, всё соберём в общий файл ниже
        )

        # Для сериализации выбрасываем объект study, оставляем только best_params + метрики
        compact_per_model = {}
        for ds_name, info in per_model_results.items():
            compact_per_model[ds_name] = {
                "best_params": info["best_params"],
                "best_value": float(info["best_value"]),
                "metrics": {
                    "swd_val": float(info["metrics"]["swd_val"]),
                    "mmd_val": float(info["metrics"]["mmd_val"]),
                    "score": float(info["metrics"]["score"]),
                },
            }

        all_results[model_name] = compact_per_model

    # --------- сохраняем в pickle ---------
    with out_pkl.open("wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved best params & metrics for all models to: {out_pkl}")

    # --------- сохраняем в JSON (для удобного просмотра) ---------
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON summary to: {out_json}")


if __name__ == "__main__":
    main()

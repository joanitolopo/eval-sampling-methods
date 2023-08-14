# Standard library imports
import gc
from time import sleep

# Third-party library imports
import joblib
import pandas as pd

# Coustom utility function imports
from utils import get_models_folder_path, plot

import argparse
from processing import Processing
from optimizing import OptunaStudy
from logger import logging

class RunModel:
    def __init__(self, name, file_path):
        self.name = name
        self.file_path = file_path
        # Other initialization code

def load_data(path, target_col="target"):
    df = pd.read_csv(path)

    X  = df.drop(columns=[target_col])
    y = df["target"]

    class_distribution = y.value_counts(normalize=True)
    minority_class_percentage = class_distribution.min() / class_distribution.sum()
    if minority_class_percentage > 0.4:
        raise ValueError("Error: Data is Balanced.")

    return X, y

def main():
    parser = argparse.ArgumentParser(description="Run Model with Command-Line Arguments")
    parser.add_argument("--file", required=True, help="CSV File")
    parser.add_argument("--sampling", default="ros", help="Sampling method (default: ros)")
    parser.add_argument("--ratio", type=float, default=0.5, help="Sampling Ratio")
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument("--custom_param_path", help="Custom your own hyperparameter values to be tuned")
    args = parser.parse_args()

    # load model
    X, y = load_data(args.file)

    # Processing data
    processed_data = Processing(X, y)
    processed_data.split_data(0.8)
    processed_data.sampling(args.sampling, args.ratio)

    processed_data.X, processed_data.y = None, None
    del X, y
    gc.collect()
    sleep(3)

    # Hyperparameter Tuning
    if args.custom_param_path is not None:
        study_results = OptunaStudy(processed_data.X_train, processed_data.y_train, 
                                    processed_data.X_test, processed_data.y_test, args.sampling, args.ratio,
                                    args.custom_param_path)
        best_trial = study_results.best_trial.params
    else:
        study_results = OptunaStudy(processed_data.X_train, processed_data.y_train, 
                                    processed_data.X_test, processed_data.y_test, args.sampling, args.ratio)
        best_trial = study_results.best_trial.params

    # Modelling
    model_results = processed_data.prep_run_model(best_trial)

    # save to files for reuse later
    model_dir = get_models_folder_path("model")
    model_results['xgb_model'].save_model(f'{model_dir}/model_{args.sampling}_{args.ratio}.json')
    joblib.dump(model_results, f"{model_dir}/model_results_{args.sampling}_{args.ratio}.dict")
    logging.info(f"Model saved in {model_dir}")
    if args.plot:
        metrics_table = plot(model_results['xgb_model'], processed_data.X_train, processed_data.y_train, processed_data.X_test, processed_data.y_test)
        logging.info("\nMetrics Summary:\n" + metrics_table)

if __name__=="__main__":
   main()
    
















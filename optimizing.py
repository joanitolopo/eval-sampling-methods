from processing import Processing
from IPython.display import clear_output
from logger import logging
import psutil
import os
import joblib
import json
import gc
from xgboost import XGBClassifier
from time import sleep
import optuna
from sklearn.datasets import make_classification
from utils import get_models_folder_path

# Additional sklearn metric imports
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    average_precision_score
    )
from imblearn.metrics import geometric_mean_score


class OptunaInstance(Processing):
    def __init__(self, X_train, y_train, X_test, y_test, custom_param_file=None):
        super().__init__(X_train, y_train)
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.custom_param_file = custom_param_file

        if self.custom_param_file:
            logging.warning("You are using custom hyperparameter values")
            self.load_custom_params()
        else:
            self.param = self.get_default_params()

        self.OPT_DIR_SAVE = get_models_folder_path("optuna")

    def get_ram(self):
        return psutil.virtual_memory().percent
    
    def save_custom_params(self):
        with open(self.custom_param_file, 'w') as json_file:
            json.dump(self.param, json_file)
    
    def load_custom_params(self):
        with open(self.custom_param_file, 'r') as json_file:
            self.param = json.load(json_file)

    def save_study(self, study, frozen_trial):
        study_path = os.path.join(self.OPT_DIR_SAVE, "xgb_optuna_study_callbacks.pkl")
        joblib.dump(study, study_path)
    
    def get_default_params(self):
        default_param = {
            'lambda': (1e-3, 10.0),
            'alpha': (1e-3, 10.0),
            'gamma':([0,1,5]),
            'objective': (['binary:logistic']),
            'min_child_weight': (1, 10),
            'reg_alpha': (0, 1),
            'reg_lambda': (0, 1),
            "scale_pos_weight": (1, 10),
            # 'max_features': ('max_features', ['auto', 'sqrt', 'log2']),
            'colsample_bytree': ([0.6,0.7,0.8,0.9,1.0]),
            'subsample': ([0.6,0.7,0.8,0.9,1.0]),
            'learning_rate': ([0.008, 0.01, 0.03, 0.05, 0.07, 0.1]),
            'max_depth': ([9,11,13]),
            'random_state': ([48]),
            # 'eval_metric': trial.suggest_categorical('eval_metric', [['auc','error']]),
        }
        return default_param

    # def get_models_folder_path(self):
    #     current_folder = os.path.dirname(os.path.abspath(__file__))
    #     models_folder_path = os.path.join(current_folder, "optuna")
    #     if not os.path.exists(models_folder_path):
    #         os.makedirs(models_folder_path)
    #     return models_folder_path

    def logging_callback(self, study, frozen_trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            clear_output(wait=True)
            self.text_out = f"Trial {frozen_trial.number} done with best value: {frozen_trial.value} and parameters: {frozen_trial.params}."
            logging.info(self.text_out)
        
        # Writting to file
        with open(f"{self.OPT_DIR_SAVE}/xgb_optuna_study_log.txt", "a") as os_log:
            os_log.write('\n\n')
            os_log.write(f"Current Ram Used: {self.get_ram()} %\n")
            os_log.write(self.text_out)
    
    def objective(self, trial, n_estimators=500, tree_method="hist"):
        nn_early_stopping_rounds = n_estimators * 0.1
        # access the hyperparamater
        lambda_range = self.param.get('lambda', (1e-3, 10.0))
        alpha_range = self.param.get('alpha', (1e-3, 10.0))
        gamma_range = self.param.get('gamma', [0, 1, 5])
        min_child_weight_range = self.param.get('min_child_weight', (1, 10))
        reg_alpha_range = self.param.get('reg_alpha', (0, 1))
        reg_lambda_range = self.param.get('reg_lambda', (0, 1))
        scale_pos_weight_range = self.param.get('scale_pos_weight', (1, 10))
        colsample_bytree_range = self.param.get('colsample_bytree', [0.6,0.7,0.8,0.9,1.0])
        subsample_range = self.param.get('subsample', [0.6,0.7,0.8,0.9,1.0])
        learning_rate_range = self.param.get('learning_rate', [0.008, 0.01, 0.03, 0.05, 0.07, 0.1])
        max_depth_range = self.param.get('max_depth', [9,11,13])

        param = {
            'tree_method':trial.suggest_categorical('tree_method', [tree_method]),
            'lambda': trial.suggest_float('lambda', *lambda_range),
            'alpha': trial.suggest_float('alpha', *alpha_range),
            'gamma':trial.suggest_categorical('gamma', gamma_range),
            'objective': trial.suggest_categorical('objective', ['binary:logistic']),
            'min_child_weight': trial.suggest_int('min_child_weight', *min_child_weight_range),
            'reg_alpha': trial.suggest_float('reg_alpha', *reg_alpha_range),
            'reg_lambda': trial.suggest_float('reg_lambda', *reg_lambda_range),
            "scale_pos_weight": trial.suggest_float('scale_pos_weight', *scale_pos_weight_range),
            # 'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree',colsample_bytree_range),
            'subsample': trial.suggest_categorical('subsample', subsample_range),
            'learning_rate': trial.suggest_categorical('learning_rate',learning_rate_range),
            'n_estimators': trial.suggest_categorical('n_estimators',[n_estimators]),
            'max_depth': trial.suggest_categorical('max_depth', max_depth_range),
            'random_state': trial.suggest_categorical('random_state', [48]),
            'early_stopping_rounds': trial.suggest_categorical('early_stopping_rounds',[nn_early_stopping_rounds]),
            # 'eval_metric': trial.suggest_categorical('eval_metric', [['auc','error']]),
        }
        
        if self.get_ram() >= 90:
            raise MemoryError("Short On Memory")
        
        if trial.number > 0:
            clear_output(wait=True)
            gc.collect()
            logging.info(self.text_out)

        model_xgbc = XGBClassifier(**param)

        eval_setparam = [(self.X_test, self.y_test)]

        logging.info(f"Current Ram Used: {self.get_ram()} %")
        model_xgbc.fit(self.X_train, self.y_train, eval_set=eval_setparam, verbose=False)
        preds = model_xgbc.predict(self.X_test)
        
        roc_auc = roc_auc_score(self.y_test, preds)
        g_mean = geometric_mean_score(self.y_test, preds)
        precision, recall, f1_score, _ = precision_recall_fscore_support(self.y_test, preds, average='macro')
        pr_auc = average_precision_score(self.y_test, preds, average='macro')

        # the decision metric
        max_auc_gmean=max(roc_auc, g_mean)
        score = max_auc_gmean * f1_score

        trial.report(score, 1)

        if trial.should_prune():
            text_prune = f'Trial {trial.number} pruned'
            # Writing to file
            with open(f"{self.OPT_DIR_SAVE}/xgb_optuna_study_log.txt", "a") as os_log:
                os_log.write('\n')
                os_log.write(text_prune)
            del model_xgbc, preds, text_prune
            gc.collect()
            sleep(3)
            raise optuna.TrialPruned()
        
        text_dtl = f"Trial {trial.number} finished with parameters: {trial.params}"
        # Writing to file
        with open(f"{self.OPT_DIR_SAVE}/xgb_optuna_study_log.txt", "a") as os_log:
            os_log.write('\n\n')
            os_log.write(f"Current Ram Used: {self.get_ram()} %")
            os_log.write(text_dtl)

        return score
    
def OptunaStudy(X_train, y_train, X_test, y_test, sampling, ratio, custom_param_file=None):
    ot = OptunaInstance(X_train, y_train, X_test, y_test,  custom_param_file)
    # ot.split_data(0.8)
    # ot.sampling(sampling=sampling, ratio=ratio)
    
    # ot.X, ot.y = None, None
    # del X, y
    # gc.collect()
    # sleep(3)

    nn_trials = 10
    nn_estimators = 100

    if os.path.exists(f'{ot.OPT_DIR_SAVE}/xgb_optuna_study_log_{sampling}_{ratio}.txt'):
        os.remove(f'{ot.OPT_DIR_SAVE}/xgb_optuna_study_log_{sampling}_{ratio}.txt')

    optuna.logging.set_verbosity(optuna.logging.WARN)
    logging.info("===================HYPERPARAMETER OPTIMIZING====================")
    logging.info(f"Please wait, finding best trial ...")
    study = optuna.create_study(direction="maximize")

    try:
        # callbacks [self.save_study] is to save study in case memory fails
        study.optimize(lambda trial: ot.objective(trial, n_estimators = nn_estimators),
                        n_trials = nn_trials,
                        callbacks = [ot.logging_callback, ot.save_study],
                        gc_after_trial = True,
                        catch = (RuntimeWarning,ArithmeticError,))
    except MemoryError as e:
        logging.info(f'{e} : Memory was getting low, Trial ended early')


    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    logging.info(f'Number of finished trials: {len(study.trials)}')
    logging.info(f'Number of pruned trials: {len(pruned_trials)}')
    logging.info(f'Number of completed trials: {len(complete_trials)}')
    logging.info(f'Best trial: {study.best_trial.params}')

    joblib.dump(study, f"{ot.OPT_DIR_SAVE}/xgb_optuna_study_{sampling}_{ratio}.pkl")

    return study

if __name__ == "__main__":
    # Create an instance of the Optuna class
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, weights=[0.9, 0.1], random_state=42)
    study_results = OptunaStudy(X, y, "sm", 0.5)

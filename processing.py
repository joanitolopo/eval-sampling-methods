from logger import logging
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

# Third-party library imports for imbalanced data handling
from imblearn.over_sampling import(
    SMOTE,
    RandomOverSampler,
    ADASYN,
    )

from imblearn.under_sampling import (
    RandomUnderSampler,
    InstanceHardnessThreshold
    )

# Resampling type to the corresponding class
resampling_methods = {
    "sm":SMOTE,
    "rus": RandomUnderSampler,
    "iht":InstanceHardnessThreshold,
    "ads": ADASYN
}


class Processing:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.xgb_model = None
       
    def log_class_distribution(self, y):
        class_counts = Counter(y)
        logging.info(f'Before Resampling -> 1: {class_counts[1]}, 0: {class_counts[0]}')

    def split_data(self, train_size = 0.7):
        # Split Data into Train:Test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y, train_size = train_size, random_state=48) 
        
        return {'X_train':self.X_train, 'y_train':self.y_train,
                'X_test':self.X_test, 'y_test':self.y_test}

    def sampling(self, sampling, ratio, print_info=True):
        logging.info("=========================Sampling=============================")
        if sampling in resampling_methods:
            ResamplerClass = resampling_methods[sampling]
            resample = ResamplerClass(sampling_strategy=ratio, random_state=42)
        else:
            resample = RandomOverSampler(sampling_strategy=ratio, random_state=42)

        if print_info == True:
            logging.info('X_train size : %s', self.X_train.shape)
            logging.info('X_test size  : %s', self.X_test.shape)

        logging.warning('Please wait, resampling train data')
        logging.info(f'Before Resampling -> 1: {Counter(self.y_train)[1]}, 0: {Counter(self.y_train)[0]}')
        X_train_sampled, y_train_sampled = resample.fit_resample(self.X_train, self.y_train)

        if print_info:
            logging.info('After Resampling -> 1 : {}, 0 : {}'.format(
                Counter(y_train_sampled)[1], Counter(y_train_sampled)[0]))
        
        self.X_train, self.y_train = X_train_sampled, y_train_sampled
    
    def prep_run_model(self, params, verbose=False):
        logging.info("=========================Training=============================")
        self.xgb_model = XGBClassifier(**params)
        logging.info('Training XGBoost Model with Parameters: %s', params)
        self.xgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=verbose)
        predictions = self.xgb_model.predict(self.X_test)

        return {'xgb_model':self.xgb_model,'predictions':predictions,
                'X_train':self.X_train, 'y_train':self.y_train,
                'X_test':self.X_test, 'y_test':self.y_test}
 
   
if __name__=="__main__":
    # Create an instance of the Processing class with imbalanced data
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, weights=[0.9, 0.1], random_state=42)
    processing_instance = Processing(X, y)

    # Run the test methods on the instance
    X_train, y_train, X_test, y_test = processing_instance.split_data()
    processing_instance.sampling()
    processing_instance.prep_run_model()
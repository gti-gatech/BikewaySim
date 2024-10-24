import pickle

from bikewaysim.impedance_calibration import post_calibration
from bikewaysim.paths import config

if __name__ == '__main__':
    
    # import testing folds
    with (config['calibration_fp']/'validation/testing_folds.pkl').open('rb') as fh:
        testing_folds = pickle.load(fh)
    
    def model_fp(fold_num):
        return config['calibration_fp'] / f'results/{fold_num},validation,0.pkl'

    testing_folds = [(model_fp(fold_num),tripids) for fold_num, tripids in testing_folds]

    # perform shortest path routing and calculate the loss values for the testing trips
    post_calibration.validation_workflow(testing_folds)
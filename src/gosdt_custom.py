'''
To `pip install gosdt`, follow these steps:

1. Open: nano ~/.bashrc
2. Add this variable: export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=true
3. Run this: source ~/.bashrc
4. pip install gosdt

'''
from gosdt.model.threshold_guess import compute_thresholds, cut
from gosdt.model.gosdt import GOSDT

import json
import pathlib
import itertools

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

class CustomGOSDT(GOSDT):
    '''
    Inherits GOSDT and adds more convenience methods.
    '''
    def __init__(self):
        self.thresholds = None
        self.np_thresholds = None
        self.header = None
        self.config = None

    def _config(self, config_dict):
        self.config = config_dict
        pass

    def confusion_matrix(self, preds, y):
        '''
        Only use this function for 0-1 labels.
        '''
        true_positives = (preds + y == 2).sum()
        true_negatives = (preds + y == 0).sum()

        negative_positions = y == 0
        false_positives = (preds[negative_positions] != y[negative_positions]).sum()

        positive_positions = y == 1
        false_negatives = (preds[positive_positions] != y[positive_positions]).sum()

        matrix = np.array([[true_positives, false_positives],
                            [false_negatives, true_negatives]])

        assert matrix.flatten().sum() == y.shape[0], \
        f"Number of elements in Conf Matrix ({matrix.flatten().sum()}) don't match the number of elements in y ({y.shape[0]})."

        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        f1 = 2 * ((precision * recall)/(precision + recall))
        accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)

        tpr = recall
        fpr = false_positives/(false_positives + true_negatives)

        return {'Confusion Matrix': matrix,
                'Precision': precision,
                'Recall': recall,
                'TPR': tpr,
                'FPR': fpr,
                'F1 Score': f1,
                'Accuracy': accuracy}

    def _compute_data_thresholds(self, X, y, n_est, max_depth):
        # n_est is the number of estimators for the boosted tree which is, in turn, used
        # to calculate the thresholds for each feature
        return compute_thresholds(X, y, n_est, max_depth)

    def fit_tree(self, X, y, n_est, max_depth, verbose):
        """
        Does the required preprocessing to fit the GOSDT tree including:
        - computing thresholds for continuous variables; it also saves these
          so that they could be used internally on the test set when using
          predict_labels() method.
        - computing lower bounds using GradientBoostingClassifier.
        """
        # compute thresholds and save them
        X_binarized, thresholds, header, _ = self._compute_data_thresholds(X.copy(), y.copy(), n_est, max_depth)
        self.thresholds = list(X_binarized.columns.values)
        self.np_thresholds = thresholds
        self.header = header

        # guess lower bound
        clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
        clf.fit(X_binarized, y.values.flatten())
        warm_labels = clf.predict(X_binarized)

        # save the labels from lower bound guesses as a tmp file and return the path to it.
        labelsdir = pathlib.Path('/tmp/warm_lb_labels')
        labelsdir.mkdir(exist_ok=True, parents=True)
        labelpath = labelsdir / 'warm_label.tmp'
        labelpath = str(labelpath)
        pd.DataFrame(warm_labels, columns=["class_labels"]).to_csv(labelpath, header="class_labels",index=None)

        # add warm labels to config
        self.config['warm_LB'] = True
        self.config['path_to_labels'] = labelpath

        # initialize and fit the GOSDT parent class
        super().__init__(self.config)
        self.fit(X_binarized, y)

        if verbose == 1:
            train_acc = self.score(X_binarized, y)
            n_leaves = self.leaves()
            time = self.utime
            print("Model training time: {}".format(time))
            print("Training accuracy: {:.1%}".format(train_acc))
            print("# of leaves: {}".format(n_leaves))
            # print("Splits: {}".format(X_binarized.columns))
            pass

        elif verbose == 2:
            train_acc = self.score(X_binarized, y)
            n_leaves = self.leaves()
            n_nodes = self.nodes()
            time = self.utime
            print("Model training time: {}".format(time))
            print("Training accuracy: {:.1%}".format(train_acc))
            print("# of leaves: {}".format(n_leaves))
            print(self.tree)
            # print("Splits: {}".format(X_binarized.columns))
            pass

        else:
            pass
        return None

    def use_computed_thresholds(self, df):
        '''
        Function to fit the thresholds calculated by the fit_tree() method on
        the test data and return the modified binarized dataframe.
        '''
        # construct dict of thresholds for each column
        cut_dct = {}
        for cut in self.thresholds:
            col, cut_thresh = cut.split('<=')
            if col in cut_dct:
                cut_dct[col].append(cut_thresh)
                pass
            else:
                cut_dct[col] = [cut_thresh]
                pass

        # binarize the df according to computed thresholds
        binarized_df = pd.DataFrame()
        for col in cut_dct:
            if len(cut_dct[col]) == 1:
                thresh = float(cut_dct[col][0])
                binarized_col_values = (df[col] <= thresh).astype('int')
                binarized_col_name = col + '<=' + str(thresh)
                binarized_col_df = pd.Series(binarized_col_values, name=binarized_col_name)
                binarized_df = pd.concat([binarized_df, binarized_col_df],
                                        axis='columns',
                                        ignore_index=False)
            else:
                for thresh_str in cut_dct[col]:
                    thresh = float(thresh_str)
                    binarized_col_values = (df[col] <= thresh).astype('int')
                    binarized_col_name = col + '<=' + str(thresh)
                    binarized_col_df = pd.Series(binarized_col_values, name=binarized_col_name)
                    binarized_df = pd.concat([binarized_df, binarized_col_df],
                                            axis='columns',
                                            ignore_index=False)

        binarized_df.reset_index(drop=True, inplace=True)
        return binarized_df

    def predict_labels(self, X):
        """
        Compute the thresholds and predict.
        """
        # X_binarized = self.use_computed_thresholds(X)
        X_binarized = cut(X.copy(), self.np_thresholds)
        X_binarized = X_binarized[self.header]

        return self.predict(X_binarized)


class GridSearchCVGOSDT:
    '''
    Takes CustomGOSDT and adds k-fold CV functionality to it.
    '''
    def __init__(self, model: CustomGOSDT, param_dict: dict, metric: str, folds: int):
        '''
        model: uninstantiated CustomGOSDT
        param_dict: hyperparameters for GOSDT
        metric: one of 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'TPR', 'FPR'
        folds: number of folds 'k'
        '''
        self.folds = folds
        self.param_dict = param_dict
        self.model = model
        self.metric = metric

        assert self.metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'TPR', 'FPR']

        # create parameter grid (permutations)
        keys, values = zip(*param_dict.items())
        self.param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]

    def confusion_matrix(self, preds, y):
        '''
        Only use this function for 0-1 labels.
        '''
        true_positives = (preds + y == 2).sum()
        true_negatives = (preds + y == 0).sum()

        negative_positions = y == 0
        false_positives = (preds[negative_positions] != y[negative_positions]).sum()

        positive_positions = y == 1
        false_negatives = (preds[positive_positions] != y[positive_positions]).sum()

        matrix = np.array([[true_positives, false_positives],
                           [false_negatives, true_negatives]])

        assert matrix.flatten().sum() == y.shape[0], f"Number of elements in Conf Matrix ({matrix.flatten().sum()}) don't match the number of elements in y ({y.shape[0]})."

        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        f1 = 2 * ((precision * recall)/(precision + recall))
        accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)

        tpr = recall
        fpr = false_positives/(false_positives + true_negatives)


        return {'Confusion Matrix': matrix,
                'Precision': precision,
                'Recall': recall,
                'TPR': tpr,
                'FPR': fpr,
                'F1 Score': f1,
                'Accuracy': accuracy}

    def run_validation(self, X: pd.DataFrame, y: pd.Series, n_est: int, max_depth: int, verbose: int) -> dict:
        '''
        Workhorse function that runs the k-fold CV.

        X: data
        y: labels - 0/1
        n_est: number of estimators for guessing thresholds
        max_depth: max tree depth for the estimator that computes thresholds, not the GOSDT.

        Returns: dict with best parameters and corresponding metric score.
        '''

        data = pd.concat([X, y], axis='columns')
        split_data_list = np.array_split(data, self.folds)

        model_param_lst = []
        all_model_param_lst = []

        # outer loop (generating folds)
        for i in range(0, self.folds):

            print(f'\n\nFold: {i+1}')
            print('='*100)

            # regenerate array every time
            split_data_list = np.array_split(data, self.folds)

            val_set = split_data_list[i]
            split_data_list.pop(i)
            train_set = pd.concat(split_data_list)

            params_metric_list = []

            # inner loop (going through all parameter combinations)
            for params in self.param_grid:

                # model
                model = self.model()
                model._config(params)

                model.fit_tree(X=train_set.iloc[:, :-1],
                                y=train_set.iloc[:, -1],
                                n_est=n_est,
                                max_depth=max_depth,
                                verbose=verbose)

                preds = model.predict_labels(val_set)

                val_result = self.confusion_matrix(preds, val_set.iloc[:, -1])
                param_dict_with_result = {**params, self.metric: val_result[self.metric]}
                params_metric_list.append(param_dict_with_result)
                print(param_dict_with_result)

                all_model_param_lst.append(param_dict_with_result)
                pass

            best_params_dict = max(params_metric_list, key=lambda x:x[self.metric])
            model_param_lst.append(best_params_dict)
            print('\nBest Result:')
            print(best_params_dict)
            pass

        all_models_df = pd.DataFrame(all_model_param_lst)
        average_scores_df = all_models_df.groupby([col for col in all_models_df.columns if col not in [self.metric]]).mean()
        average_scores_df.rename(columns={self.metric: 'Mean ' + self.metric},
                                 inplace=True)
        best_results_dict = average_scores_df.reset_index().max()

        best_param_dict = best_results_dict.copy()
        best_param_dict.pop('Mean ' + self.metric)
        best_param_dict = best_param_dict.to_json()

        result_dict = {'average_scores': average_scores_df,
                       'best_params': json.loads(best_param_dict),
                       'best_result': best_results_dict,
                       'all_models': all_models_df}
        
        return result_dict


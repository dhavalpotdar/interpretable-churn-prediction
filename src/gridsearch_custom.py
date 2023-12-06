import itertools
import numpy as np
import pandas as pd

class KFoldCrossValidator:
    '''
    Generic class that can do k-fold cross validation with hyperparameter selection
    for the given metric.
    '''
    def __init__(self, model: object, param_dict: dict, metric: str, folds: int):
        '''
        model: uninstantiated model object that has a predict and fit method.
        param_dict: dict, parameters to choose from.
        metric: one of 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'TPR', 'FPR'.
        folds: int, number of folds (k)
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

    def run_validation(self, data: pd.DataFrame) -> dict:
        '''
        Workhorse function for k-fold CV with hyperparameter selection.
        data: pd.DataFrame, the last column must have the target variable.

        Returns: dict with best parameters and corresponding metric score.
        '''
        model_param_lst = []
        all_model_param_lst = []

        # outer loop (generating folds)
        for i in range(0, self.folds):

            print(f'\n\nFold: {i+1}')
            print('='*100)
            # regenerate array every time
            split_data_list = np.array_split(data.values, self.folds)
            val_set = split_data_list[i]
            split_data_list.pop(i)
            train_set = np.concatenate(split_data_list)

            params_metric_list = []
            
            # inner loop (going through all parameter combinations)
            for params in self.param_grid:

                # model 
                model = self.model(**params)
                model.fit(train_set[:, :-1], train_set[:, -1])

                # predict 
                preds = model.predict(val_set[:, :-1])
                preds = np.array(preds, dtype='float')

                val_result = self.confusion_matrix(preds, val_set[:, -1])
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
        
        return {'best_overall': max(model_param_lst, key=lambda x:x[self.metric]), 
                'best_k': model_param_lst,
                'all_models': all_model_param_lst}


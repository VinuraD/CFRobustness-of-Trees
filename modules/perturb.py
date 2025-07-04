import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm

class Perturbation():
    def __init__(self,data_module, model=None, data=None, perturb_type=None, perturb_hp=None):
        self.model = model
        self.data = data
        self.perturb_type = perturb_type
        self.perturb_hp = perturb_hp
        self.data_module = data_module
    
    def get_data(self,fold):
        train,val,test = self.data_module.get_data(fold) #change based on data module
        if not isinstance(train, pd.DataFrame):
            print("Warning: a DataFrame is expected, but got:", type(train))

        return train,val,test
    
    def perturb_data(self,perturb_type,perturb_hp,data):
        if perturb_type=='data_add_minor':
            perturbed_data=self._data_add(data,perturb_hp)
        elif perturb_type=='data_add_major':
            assert perturb_hp['total_bin']==2
            perturbed_data=self.data_add(data,perturb_hp)
        elif perturb_type=='data_remove_minor':            
            assert perturb_hp['total_bin']==2
            perturbed_data=self.data_remove(data,perturb_hp)
        elif perturb_type=='data_remove_major':
            perturbed_data=self.data_remove(data,perturb_hp)

        return perturbed_data
    
    def perturb_model(self,perturb_type,perturb_hp):
        if perturb_type=='n_est':
            perturbed_model=self.model_change_n_est(perturb_hp)
        elif perturb_type=='max_depth':
            perturbed_model=self.model_change_max_depth(perturb_hp)

        return perturbed_model
    
    @staticmethod
    def data_add(data,perturb_hp): #here, data will be either val or val+train
        bin_num= perturb_hp['bin_num']
        assert bin_num>0, "bin_num should be greater than 0"
        total_bin=perturb_hp['total_bin']
        amount=int(bin_num*data.shape[0]/ total_bin)
        if amount> data.shape[0]:
            amount=data.shape[0]-2
        new_data=data.iloc[2:amount]

        return new_data
    
    @staticmethod
    def data_remove(data,perturb_hp):
        bin_num= perturb_hp['bin_num']
        assert bin_num>0, "bin_num should be greater than 0"
        total_bin=perturb_hp['total_bin']
        amount=int(bin_num*data.shape[0]/ total_bin)
        if amount> data.shape[0]:
            amount=data.shape[0]-2
        new_data=data.iloc[2:amount,:]

        return new_data
    
    @staticmethod
    def model_change_n_est(self,model,perturb_hp):
        assert perturb_hp['n_est'] is not None, "n_est should not be None"
        n_est = perturb_hp['n_est']

        if isinstance(model, xgb.XGBClassifier):
            model.set_params(n_estimators=n_est)
        elif isinstance(model, RandomForestClassifier):
            model.set_params(n_estimators=n_est)
        elif isinstance(model, AdaBoostClassifier):
            model.set_params(n_estimators=n_est)
        elif isinstance(model, MLPClassifier):
            print("Warning: MLPClassifier does not have n_estimators parameter, skipping.")
        elif isinstance(model, lightgbm.LGBMClassifier):
            model.set_params(n_estimators=n_est)
        else:
            raise ValueError("Unsupported model type for n_estimators change")

        return model

    @staticmethod
    def model_change_max_depth(self,model,perturb_hp):
        assert perturb_hp['max_depth'] is not None, "max_depth should not be None"
        max_depth=perturb_hp['max_depth']
        
        if isinstance(model, xgb.XGBClassifier):
            model.set_params(max_depth=max_depth)
        elif isinstance(model, RandomForestClassifier):
            model.set_params(max_depth=max_depth)
        elif isinstance(model, AdaBoostClassifier):
            print("Warning: AdaBoostClassifier does not have max_depth parameter, skipping.")
        elif isinstance(model, MLPClassifier):
            print("Warning: MLPClassifier does not have max_depth parameter, skipping.")
        elif isinstance(model, lightgbm.LGBMClassifier):
            model.set_params(max_depth=max_depth)
        else:
            raise ValueError("Unsupported model type for max_depth change")

        return model
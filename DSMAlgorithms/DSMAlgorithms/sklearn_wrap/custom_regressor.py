import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import inspect


def get_constructor_param_type_enhanced(cls, param_name):
    """
    增强版参数类型获取函数
    """
    if cls is None:
        return None

    try:
        sig = inspect.signature(cls.__init__)
        param = sig.parameters.get(param_name)

        if not param:
            return None

        # 优先使用类型注解
        if param.annotation != inspect.Parameter.empty:
            return param.annotation

        # 其次从默认值推断
        if param.default != inspect.Parameter.empty:
            return type(param.default)

        return None
    except Exception:
        return None

'''
自定义回归器的sklearn包装器
'''


class CustomRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        new_kwargs = kwargs.copy()
        model_class = new_kwargs.pop('model_class', None)

        for key, value in new_kwargs.items():
            param_type = get_constructor_param_type_enhanced(model_class, key)
            if param_type == int:
                new_kwargs[key] = int(value)
            elif param_type == float:
                new_kwargs[key] = float(value)
            elif param_type == bool:
                new_kwargs[key] = round(value)==1  # 取值范围从1-2，近似1代表true,近似2代表false
        self.regressor = model_class(**new_kwargs)

    def __sklearn_tags__(self):  # 重要，否则堆叠过程中不认这个回归器
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    '''
    X:所有环境协变量的值
    y：预测变量的值
    '''

    def fit(self, train_X: pd.DataFrame, train_y: np.ndarray):
        fit_method = getattr(self.regressor, 'fit')
        fit_method(train_X, train_y)
        return self

    def predict(self, X: pd.DataFrame):
        predict_method = getattr(self.regressor, 'predict')
        return predict_method(X)

    def get_params(self, deep=True):
        new_dict ={}
        for key, value in self.kwargs.items():
            new_dict[key] = value
        return new_dict

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Scaling functions for the pyforce library.
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

from .functions_list import FunctionsList
from sklearn.preprocessing import StandardScaler as sk_StandardScaler, MinMaxScaler as sk_MinMaxScaler
import numpy as np

class ScalerWrapper:
    """
    Base class for scaling functions in the pyforce library.
    """

    def __init__(self):
        self._scaler = None

    def fit(self, train_X: FunctionsList, **kwargs):
        """
        Fit the scaler to the data.

        Args:
            train_X (FunctionsList): The data to fit the scaler to.
        """
        pass

    def transform(self, X: FunctionsList):
        """
        Transform the data using the fitted scaler.

        Args:
            X (FunctionsList): The data to transform.

        Returns:
            FunctionsList: The transformed data.
        """
        pass

    def inverse_transform(self, X: FunctionsList):
        """
        Inverse transform the data using the fitted scaler.

        Args:
            X (FunctionsList): The data to inverse transform.

        Returns:
            FunctionsList: The inverse transformed data.
        """
        pass

class StandardScaler(ScalerWrapper):
    """
    Scaler that centers the data to have zero mean and unit variance (if `with_std` is True).

    Parameters
    ----------
    wrt_ic : bool, optional
        If True, the scaling is done with respect to the initial condition - element 0 of the input data (default is False).
    global_scale : bool, optional
        If True, the scaling is done globally across all data (default is False).
    with_std : bool, optional
        If True, the data is scaled to have unit variance (default is True).

    """

    def __init__(self, wrt_ic: bool = False, global_scale: bool = False, with_std: bool = True):
        super().__init__()
        self.wrt_ic = wrt_ic
        self.global_scale = global_scale
        self.with_std = with_std

    def fit(self, train_X: FunctionsList):
        """
        Fit the StandardScaler to the training data.

        Args:
            train_X (FunctionsList): The training data to fit the scaler to.
        """

        if self.wrt_ic:
            self._mean = np.ones((train_X.fun_shape,)) * train_X(0).mean()
            self._std = np.ones((train_X.fun_shape,)) * train_X(0).std() if self.with_std else None
        elif self.global_scale:
            self._mean = np.ones((train_X.fun_shape,)) * train_X.mean()
            self._std = np.ones((train_X.fun_shape,)) * train_X.std() if self.with_std else None
        else:
            self._mean = train_X.mean(axis=1)
            if self.with_std:
                if train_X.return_matrix().std(axis=1).any() == 0:
                    raise ValueError("Standard deviation is zero for some features. Cannot scale to unit variance.")
                else:
                    self._std = train_X.std(axis=1)
            else:
                self._std = None
        

    def transform(self, X: FunctionsList):
        """
        Transform the data using the fitted StandardScaler.

        Args:
            X (FunctionsList): The data to transform.

        Returns:
            FunctionsList: The transformed data.
        """

        _new_X = FunctionsList(X.fun_shape)
        
        for ii in range(len(X)):
            _new_X.append( 
                (X(ii) - self._mean) / (self._std+1e-12 if self.with_std else 1)
            )

        return _new_X
    
    def inverse_transform(self, X: FunctionsList):
        """
        Inverse transform the data using the fitted StandardScaler.

        Args:
            X (FunctionsList): The data to inverse transform.

        Returns:
            FunctionsList: The inverse transformed data.
        """

        _new_X = FunctionsList(X.fun_shape)
        
        for ii in range(len(X)):
            _new_X.append(
                X(ii) * (self._std if self.with_std+1e-12 else 1) + self._mean
            )

        return _new_X

class MinMaxScaler(ScalerWrapper):
    """
    Scaler that scales the data to a specified range (default is [0, 1]).

    Parameters
    ----------
    wrt_ic : bool, optional
        If True, the scaling is done with respect to the initial condition - element 0 of the input data (default is False).
    global_scale : bool, optional
        If True, the scaling is done globally across all data (default is False).
    """

    def __init__(self, wrt_ic: bool = False, global_scale: bool = False):
        super().__init__()
        self.wrt_ic = wrt_ic
        self.global_scale = global_scale

    def fit(self, train_X: FunctionsList):
        """
        Fit the MinMaxScaler to the training data.

        Args:
            train_X (FunctionsList): The training data to fit the scaler to.
        """

        if self.wrt_ic:
            self._min = np.ones((train_X.fun_shape,)) * train_X(0).min()
            self._max = np.ones((train_X.fun_shape,)) * train_X(0).max()
        elif self.global_scale:
            self._min = np.ones((train_X.fun_shape,)) * train_X.min()
            self._max = np.ones((train_X.fun_shape,)) * train_X.max()
        else:
            self._min = train_X.min(axis=1)
            self._max = train_X.max(axis=1)

    def transform(self, X: FunctionsList):
        """
        Transform the data using the fitted MinMaxScaler.

        Args:
            X (FunctionsList): The data to transform.

        Returns:
            FunctionsList: The transformed data.
        """

        _new_X = FunctionsList(X.fun_shape)
        
        for ii in range(len(X)):
            _new_X.append(
                (X(ii) - self._min) / (self._max - self._min + 1e-12)
            )

        return _new_X
    
    def inverse_transform(self, X: FunctionsList):
        """
        Inverse transform the data using the fitted MinMaxScaler.

        Args:
            X (FunctionsList): The data to inverse transform.

        Returns:
            FunctionsList: The inverse transformed data.
        """

        _new_X = FunctionsList(X.fun_shape)
        
        for ii in range(len(X)):
            _new_X.append(
                X(ii) * (self._max - self._min) + self._min
            )

        return _new_X
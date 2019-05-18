# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression


def __default_predict_method(model):
    try:
        method = model.predict_proba
    except AttributeError as e: 
        method = model.predict
    return method


def __link_equal(x):
    return x


def __link_logit(x):
    return np.log(x / (1-x))


def get_local_linear_model(model, x0,
                           predict_method=__default_predict_method,
                           link='logit',
                           epsilon=1e-4,
                           n_estimate=1000):
    """
    explore neighbor of x0 randomly and make local model.
    """
    if link == 'equal':
        link = __link_equal
    elif link == 'logit':
        link = __link_logit
    elif not isinstance(link, function):
        raise ValueError('link must be "logit" or function.')



    nx = np.random.normal(x0, epsilon, size=(n_estimate, len(x0)))
    ny = predict_method(model)(nx)
    if ny.ndim >= 2:
        ny = ny[:, 1]
    ny = link(ny)
    linr = LinearRegression()
    linr.fit(nx, ny)
    return linr
    

def local_gradient(model, x0,
                   predict_method=__default_predict_method,
                   link='logit',
                   epsilon=1e-4,
                   n_estimate=1000):
    """ only provide gradient. """
    return get_local_linear_model(model=model, x0=x0,
                                  predict_method=predict_method,
                                  link=link,
                                  epsilon=epsilon,
                                  n_estimate=n_estimate).coef_


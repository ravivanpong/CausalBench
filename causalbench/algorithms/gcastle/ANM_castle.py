from .model import Model
from castle.algorithms import ANMNonlinear
import numpy as np


class ANM_castle(Model):
    name = 'ANM'
    lib = 'castle'

    def fit(self, data, parameters):
        default_param = {'alpha': 0.05}
        
        alpha = parameters['alpha'] if 'alpha' in parameters else default_param['alpha']
        
        try:
            anm = ANMNonlinear( alpha=alpha)
            anm.learn(data)
            return np.asarray(anm.causal_matrix)
        except Exception as e:
            print(e)
            return np.array([])

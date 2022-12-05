from .base import BaseCastle
from castle.algorithms import PC
import numpy as np


class PC_castle(BaseCastle):
    name = 'PC'
    lib = 'castle'

    def fit(self, data, parameters):
        default_param = {
            'variant': 'original',
            'alpha': 0.05,
            'ci_test': 'fisherz',
            'priori_knowledge': None
        }
        variant = parameters['variant'] if 'variant' in parameters else default_param['variant']
        alpha = parameters['alpha'] if 'alpha' in parameters else default_param['alpha']
        ci_test = parameters['ci_test'] if 'ci_test' in parameters else default_param['ci_test']
        priori_knowledge = parameters['priori_knowledge'] if 'priori_knowledge' in parameters else default_param['priori_knowledge']
        try:
            pc = PC(variant=variant, alpha=alpha, ci_test=ci_test,
                    priori_knowledge=priori_knowledge)
            pc.learn(data)
            return np.asarray(pc.causal_matrix)
        except Exception as e:
            print(e)
            return np.array([])

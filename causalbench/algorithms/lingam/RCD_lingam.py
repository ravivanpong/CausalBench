from causalbench.algorithms.base import Base
from lingam import RCD
import numpy as np


class RCD_lingam(Base):
    name = 'RCD'
    lib = 'lingam'

    def fit(self, data, parameters):
        default_param = {
            'max_explanatory_num':2,
            'cor_alpha':0.01,
            'ind_alpha':0.01,
            'shapiro_alpha':0.01,
            'MLHSICR':False,
            'bw_method':"mdbs",
        }
        max_explanatory_num = parameters['max_explanatory_num'] if 'max_explanatory_num' in parameters else default_param['max_explanatory_num']
        cor_alpha = parameters['cor_alpha'] if 'cor_alpha' in parameters else default_param['cor_alpha']
        ind_alpha = parameters['ind_alpha'] if 'ind_alpha' in parameters else default_param['ind_alpha']
        shapiro_alpha = parameters['priori_knshapiro_alphaowledge'] if 'shapiro_alpha' in parameters else default_param['shapiro_alpha']
        bw_method = parameters['bw_method'] if 'bw_method' in parameters else default_param['bw_method']
        try:
            rcd = RCD(max_explanatory_num=max_explanatory_num, cor_alpha=cor_alpha, ind_alpha=ind_alpha,
                    shapiro_alpha=shapiro_alpha, bw_method=bw_method)
            rcd.fit(data)
            # TODO: how to deal with the output?
            # return np.asarray(rcd.adjacency_matrix_)
        except Exception as e:
            print(e)
            return np.array([])

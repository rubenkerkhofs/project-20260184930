from prisk.asset import ProducingAsset

import numpy as np


class CoalTerminal(ProducingAsset):
    def __init__(self, product, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.product = product
        self.capacity_factor = 0
        self.__lower_bound_lead_time = 1
        self.__upper_bound_lead_time = 1
        self.__transportation_lead_time = 3
        self.production_pipeline = np.zeros(self.__upper_bound_lead_time + 1)
        


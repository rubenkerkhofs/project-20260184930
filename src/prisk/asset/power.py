import numpy as np

from prisk.asset import ProducingAsset


class PowerPlant(ProducingAsset):
    def __init__(self, product, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.product = product
        self.input_inventory_desired = 25
        self.reorder_point = 25
        self.capacity_factor = 0.55
        self.__lower_bound_lead_time = 0
        self.__upper_bound_lead_time = 0
        self.__transportation_lead_time = 1 
        self.production_pipeline = np.zeros(self.__upper_bound_lead_time + 1)

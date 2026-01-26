""" The shocks used in the agent-based model """
class Shock:
    """ Base class for further development of supply chain shocks. Three
    types of shocks exist
        (1) Demand shock
        (2) Production shock
        (3) Transportation shock
    """
    def __init__(
            self,
            time: int,
            asset,
            size: float,
            recovery_time: float=5) -> None:
        self.asset = asset
        self.size = size
        self.time = time
        self.recovery_time = recovery_time
        self.original_capacity = asset.capacity.maximum
        assert size > 0 and size <= 1, \
            'Shock size should be between 0 (excl.) and 1 (incl.)'

class ProductionShock(Shock):
    def apply(self, time):
        if time == self.time:
            self.asset.capacity.constrained *= (1-self.size)
            self.asset.production_pipeline *= (1-self.size)
            self.asset.inventory.current *= (1-self.size)
            # Also reduce the input inventories
            for input_name, input_inventory in self.asset.input_inventory.items():
                if input_inventory > 0:
                    input_inventory *= (1-self.size)
                    self.asset.input_inventory[input_name] = input_inventory

    def recover(self, time):
        if time >= int(self.time + self.recovery_time  ):
            self.asset.shocks.remove(self)
            if len(self.asset.shocks) == 0:
                # If no shocks are left, we can restore the capacity
                self.asset.capacity.constrained = self.asset.capacity.maximum
            return
        if time > self.time and time < self.time + self.recovery_time  :
            constrained_capacity = self.original_capacity - (self.original_capacity * \
                (1 - self.size * (time - self.time) / self.recovery_time  ))
            if constrained_capacity <= self.asset.capacity_factor * self.original_capacity:
                # If the constrained capacity is less than the capacity factor, we set it to 0
                constrained_capacity = 0
            self.asset.capacity.constrained = constrained_capacity


class AbruptShock(ProductionShock):
    def recover(self, time):
        if time == int(self.time + self.recovery_time):
            self.asset.capacity.constrained = self.asset.capacity.maximum



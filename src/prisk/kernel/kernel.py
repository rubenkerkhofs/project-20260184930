from functools import cached_property
from queue import PriorityQueue
from typing import Optional

from numpy import floor
import numpy as np

from prisk.kernel.message import (
    FloodEvent,
    TropicalCycloneEvent,
    StartofYearEvent,
    InsuranceDropoutEvent,
    EndOfDayEvent,
)
from prisk.demand import ExponentialSmooting
from prisk.asset import ProducingAsset


class Kernel:
    """
    The Kernel provides the environment in which the simulations
    will take place
    """

    def __init__(self, assets, insurers):
        self.messages = PriorityQueue()
        self.internal_time = 0  # Expressed in days
        self.assets = assets
        self.insurers = insurers

    @cached_property
    def firms(self):
        firms = []
        for asset in self.assets:
            parents = asset.parents
            for parent in parents:
                f = parent.get("firm")
                firms.append(f)
        # Ensure that the order is always the same but elements are unique
        # This is important for the simulation to work correctly with a seed
        return list(dict.fromkeys(firms))

    @cached_property
    def producing_firms(self):
        return [firm for firm in self.firms if isinstance(firm, ProducingAsset)]

    def run(self, time_horizon, verbose: int = 0, seed: Optional[int] = None):
        """Run the simulation"""
        from time import time
        if seed is not None:
            np.random.seed(seed)
        latest_time = time()
        if verbose > 0:
            print("Starting simulation")
            print("-------------------")
            print("Adding End of Year events...")
        if num_years := int(floor(time_horizon / 365)):
            for i in range(0, num_years):
                self.messages.put(StartofYearEvent(i * 365))
        for i in range(0, time_horizon):
            self.messages.put(EndOfDayEvent(i))
        while not self.messages.empty() and self.internal_time < time_horizon:
            message = self.messages.get()
            self.internal_time = message.time
            if self.internal_time >= time_horizon:
                break
            if isinstance(message, FloodEvent):
                if verbose > 1:
                    print(
                        f"Flood event at day {int(floor(self.internal_time))} at {message.asset} with depth {message.depth}"
                    )
                message.asset.flood(time=message.time, depth=message.depth)
            elif isinstance(message,TropicalCycloneEvent):
                if verbose > 1:
                    print(
                        f"Tropical cyclone event at day {int(floor(self.internal_time))} at {message.asset} with wind speed {message.wind_speed}"
                    )
                message.asset.tc_impact(time=message.time, wind_speed=message.wind_speed)
            elif isinstance(message, StartofYearEvent):
                if verbose > 0:
                    year_sim_time = time() - latest_time
                    latest_time = time()
                    print(
                        f"Start of year {int(floor(self.internal_time / 365))} at day {int(floor(self.internal_time))} - {year_sim_time:.2f} seconds elapsed"
                    )
                for insurer in self.insurers:
                    insurer.collect_premiums(message.time)
            elif isinstance(message, InsuranceDropoutEvent):
                if verbose:
                    print(
                        f"Insurance dropout at year {int(floor(self.internal_time))} at {message.asset}"
                    )
                message.asset.remove_insurer()
            elif isinstance(message, EndOfDayEvent):
                for firm in self.firms:
                    firm.forecast_demand(algorithm=ExponentialSmooting)
                for firm in self.firms:
                    firm.handle_incoming_goods()
                for firm in self.firms:
                    firm.determine_production_schedule()
                for firm in self.firms:
                    firm.explode(time=message.time)
                for firm in self.firms:
                    firm.produce()
                for firm in self.firms:
                    firm.put_to_stock()
                for firm in self.firms:
                    firm.affirm_need()
                for asset in self.assets:
                    for shock in asset.shocks:
                        shock.recover(time=message.time)
                for firm in self.firms:
                    firm.cancel_orders(time=message.time)
                for firm in self.firms:
                    firm.progress()

        self.internal_time = time_horizon
        if verbose > 0:
            print("-------------------")
            print("Simulation finished")
            print(f"Simulation time: {num_years} years")

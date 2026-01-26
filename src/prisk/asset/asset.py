from collections import Counter
from dataclasses import dataclass
from typing import List

import pandas as pd
import plotly.graph_objects as go
import numpy as np

from prisk.insurance.industry import Insurance
from prisk.shock import ProductionShock


@dataclass
class Order:
    """An order represents an order made by a firm
    to one of its suppliers"""

    time: int
    quantity: float
    client: "Asset"
    product: str
    fulfilled: bool = False
    received: bool = False
    time_till_delivery: int = 0

@dataclass
class Capacity:
    """Keeps track of the capacity of a firm"""

    maximum: float
    constrained: float
    current: float


@dataclass
class Inventory:
    """Keeps track of the inventory of a firm"""

    maximum: float
    desired: float
    current: float

    def reorder_quantity(self) -> float:
        """The amount that needs to be reordered"""
        return self.desired - self.current


class Asset:
    def __init__(
        self,
        name: str,
        latitude: float,
        longitude: float,
        flood_damage_curve: str,
        flood_exposure: List,
        replacement_value: float,
        unit_price: float = 45, # 3,88 rupees per kWh
        margin: float = 0.2,
        discount_rate: float = 0.05,
        flood_protection: float = 0.0,
        tc_protection: float = 25,
        tc_half: float = 50,
        final_demand: float = 0.0,
        insurer=None,
    ) -> None:
        """Initialize the AssetSim object

        Parameters
        ----------
        name : str
            Name of the asset
        flood_damage_curve : str
            The flood damage curve of the asset. This is used to compute the expected
            damage of the asset.
        flood_exposure : List
            The flood exposure of the asset. This is needed to compute the expected
            damage needed to determine the insurer's premium.
        replacement_cost : float
            The replacement cost of the asset. This is used to compute the damage in case of a flood.
        unit_price : float
            The unit price of the asset. This is used to compute the revenue path.
        margin : float
            The margin of the asset. This is used to compute the cost path.
        discount_rate : float
            The discount rate of the asset. This is used to compute the NPV.
        flood_protection : float
            The flood protection of the asset. This is used to compute the expected damage
            against floods.
        insurer : Insurance
            The insurance company that insures the asset. This is used to compute the premium.
        """
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.final_demand = final_demand
        self.final_demand_backlog = 0
        self.maximum_backlog = 0

        self.flood_damage_curve = flood_damage_curve
        self._TIME_HORIZON = len(self.production_path)
        self.damages = np.repeat(0.0, self._TIME_HORIZON)
        self.discount_rate = discount_rate
        self.replacement_cost = replacement_value
        self.flood_exposure = flood_exposure
        self._insurer = insurer

        self.unit_price = unit_price
        self._MARGIN = margin
        self.cost_path = self.revenue_path * (1 - self._MARGIN)
        self.flood_protection = flood_protection
        self.update_expected_damage()
        self.replacement_cost_path = np.repeat(0, self._TIME_HORIZON)
        self.business_disruption_path = np.repeat(0, self._TIME_HORIZON)
        self.insurance_fair_premium_path = np.repeat(0, self._TIME_HORIZON)
        self.insurance_adjustment_path = np.repeat(0, self._TIME_HORIZON)
        self.base_value = self.npv
        self.parents = []
        self.shocks = []
        self.v_half = tc_protection
        self.v_threshold = tc_half
        self.v_threshold_business_disruption = 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def revenue_path(self) -> np.ndarray:
        return self.production_path * self.unit_price

    @property
    def cash_flow_path(self) -> np.ndarray:
        return self.revenue_path - self.cost_path - self.climate_cost_path

    @property
    def discounted_cash_flow(self) -> float:
        return np.sum(
            self.cash_flow_path
            * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON)
        )

    @property
    def climate_cost_path(self) -> np.ndarray:
        return (
            self.replacement_cost_path
            + self.business_disruption_path
            + self.insurance_fair_premium_path
            + self.insurance_adjustment_path
        )

    @property
    def terminal_value(self) -> float:
        return (self.cash_flow_path[-1] + self.climate_cost_path[-1]) / (
            self.discount_rate
        )

    @property
    def npv(self) -> float:
        return self.discounted_cash_flow + self.terminal_value / (
            1 + self.discount_rate
        ) ** (self._TIME_HORIZON + 1)

    @property
    def total_replacement_costs(self) -> float:
        return np.sum(
            self.replacement_cost_path
            * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON)
        )

    @property
    def total_business_disruption(self) -> float:
        return np.sum(
            self.business_disruption_path
            * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON)
        )

    @property
    def total_fair_insurance_premiums(self) -> float:
        return np.sum(
            self.insurance_fair_premium_path
            * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON)
        )

    @property
    def total_insurance_adjustments(self) -> float:
        return np.sum(
            self.insurance_adjustment_path
            * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON)
        )
    
    @property
    def disrupted(self) -> bool:
        return self.capacity.constrained < self.capacity.current \
                or self.bottleneck_production < self.capacity.current
    
    @property
    def parent_objects(self) -> List["Holding"]:
        """Returns the parent objects of the asset. Parents are assets that this asset depends on for its production."""
        return [parent['firm'] for parent in self.parents]
    
    @property
    def clients(self) -> List["ProducingAsset"]:
        """Returns the clients of the asset. Clients are assets that depend on this asset for their production."""
        clients = []
        for parent in self.parent_objects:
            clients.extend(parent.clients)
        return clients
    
    @property
    def client_disrupted(self) -> bool:
        return any(client.disrupted for client in self.clients)


    @property
    def expected_damage(self) -> float:
        return self.__expected_damage

    def reset(self) -> None:
        self.damages = np.repeat(0.0, self._TIME_HORIZON)
        self.replacement_cost_path = np.repeat(0, self._TIME_HORIZON)
        self.business_disruption_path = np.repeat(0, self._TIME_HORIZON)
        self.insurance_fair_premium_path = np.repeat(0, self._TIME_HORIZON)
        self.insurance_adjustment_path = np.repeat(0, self._TIME_HORIZON)
        self.cost_path = self.revenue_path * (1 - self._MARGIN)
        if self._insurer:
            self.remove_insurer()
        self.update_expected_damage()
        self.has_produced = False
        assert self.npv == self.base_value, "The NPV is not reset to the base value."

    def update_expected_damage(self) -> None:
        """
        The expected damages can be derived based on the flood damage curves, the flood exposure
        and the flood protection of the asset. The expected damage is the sum of the damage
        of each flood exposure event, weighted by the Poisson probability of the flood event.
        """
        expected_damage = 0
        for flood_exposure in self.flood_exposure:
            impact_depth = round(
                max(0, flood_exposure.depth - self.flood_protection), 2
            )
            expected_damage += (
                self.flood_damage_curve.loc[impact_depth]
                * flood_exposure.poisson_probability
                * self.replacement_cost
            )
        self.__expected_damage = expected_damage

    def add_insurer(self, insurer: "Insurance") -> None:
        """
        Parameters
        ----------
        insurer : Insurance
            The insurer that insures the asset. For now, we assume that an insurer
            only insurers against capital damages. In the future, we need to extend
            this to business disruptions as well.

        A firm can only have a single insurer. If a firm already has an insurer,
        the new insurer will replace the old insurer.
        """
        if self._insurer:
            self._insurer.remove_subscriber(self)
        self._insurer = insurer
        insurer.add_subscriber(self)

    def remove_insurer(self) -> None:
        self._insurer.remove_subscriber(self)
        self._insurer = None

    def pay_insurance_premium(self, time: float) -> None:
        """Pay the insurance premium to the insurer"""
        premium = self._insurer.premium(self) * self.replacement_cost
        fair_premium = self._insurer.get_fair_premium(self) * self.replacement_cost
        year = int(np.floor(time))
        self.insurance_fair_premium_path[year] += fair_premium
        self.insurance_adjustment_path[year] += premium - fair_premium

    def flood(self, depth: float, time: pd.Timestamp):
        """Simulate the flood event and its impact on capital damages and production.

        Parameters
        ----------
        depth : float
            The depth of the flood event
        time : pd.Timestamp
            The time of the flood event
        """
        impact_depth = round(max(0, depth - self.flood_protection), 2)
        damage = self.flood_damage_curve.loc[impact_depth].damage
        disruption_days = np.ceil(self.flood_damage_curve.loc[impact_depth].production)
        year = time // 365
        if self._insurer is None:
            self.replacement_cost_path[year] += self.replacement_cost * damage
        else:
            self._insurer.payout(self.replacement_cost * damage)
        if disruption_days > 0:
            shock = ProductionShock(
                time=time, 
                asset=self,
                size=1,
                recovery_time=disruption_days)
            self.shocks.append(shock)
            shock.apply(time)

    def tc_impact(
            self, 
            wind_speed: float, 
            time: pd.Timestamp):
        """Simulate the impact of a tropical cyclone on the asset.

        Parameters
        ----------
        wind_speed : float
            The wind speed of the tropical cyclone
        time : pd.Timestamp
            The time of the tropical cyclone event
        """
        v = max(0, wind_speed-self.v_threshold) / (self.v_half - self.v_threshold)
        damage = v**3 / (1 + v**3)
        year = int(np.floor(time/365))
        if wind_speed < self.v_threshold:
            business_disruption = 0
        else:
            business_disruption = int(np.ceil(61.96*(1/(1+np.exp(-0.1036*(wind_speed-69.21))))))
        if self._insurer is None:
            self.replacement_cost_path[year] += self.replacement_cost * damage
        else:
            self._insurer.payout(self.replacement_cost * damage)
        
        if business_disruption > 0:
            shock = ProductionShock(
                time=time, 
                asset=self,
                size=1,
                recovery_time=business_disruption)
            self.shocks.append(shock)
            shock.apply(time)


    def plot_risk(self) -> None:
        fig = go.Figure(
            go.Waterfall(
                name="PRISK - Waterfall - " + self.name,
                orientation="v",
                measure=[
                    "relative",
                    "relative",
                    "relative",
                    "relative",
                    "relative",
                    "total",
                ],
                x=[
                    "Base Value",
                    "Capital Damagages",
                    "Business disruptions",
                    "Fair insurance premiums",
                    "Insurance adjustments",
                    "Adjusted Value",
                ],
                textposition="outside",
                text=[
                    "{:,.2f}M".format(self.base_value / 1e6),
                    "{:,.2f}M".format(-self.total_replacement_costs / 1e6),
                    "{:,.2f}M".format(-self.total_business_disruption / 1e6),
                    "{:,.2f}M".format(-self.total_fair_insurance_premiums / 1e6),
                    "{:,.2f}M".format(-self.total_insurance_adjustments / 1e6),
                    "{:,.2f}M".format(self.npv / 1e6),
                ],
                y=[
                    self.base_value,
                    -self.total_replacement_costs,
                    -self.total_business_disruption,
                    -self.total_fair_insurance_premiums,
                    -self.total_insurance_adjustments,
                    self.npv,
                ],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )

        fig.update_layout(
            title="PRISK - Waterfall chart - " + self.name,
            showlegend=False,
            template="simple_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Asset value impacts",
            yaxis_title="Value",
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            yaxis_showgrid=True,
        )

        fig.show()


class ProducingAsset(Asset):
    plants = []

    def __init__(self, 
                 inventory: Inventory, 
                 capacity: Capacity, 
                 *args, **kwargs):
        """
        A producing Asset produces actual goods and can as a result be linked to other
        firms in the supply chain. The producing asset has an input inventory and an output
        inventory. The input inventory is used to store the inputs that are needed for production.

        Furthermore, the producing asset has a production schedule. This production schedule
        is set by the firm that owns the asset. The production schedule is the amount of goods
        that the firm wants to produce at each timestep. The production schedule is constrained
        by the capacity of the firm. The capacity of the firm is the maximum amount of goods that
        can be produced at each timestep.

        Parameters
        ----------
        inventory : Inventory
            The inventory of the asset
        capacity : Capacity
            The capacity of the asset
        """
        self.production_path = np.repeat(capacity.current*24*365, 10)
        super().__init__(*args, **kwargs)
        self.inventory = inventory
        self.capacity = capacity
        self.__latest_demand = 0
        self.__production_history = []
        self.__disruption_impacts = []
        self.__constrained_history = []
        self.__total_demand_history = []
        self.upstream_history = []
        self.downstream_history = []
        self.stable_production_history = []
        self.stable_production_sum = capacity.current # optimization
        self.stable_production_count = 1 # optimization
        self.downstream_shocks = np.repeat(0.0, self._TIME_HORIZON)
        self.upstream_shocks = np.repeat(0.0, self._TIME_HORIZON)
        self.direct_disruptions = np.repeat(0.0, self._TIME_HORIZON)
        self.bottleneck = np.inf

        # The input recipe shows the proportion of input goods needed for the production
        # of a single output. For example a recipe of 2 means that 2 units of input are needed
        # for the production of a single output.
        self.recipe = {}
        self._suppliers = []
        self._supplier_recipes = {}
        self._clients = []
        self.input_inventory = {}
        self.reorder_point = 25  # x times the input needed for max capacity
        self.input_inventory_desired = 25  # x times the input needed for max
        self.__orders = []
        self.__inventory_history = []
        self.running_total = 0

        # Demand forecasting
        self.__forecasted_demand = 0

        self.production_schedule = 0
        self.has_produced = False
        self.has_ordered = False

        self.__lower_bound_lead_time = 0
        self.__upper_bound_lead_time = 0
        self.__transportation_lead_time = 1 
        self.production_pipeline = np.zeros(self.__upper_bound_lead_time + 1)
        self.plants.append(self)

    @property
    def production_capacity(self):
        """The capacity for production of the facility"""
        return np.ceil(min(self.capacity.constrained, self.capacity.maximum))

    @property
    def bottleneck_production(self):
        """Provides the maximum production based on the inputs available.
        At this stage, all inputs should be stored in the input inventory
        including the inputs that have been delivered at the current
        timestep"""
        max_production = 1e100
        for product, recipe in self.recipe.items():
            max_production = min(
                max_production, self.input_inventory[product] / recipe
            )
        return max_production
    
    @property
    def production_history(self) -> List[float]:
        """Returns the production history of the asset"""
        return self.__production_history
    
    @property
    def disruption_impacts(self) -> List[float]:
        return self.__disruption_impacts

    @property
    def constrained_history(self) -> List[float]:
        """Returns the constrained production history of the asset"""
        return self.__constrained_history
    
    @property
    def total_demand_history(self) -> List[float]:
        """Returns the total demand history of the asset"""
        return self.__total_demand_history
    
    @property
    def total_upstream_shocks(self):
        return np.sum(
            self.upstream_shocks
            * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON)
        )
    
    @property
    def total_downstream_shocks(self):
        return np.sum(
            self.downstream_shocks
            * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON)
        )
    
    @property
    def total_direct_disruptions(self):
        return np.sum(
            self.direct_disruptions
            * (1 - self.discount_rate) ** np.arange(self._TIME_HORIZON)
        )
        
    ####
    # Demand forecasting functionality
    ####
    @property
    def forecasted_demand(self) -> float:
        return self.__forecasted_demand

    def forecast_demand(self, algorithm) -> None:
        if len(self.__total_demand_history) > 5:
            alg = algorithm(self.__total_demand_history)
            self.__forecasted_demand = alg.forecast
        else:
            self.__forecasted_demand = self.capacity.maximum * 0.8

    ####
    # Supply chain functionality
    ####
    @property
    def suppliers(self):
        """Returns the suppliers of the asset. Suppliers are assets that provide inputs for the production."""
        return self._suppliers

    @property
    def clients(self):
        """Returns the clients of the asset. Clients are assets that depend on this asset for their production."""
        return self._clients

    def add_supplier(
            self,
            supplier: "ProducingAsset",
            product: str,
            recipe_input: float,
    ):
        assert product == supplier.product, f"The product of the supplier must match the product of the asset. {product} != {supplier.product}"
        assert recipe_input > 0, "The recipe input must be greater than 0"
        assert recipe_input <= 1, "The recipe input must be less than or equal to 1"
        if supplier not in self._suppliers:
            self._suppliers.append(supplier)
        if product not in self.recipe:
            self.recipe[product] = recipe_input
            self.input_inventory[product] = recipe_input * self.capacity.maximum * self.input_inventory_desired
        else:
            self.recipe[product] += recipe_input
            self.input_inventory[product] += recipe_input * self.capacity.maximum * self.input_inventory_desired
        self._supplier_recipes[supplier] = recipe_input
    
    def add_client(self, client: "ProducingAsset") -> None:
        if client not in self._clients:
            self._clients.append(client)

    def get_in_transit(self, recipient: "ProducingAsset") -> List:
        return [
            order
            for order in self.orders
            if order.client == recipient and not order.received and order.fulfilled
        ]
    
    def handle_incoming_goods(self) -> None:
        """Handles the incoming goods from suppliers. This is used to update the input inventory."""
        for supplier in self.suppliers:
            orders_in_transit = supplier.get_in_transit(recipient=self)
            for order in orders_in_transit:
                if order.fulfilled and not order.received:
                    self.input_inventory[order.product] += order.quantity
                    order.received = True
                    order.fulfilled = False
                    supplier.remove_order(order)

    ####
    # Production chain
    # The production chain at a facility looks as follows:
    #  1. The facility will procure the required materials
    #     which means that these goods will be taken from the
    #     input inventory. This is based on the production
    #     schedule.
    #  2. The asset will produce goods according to the
    #     production schedule. This production schedule
    #     is determined by the firm and takes into account
    #     the production capacity.
    #  3. The goods produced are put into stock
    #  4. New orders are made to replenish input inventories
    ####
    def determine_production_schedule(self) -> None:
        """
            Determines how much the asset will produce given the current
            production capacity and forecasted demand.
        """
        desired_inventory_change = self.inventory.desired - self.inventory.current
        in_production = sum(production_time_t for production_time_t in self.production_pipeline)
        order_backlog = sum(order.quantity for order in self.orders if not order.fulfilled)
        desired_production = (
            self.forecasted_demand
            + desired_inventory_change
            - in_production
            + self.final_demand_backlog
            + order_backlog
        )
        self.production_schedule = max(
            0, min(
                self.production_capacity,
                desired_production,
                self.bottleneck_production
            )
        )
        self.bottleneck = self.bottleneck_production  
              

    def procure_materials(self):
        """Based on the assigned production schedule, the input inventories will
        be updated to reflect the required inputs for the production"""
        for product, recipe in self.recipe.items():
            used_inputs = np.ceil(self.production_schedule * recipe)
            self.input_inventory[product] = np.ceil(self.input_inventory[product] - used_inputs)

    def produce(self):
        """Produce the goods based on the production schedule."""
        lead_times = Counter(np.random.randint(self.__lower_bound_lead_time, self.__upper_bound_lead_time + 1, size=100))
        assert self.production_schedule <= self.production_capacity, "Production schedule exceeds capacity: {} {}".format(self.production_schedule, self.production_capacity)
        production_factor = self.production_schedule / 100
        for time, proportion in lead_times.items():
            self.production_pipeline[time] += proportion * production_factor
        self.has_produced = True

    def put_to_stock(self):
        """Puts the produced items to the stock. Any overproduction is discarded."""
        self.inventory.current = min(
            np.ceil(self.inventory.current + self.production_pipeline[0]), self.inventory.maximum
        )
        self.__production_history.append(self.production_pipeline[0])
        self.__constrained_history.append(self.capacity.constrained)
        
    ###
    # Ordering functionality
    ###
    @property
    def orders(self):
        """Returns the orders made at the asset. Orders are used to keep track of the orders made by clients."""
        return self.__orders
    
    @property
    def inventory_history(self) -> List[float]:
        """Returns the inventory history of the asset"""
        return self.__inventory_history
    
    def remove_order(self, order):
        """Removes an order from the asset's orders"""
        if order in self.__orders:
            self.__orders.remove(order)
        else:
            raise ValueError("Order not found in the asset's orders")
        
    def cancel_orders(self, time: int) -> None:
        """Cancels all orders that are not fulfilled and have not been received"""
        for order in self.__orders:
            if not order.fulfilled and not order.received and time - order.time > 5:
                self.__orders.remove(order)
    
    def add_order(self, client, product, quantity, time) -> None:
        """Adds an order to the existing list of orders"""
        delivery_time = np.random.randint(0, self.__transportation_lead_time)
        order = Order(
            time=time, 
            quantity=quantity, 
            client=client, 
            product=product, 
            fulfilled=False, 
            received=False, 
            time_till_delivery=delivery_time)
        self.__orders.append(order)
        self.__latest_demand += quantity

    def __make_order(self, asset, product, quantity: float, time: int) -> None:
        """Allows the firm to make an order at one of their suppliers"""
        asset.add_order(
            client=self,  
            product=product, 
            quantity=quantity, 
            time=time)

    def order_raw_materials(self, time):
        """Creates the input orders to fill inventory to the
        desired level"""
        orders_in_progress = [
            order 
            for supplier in self.suppliers
            for order in supplier.orders
            if order.client == self]
        for product, inventory in self.input_inventory.items():
            in_order = sum(order.quantity for order in orders_in_progress 
                           if order.product == product)
            recipe_quantity = self.recipe[product] * self.capacity.maximum
            reorder_point = np.floor(recipe_quantity * self.reorder_point)
            if inventory + in_order < reorder_point:
                desired_inventory = recipe_quantity * self.input_inventory_desired
                order_quantity = np.ceil(desired_inventory - inventory - in_order) 
                if order_quantity > 0.01:  # rounding errors
                    # Split order over all suppliers
                    supplier_capacities = {
                        supplier: 1 # Capacities already reflected in leontief productions
                        for supplier in self.suppliers
                        if not supplier.disrupted and product == supplier.product
                    }
                    total_capacity = sum(supplier_capacities.values())
                    if total_capacity == 0:
                        # If None of them have any capacity, then we order from all of them
                        # the same amount
                        supplier_capacities = {
                            supplier: 1 for supplier in self.suppliers
                            if product == supplier.product
                        }
                        total_capacity = len(supplier_capacities)
                    for supplier, capacity in supplier_capacities.items():
                        order_quantity_supplier = order_quantity*(capacity/total_capacity)
                        self.__make_order(
                            asset=supplier, 
                            product=product, 
                            quantity=min(np.ceil(order_quantity_supplier), supplier.capacity.maximum), 
                            time=time)
        self.has_ordered = True

    def adjust_production_paths(self, day):
        """Adjust the production paths based on the over/under production"""
        latest_production = self.production_schedule
        stable_production = self.stable_production_sum / self.stable_production_count
    
        if self.disrupted: 
            # There is a direct impact or bottleneck
            over_under_production = latest_production - stable_production
            year = day // 365
            disruption = min(over_under_production*24*self.unit_price, 0)
            self.business_disruption_path[year] -= disruption
            if self.bottleneck_production < stable_production:
                self.upstream_shocks[year] -= disruption
                self.upstream_history.append(-min(over_under_production,0))
                self.downstream_history.append(0)
                self.__disruption_impacts.append(disruption)
            else:
                self.direct_disruptions[year] -= min(over_under_production*24*self.unit_price, 0)
                self.__disruption_impacts.append(min(over_under_production*24*self.unit_price, 0))
                self.upstream_history.append(0)
                self.downstream_history.append(0)
        elif (latest_production < (1/1.05)*stable_production \
                or latest_production > 1.05*stable_production) and self.client_disrupted:
            over_under_production = latest_production - stable_production
            year = day // 365
            self.business_disruption_path[year] -= over_under_production*24*self.unit_price
            self.downstream_shocks[year] -= over_under_production*24*self.unit_price
            self.__disruption_impacts.append(over_under_production*24*self.unit_price)
            self.upstream_history.append(0)
            self.downstream_history.append(-over_under_production)
        else:
            if (latest_production > (1/1.05)*stable_production \
                or latest_production < 1.05*stable_production):
                self.stable_production_sum += latest_production
                self.stable_production_count += 1
            self.__disruption_impacts.append(0)
            self.upstream_history.append(0)
            self.downstream_history.append(0)
        self.stable_production_history.append(
            self.stable_production_sum / self.stable_production_count
        )

    ###
    # Demand fulfullment functionality
    ###
    def __adjust_stock(self, order_size):
        """Adjusts the output inventory of the asset"""
        self.inventory.current -= order_size

    def affirm_need(self):
        """Searches for orders that need to be fulfilled"""
        unfulfilled = [order for order in self.__orders if not order.fulfilled]
        total_value = sum(order.quantity for order in unfulfilled)
        if (
            self.inventory.current
            >= total_value + self.final_demand.current + self.final_demand_backlog
        ):
            # Fulfill orders
            for order in unfulfilled:
                order.fulfilled = True
                self.__adjust_stock(order.quantity)
            # Fulfill final demand
            self.__adjust_stock(self.final_demand.current)
            # Fulfil backlog
            if self.inventory.current >= self.final_demand_backlog:
                self.__adjust_stock(self.final_demand_backlog)
                self.final_demand_backlog = 0
            else:
                prop_backlog = self.inventory.current / self.final_demand_backlog
                self.__adjust_stock(self.inventory.current)
                self.final_demand_backlog *= (1 - prop_backlog)
        else:
            ratio = self.inventory.current / (
                self.final_demand.current + total_value + self.final_demand_backlog
            )
            self.__adjust_stock(ratio * self.final_demand.current)
            self.__adjust_stock(ratio * self.final_demand_backlog)
            self.final_demand_backlog *= (1 - ratio)
            if len(unfulfilled) > 0:
                order = unfulfilled[0]
                while order.quantity <= self.inventory.current:
                    order.fulfilled = True
                    self.__adjust_stock(order.quantity)
                    unfulfilled = [
                        order for order in self.__orders if not order.fulfilled
                    ]
                    if len(unfulfilled) > 0:
                        order = unfulfilled[0]
                    else:
                        break
                # Do a partial fulfillment of the last order
                if len(unfulfilled) > 0:
                    order = unfulfilled[0]
                    if order.quantity > self.inventory.current:
                        order.quantity = self.inventory.current
                        order.fulfilled = True
                        self.__adjust_stock(order.quantity)
            self.final_demand_backlog = max(
                self.maximum_backlog,
                self.final_demand_backlog - ratio * self.final_demand_backlog,
            )

    ###
    # Progress
    ###
    def progress(self):
        self.__inventory_history.append(self.inventory.current)
        self.__total_demand_history.append(
            max(self.__latest_demand + self.final_demand.current, 0.5*self.capacity.current)  # Avoid zero demand
        )
        day = len(self.__total_demand_history)
        self.adjust_production_paths(day)
        self.production_pipeline = np.roll(self.production_pipeline, -1)
        self.production_pipeline[-1] = 0
        self.final_demand.current = np.floor(
            np.random.normal(
                    loc=self.final_demand.initial, scale=self.final_demand.volatility()
            )
        )
        self.__latest_demand = 0
        self.has_produced = False
        self.has_ordered = False

    def clear(self):
        """Clears the production pipeline and resets the production history"""
        self.production_pipeline = np.zeros(self.__upper_bound_lead_time + 1)
        self.__production_history = []
        self.__constrained_history = []
        self.__total_demand_history = []
        self.has_produced = False
        self.has_ordered = False

    def visualize_production_history(self, window_size=0, proportion=False):
        import matplotlib.pyplot as plt
        plt.figure()
        if window_size > 0:
            production_history = np.convolve(self.__production_history, np.ones(window_size) / window_size, mode="valid")
        else:
            production_history = self.__production_history
        constrained_production = self.__constrained_history
        if proportion:
            production_history = [prod / self.capacity.current for prod in production_history]
            constrained_production = [prod / self.capacity for prod in constrained_production]
        plt.plot(production_history, label=f"Production - {self.product}")
        plt.plot(constrained_production, label=f"Constrained Production - {self.product}", alpha=0.6)
        plt.plot(self.stable_production_history, color='r', linestyle='--', label="Stable Production")
        # Plot upstream history on separate y axis
        if self.upstream_history:
            plt.plot(self.upstream_history, label="Upstream Shocks", linestyle='--', color='orange')
        if self.downstream_history:
            plt.plot(self.downstream_history, label="Downstream Shocks", linestyle='--', color='green')
        plt.xlabel("Timestep")
        plt.ylabel("Production")
        plt.title(f"Production History for {self}")
        plt.legend()
        plt.show()
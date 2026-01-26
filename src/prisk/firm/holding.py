from functools import cached_property

import plotly.graph_objects as go

from prisk.asset.asset import Asset, ProducingAsset


class Holding:
    """
    The holding firm consists of a collection of assets in which
    it has a certain percentage of ownership.
    """

    def __init__(self, name: str, leverage_ratio: float = 0.50) -> None:
        """
        Parameters
        ----------
        name : str
            The name of the holding firm
        leverage_ratio : float
            The leverage ratio of the holding firm. The default value is 0.407867 which is based
            on an industry average obtained through FactSet.
        """

        self.name = name
        self.assets = []
        self.ownership = {}
        self.leverage_ratio = leverage_ratio

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def add_asset(self, asset: "Asset", ownership: float) -> None:
        """
        Add an asset to the holding firm

        Parameters
        ----------
        asset : Asset
            The asset to be added to the holding firm
        ownership : float
            The percentage ownership of the holding firm in the asset
        """
        self.assets.append(asset)
        self.ownership[asset] = ownership
        asset.parents.append({"firm": self, "ownership": ownership})

    def remove_asset(self, asset: "Asset") -> None:
        """
        Parameters
        ----------
        asset : Asset
            The asset to be removed from the holding firm

        """
        self.assets.remove(asset)
        del self.ownership[asset]
        asset.parents = [parent for parent in asset.parents if parent["firm"] != self]

    def get_asset_ownership(self, asset: "Asset") -> float:
        """
        Parameters
        ----------
        asset : Asset
            The asset to get the ownership of
        """
        return self.ownership.get(asset, 0)

    @property
    def base_value(self) -> float:
        return sum(asset.base_value * self.ownership[asset] for asset in self.assets)

    @property
    def total_replacement_costs(self) -> float:
        return sum(
                asset.total_replacement_costs * self.ownership[asset]
                for asset in self.assets
        )

    @property
    def total_business_disruption(self) -> float:
        return sum(
                asset.total_business_disruption * self.ownership[asset]
                for asset in self.assets
        )

    @property
    def total_fair_insurance_premiums(self) -> float:
        return sum(
                asset.total_fair_insurance_premiums * self.ownership[asset]
                for asset in self.assets
        )

    @property
    def total_insurance_adjustments(self) -> float:
        return sum(
                asset.total_insurance_adjustments * self.ownership[asset]
                for asset in self.assets
        )

    @property
    def npv(self) -> float:
        return sum(asset.npv * self.ownership[asset] for asset in self.assets)

    @property
    def liabilities(self) -> float:
        return self.leverage_ratio * self.original_assets

    @property
    def original_liabilities(self) -> float:
        return self.leverage_ratio * self.original_assets

    @property
    def leverage(self) -> float:
        return self.liabilities / self.total_assets

    @property
    def original_leverage(self) -> float:
        return self.original_liabilities / self.original_assets

    @property
    def delta_leverage(self) -> float:
        return self.leverage - self.original_leverage

    @property
    def profitability(self) -> float:
        return self.revenue / self.original_assets

    @property
    def original_profitability(self) -> float:
        return self.original_revenue / self.original_assets

    @property
    def delta_profitability(self) -> float:
        return self.profitability - self.original_profitability

    @property
    def original_revenue(self) -> float:
        return sum(
                sum(asset.revenue_path)
                * self.ownership[asset]
                / len(asset.revenue_path)
                for asset in self.assets
        )

    @property
    def revenue(self) -> float:
        return sum(
                sum(asset.cash_flow_path + asset.cost_path)
                * self.ownership[asset]
                / len(asset.revenue_path)
                for asset in self.assets
        )

    @property
    def original_assets(self) -> float:
        return sum(asset.base_value * self.ownership[asset] for asset in self.assets)

    @property
    def total_assets(self) -> float:
        return sum(asset.npv * self.ownership[asset] for asset in self.assets)

    @property
    def delta_pd(self) -> float:
        # Parameters obtained from a regression analysis conducted by the ECB (2021 Economy
        # wide stress test). The parameters have been adjusted such that the delta_pd
        # is comparable to the merton model
        return 0.454 * self.delta_leverage - 0.533 * self.delta_profitability

    def plot_risk(self) -> None:
        fig = go.Figure(
            go.Waterfall(
                name="PRISK - Waterfall",
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
            title="PRISK - Waterfall chart",
            showlegend=False,
            template="simple_white",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Firm value impacts: {0}".format(self.name),
            yaxis_title="Value",
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            yaxis_showgrid=True,
        )

        fig.show()


class Firm(Holding):
    def __init__(self, name: str, leverage_ratio: float = 0.5):
        super().__init__(name, leverage_ratio)

        # Production
        self.stable_production_sums = {}
        self.stable_production_counts = {}
        self.production_schedule = {}
        self.__production_history = {}

    @property
    def products(self):
        return set([asset.product for asset in self.assets])

    @property
    def suppliers(self):
        suppliers = []
        for asset in self.assets:
            suppliers.extend(asset.suppliers)
        return self.suppliers

    @property
    def clients(self):
        clients = []
        for asset in self.assets:
            clients.extend(asset.clients)
        return clients

    def add_asset(self, asset: "Asset", ownership: float) -> None:
        """
        Add an asset to the holding firm

        Parameters
        ----------
        asset : Asset
            The asset to be added to the holding firm
        ownership : float
            The percentage ownership of the holding firm in the asset
        """
        self.assets.append(asset)
        self.ownership[asset] = ownership
        asset.parents.append({"firm": self, "ownership": ownership})

    @cached_property
    def producing_assets(self):
        return [asset for asset in self.assets if isinstance(asset, ProducingAsset)]

    # Inventory
    @property
    def desired_inventory(self):
        return {
            product: sum(asset.inventory.desired*self.ownership[asset] for asset in self.producing_assets if asset.product == product)
            for product in self.products}

    @property
    def inventory(self):
        return {
            product: sum(asset.inventory.current*self.ownership[asset] for asset in self.producing_assets if asset.product == product)
            for product in self.products}

    # Production
    @cached_property
    def capacity(self):
        return {
            product: sum(asset.capacity.maximum*self.ownership[asset] for asset in self.producing_assets if asset.product == product)
            for product in self.products}

    @property
    def maximum_production(self) -> float:
        """Displays the maximum current firm production"""
        return {
            product: sum(min(asset.production_capacity, asset.bottleneck_production)*self.ownership[asset] for asset in self.producing_assets if asset.product == product)
            for product in self.products}
    
    @property
    def constrained_production(self) -> float:
        return {
            product: sum(asset.production_capacity*self.ownership[asset] for asset in self.producing_assets if asset.product == product)
            for product in self.products}
    
    @property
    def production_history(self) -> dict:
        return self.__production_history
    
    @property
    def upstream_shocks(self) -> float:
        """Returns the total upstream shocks"""
        return sum(asset.total_upstream_shocks * self.ownership[asset] for asset in self.producing_assets)
    
    @property
    def downstream_shocks(self) -> float:
        """Returns the total downstream shocks"""
        return sum(asset.total_downstream_shocks * self.ownership[asset] for asset in self.producing_assets)
    
    @property
    def direct_disruptions(self) -> float:
        """Returns the total direct disruptions"""
        return sum(asset.total_direct_disruptions * self.ownership[asset] for asset in self.producing_assets)

    @property
    def in_production(self) -> float:
        """Computes the total number of goods in production"""
        product_productions = {product: 0 for product in self.products}
        for asset in self.producing_assets:
            product_productions[asset.product] += sum(asset.production_pipeline)
        return product_productions


    @property
    def input_bottleneck(self) -> float:
        """Provides the maximum production based on the inputs available
        at each asset"""
        return {
            product: sum(asset.bottleneck_production*self.ownership[asset] for asset in self.producing_assets if asset.product == product)
            for product in self.products}
    
    @property
    def disrupted(self) -> bool:
        return {
            product: any(asset.disrupted for asset in self.producing_assets if asset.product == product)
            for product in self.products}
    
    def forecast_demand(self, algorithm) -> None:
        """Forecasts the demand for the products based on the historical data"""
        for asset in self.producing_assets:
            asset.forecast_demand(algorithm=algorithm)

    def determine_production_schedule(self) -> None:
        """Determines the production schedule for the firm"""
        for asset in self.producing_assets:
            asset.determine_production_schedule()

    def explode(self, time):
        """Creates the demand for the source products based on the
        production schedule
        """
        for asset in self.producing_assets:
            asset.procure_materials()
            asset.order_raw_materials(time)

    def produce(self):
        """Produces the goods based on the production schedule"""
        for asset in self.producing_assets:
            if not asset.has_produced:
                asset.produce()

    def put_to_stock(self):
        """Puts the amount produced in the final goods inventory"""
        for asset in self.producing_assets:
            asset.put_to_stock()

    def affirm_need(self):
        """Affirms the need for the goods based on the production schedule"""
        for asset in self.producing_assets:
            asset.affirm_need()

    def handle_incoming_goods(self):
        """Distributes the incoming goods to the respective facilities"""
        for asset in self.producing_assets:
            asset.handle_incoming_goods()

    def cancel_orders(self, time: int) -> None:
        """Cancels all orders that are not fulfilled after 5 days"""
        for asset in self.producing_assets:
            asset.cancel_orders(time)

    def progress(self):
        """progresses the firm a timestep"""
        for asset in self.producing_assets:
            asset.progress()


    #####
    # Other
    ######
    def clear(self):
        for asset in self.assets:
            asset.clear()   

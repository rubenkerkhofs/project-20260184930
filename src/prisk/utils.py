from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import re
from scipy.stats import norm, poisson

from prisk.firm import Holding
from prisk.asset import PowerPlant
from prisk.flood import FloodExposure


def convert_to_continous_damage(damage_curves):
    continuous_curves = pd.DataFrame(
        {"index": range(0, int(max((damage_curves["depth"] + 0.01) * 100)))}
    )
    continuous_curves["index"] = continuous_curves["index"] / 100
    continuous_curves.set_index("index", inplace=True)
    continuous_curves = continuous_curves.merge(
        damage_curves, how="left", left_index=True, right_on="depth"
    )
    continuous_curves.interpolate(method="linear", inplace=True)
    continuous_curves.set_index("depth", inplace=True)
    return continuous_curves

BASE_PATH = "https://kuleuven-prisk.s3.eu-central-1.amazonaws.com"
BASE_PATH = "/Users/ruben/Desktop/do_not_delete"

damage_curves = pd.read_excel(
    f"{BASE_PATH}/damage_curves.xlsx"
)
power = pd.read_excel(f"{BASE_PATH}/power.xlsx")
indian_firms = pd.read_excel(
    f"{BASE_PATH}/Indian_firms.xlsx"
)
indian_firm_mapping = mapping = {
    row["name"]: row["clean"] for _, row in indian_firms[["name", "clean"]].iterrows()
}
power.drop(columns=[2], inplace=True)
continuous_curves = convert_to_continous_damage(damage_curves)
return_period_columns = [5, 10, 25, 50, 100, 200, 500, 1000]

hybas_basins = f"{BASE_PATH}/hybas_as_lev06_v1c.shp"
basins = gpd.read_file(hybas_basins)
basin_geometries = basins.set_index('HYBAS_ID')['geometry'].to_dict()

alpha_params = pd.read_excel(f'{BASE_PATH}/alpha_params_tc.xlsx')


def bootstrap_mean(data, column_name, samples=100):
    bootstrapped = data.sample(n=samples, replace=True)
    return bootstrapped[column_name].mean()


def plot_bootstrap(data, column_name):
    bootstraps = [bootstrap_mean(data, column_name, samples=100) for i in range(10000)]
    bootstraps = np.array(bootstraps)
    plt.hist(bootstraps, bins=50, density=True)
    plt.axvline(np.quantile(bootstraps, 0.01), color="red")
    plt.axvline(np.quantile(bootstraps, 0.99), color="red")
    plt.title("Bootstrapped mean of " + column_name)
    print(
        "Width of CI: ",
        round(np.quantile(bootstraps, 0.95) - np.quantile(bootstraps, 0.05), 4),
    )
    print("Mean of CI:  ", round(np.mean(bootstraps), 4))
    print("Std of CI:   ", round(np.std(bootstraps), 4))
    print("Q1:          ", round(np.quantile(bootstraps, 0.05), 4))
    print("Q99:         ", round(np.quantile(bootstraps, 0.95), 4))
    print("Skewness:    ", round(pd.Series(bootstraps).skew(), 4))
    print("Kurtosis:    ", round(pd.Series(bootstraps).kurtosis(), 4))


def plot_risk_factors(
    base_value,
    capital_damages,
    business_disruption,
    fair_premium,
    insurance_adjustment,
    npv,
):
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
                "Capital Damages",
                "Business disruptions",
                "Fair insurance premiums",
                "Insurance adjustments",
                "Adjusted Value",
            ],
            textposition="outside",
            text=[
                "{:,.2f}M".format(base_value / 1e6),
                "{:,.2f}M".format(capital_damages / 1e6),
                "{:,.2f}M".format(business_disruption / 1e6),
                "{:,.2f}M".format(fair_premium / 1e6),
                "{:,.2f}M".format(abs(insurance_adjustment) / 1e6),
                "{:,.2f}M".format(npv / 1e6),
            ],
            y=[
                base_value,
                -capital_damages,
                -business_disruption,
                -fair_premium,
                -insurance_adjustment,
                npv,
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    fig.update_layout(
        title="PRISK - Waterfall chart",
        showlegend=False,
        template="simple_white",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Asset value impacts",
        yaxis_title="Value",
        yaxis_tickprefix="$",
        yaxis_tickformat=",",
        yaxis_showgrid=True,
    )
    return fig


def clean_owner_name(owner):
    owner = re.sub(r"\[[^)]*\]", "", owner)
    owner = owner.strip()
    owner = owner.title()
    if owner in indian_firm_mapping:
        owner = indian_firm_mapping[owner]
        return owner
    return owner


def extract_firms(
    assets,
    damage_curves=None,
    leverage_ratios={},
    discount_rate=0.05,
    unit_price=60,
    margin=0.2,
    time_horizon=25,
):
    assets.sort_values("Owner", inplace=True)
    if damage_curves is None:
        damage_curves = continuous_curves
    assets.loc[:, "asset"] = assets.apply(
        lambda x: PowerPlant(
            name=x["Plant / Project name"],
            flood_damage_curve=damage_curves,
            flood_exposure=[
                FloodExposure(return_period, x[return_period])
                for return_period in return_period_columns
                if x[return_period] > 0
            ],
            flood_protection=x["flood_protection"],
            production_path=np.repeat(x["Capacity (MW)"] * 24 * 365, time_horizon),
            replacement_cost=x["Value"],
            unit_price=unit_price,
            discount_rate=discount_rate,
            margin=margin,
        ),
        axis=1,
    )
    list_of_owners = []
    for owners in assets["Owner"].unique():
        if pd.isna(owners):
            continue
        owners = owners.split(";")
        for o in owners:
            o = clean_owner_name(o)
            list_of_owners.append(o)
    list_of_owners = list(OrderedDict.fromkeys(list_of_owners))
    owner_map = {
        owner: Holding(owner, leverage_ratio=leverage_ratios.get(owner))
        for owner in list_of_owners
    }
    holdings = []
    for i, owner in enumerate(assets["Owner"]):
        if pd.isna(owner):
            continue
        for o in owner.split(";"):
            share = re.findall(r"\[(.*?)\]", o)
            if share:
                share = float(share[0].replace("%", "")) / 100
            else:
                share = 1
            holding = owner_map[clean_owner_name(o)]
            holding.add_asset(assets.loc[i, "asset"], share)
            holdings.append(holding)
    return list(OrderedDict.fromkeys(holdings))


def link_basins(data, basins, basin_outlet_file, visualize=True, save=False):
    geo_data = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.Longitude, data.Latitude),
        crs="EPSG:4326",
    )
    get_colors = lambda n: [
        (50 / 256, 100 / 256, np.random.choice(range(150)) / 256) for _ in range(n)
    ]
    basins = gpd.read_file(basins)
    basins.loc[:, "color"] = get_colors(len(basins))
    country_basins = pd.read_csv(basin_outlet_file).HYBAS_ID.to_list()
    basins = basins[basins.HYBAS_ID.isin(country_basins)]
    data_merged = geo_data.sjoin(basins[["HYBAS_ID", "geometry"]], how="left")
    data_merged.loc[:, "HYBAS_ID"] = data_merged.HYBAS_ID.apply(
        lambda x: str(int(x)) if not pd.isnull(x) else pd.NA
    )
    if visualize:
        basins.plot(color=basins.color, figsize=(20, 20))
        plt.scatter(data.Longitude, data.Latitude, c="red", s=50)
        if save:
            plt.savefig("map.png", transparent=True)
    return data_merged, basins


def merton_probability_of_default(V, sigma_V, D, r=0, T=1):
    """
    Calculate the probability of default using the Merton model.

    Parameters:
    V (float): Current value of the company's assets.
    sigma_V (float): Volatility of the company's assets.
    D (float): Face value of the company's debt.
    r (float): Risk-free interest rate.
    T (float): Time to maturity of the debt.

    Returns:
    float: Probability of default.
    """
    # Calculate d2
    d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    # Calculate the probability of default
    PD = norm.cdf(-d2)
    return PD


def non_homogeneous_poisson_events(
        discharge_data: pd.DataFrame, 
        base_yearly_arrival_rate: 
        float, years: int
    ):
    normalized_discharge = discharge_data / discharge_data.mean()
    normalized_discharge = np.tile(normalized_discharge.to_numpy(), (years, 1))
    random_numbers = np.random.random(normalized_discharge.shape)
    max_normalized_discharge = np.max(normalized_discharge, axis=0, keepdims=True)
    poisson_thresholds = poisson.ppf(
        random_numbers, max_normalized_discharge * (base_yearly_arrival_rate / 365)
    )
    acceptance_probs = normalized_discharge / max_normalized_discharge
    rejection_randoms = np.random.random(normalized_discharge.shape)
    poisson_thresholds[rejection_randoms > acceptance_probs] = 0
    # Convert results to a DataFrame
    event_data = pd.DataFrame(
        poisson_thresholds, columns=discharge_data.columns
    ).replace(0, pd.NA)
    simulated_data = event_data.reset_index().melt(id_vars="index").dropna()
    simulated_data["return_period"] = 1 / base_yearly_arrival_rate
    simulated_data.columns = ["day", "basin", "events", "return_period"]
    simulated_data.sort_values("day", inplace=True)
    return simulated_data.reset_index(drop=True)

def non_homogenous_poisson_hawkes_events(
    discharge_data: pd.DataFrame,
    base_yearly_arrival_rate: float,
    years: int,
    tc_event_times: list,
    tc_event_geometries: list,
    days_increased_lambda: int = 5,
    storm_presence: float = 0.004738338422659787
) -> pd.DataFrame:
    """
    Simulates flood events using a non-homogeneous Poisson process with Hawkes excitation
    based on cyclone events intersecting flood basins.

    Parameters:
        discharge_data (pd.DataFrame): Daily discharge data per basin.
        base_yearly_arrival_rate (float): Baseline flood arrival rate per year.
        years (int): Number of years to simulate.
        alpha_params (pd.DataFrame): DataFrame with 'basin' and 'alpha' columns.
        basin_geometries (dict): Dictionary mapping basin ids to geometries.
        tc_event_times (list): List of cyclone landfall days.
        tc_event_geometries (list): List of cyclone geometries per event.
        days_increased_lambda (int): Number of days cyclone influence persists.
        storm_presence (float): Average daily probability of a cyclone.

    Returns:
        pd.DataFrame: Simulated flood event records with Hawkes flag.
    """
    basins = discharge_data.columns
    normalized_discharge = (discharge_data / discharge_data.mean()).to_numpy()
    normalized_discharge = np.tile(normalized_discharge, (years, 1))
    random_draws = np.random.random(normalized_discharge.shape)
    max_norm_discharge = np.max(normalized_discharge, axis=0, keepdims=True)

    events = []

    for i, basin in enumerate(basins):
        basin_id = int(basin)
        alpha = alpha_params.loc[alpha_params['basin'] == basin_id, 'alpha'].values[0]
        bg_rate = base_yearly_arrival_rate / (
            (1 - storm_presence * days_increased_lambda) + alpha * storm_presence * days_increased_lambda
        )
        daily_rate = bg_rate / 365
        thresholds = poisson.ppf(random_draws[:, i], max_norm_discharge[0, i] * daily_rate)
        flood_days = np.where(thresholds >= 1)[0]

        for day in flood_days:
            events.append({
                "day": day,
                "basin": basin_id,
                "events": 1,
                "return_period": 1 / base_yearly_arrival_rate,
                "hawkes": False,
                "bg_rate": bg_rate,
                "alpha_parameter": alpha
            })

        if alpha <= 1.0:
            continue

        hawkes_rate = bg_rate * (alpha - 1)
        hawkes_daily_rate = hawkes_rate * days_increased_lambda / 365

        for event_time, geometry in zip(tc_event_times, tc_event_geometries):
            if basin_geometries[basin_id].intersects(geometry):
                if poisson.rvs(hawkes_daily_rate) > 0:
                    day_offset = np.random.randint(0, days_increased_lambda - 1)
                    events.append({
                        "day": event_time + day_offset,
                        "basin": basin_id,
                        "events": 1,
                        "return_period": 1 / base_yearly_arrival_rate,
                        "hawkes": True,
                        "bg_rate": bg_rate,
                        "alpha_parameter": alpha
                    })
    if len(events) == 0:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values("day").reset_index(drop=True)
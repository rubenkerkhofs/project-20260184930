from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import pickle


from climada.hazard.trop_cyclone.trop_cyclone_windfields import compute_windfields_sparse
from climada.hazard import Centroids

from prisk.kernel.message import TropicalCycloneEvent

TC_BASE_FREQUENCIES: dict = {
    -1: 0.21531100478468898*365.25,
    0: 0.3284671532846716*365.25,
    1: 1.5517241379310345*365.25,
    2: 1.956521739130435*365.25,
    3: 4.5*365.25,
    4: 2.5*365.25,
    5: 15.0*365.25}

SYNTHETIC_TRACKS_DF = pd.read_parquet("https://riskconcile-courses.s3.eu-central-1.amazonaws.com/synthetic_tracks.parquet")

class TropicalCycloneSim:
    """
        The tropical cyclone simulation allows the simulation of 
        tropical cyclone paths
    """
    def __init__(
            self, 
            assets: gpd.GeoDataFrame,
            asset_objects: Optional[list] = None,
            model: str = "poisson",
            random_seed: Optional[int] = None
    ):
        self.random_seed=random_seed
        self.model = model
        self.tracks = SYNTHETIC_TRACKS_DF
        self.assets = assets
        self.cent = Centroids.from_geodataframe(assets)
        self.asset_objects = asset_objects if asset_objects is not None else []
        self.event_times = []
        self.event_geometries = []


    def _simulate_non_homogeneous_poisson(self, time_horizon: float, kernel):
        if self.random_seed:
            np.random.seed(self.random_seed)
        for cat, freq in TC_BASE_FREQUENCIES.items():
            time = np.random.exponential(freq)
            while time < time_horizon:
                track = self.tracks[self.tracks.category == cat].sample().to_dict(orient='records')[0]
                track_day_of_year= track['time'].dayofyear
                track_time = int(time % 365 + track_day_of_year + np.random.uniform(-10, 10))
                track = track['sid']
                with open(f'tc_tracks/{track}.pkl', 'rb') as file_name:
                    track = pickle.load(file_name)
                # Get the track path
                points = gpd.points_from_xy(track['lon'], track['lat'])
                self.event_geometries.append(
                    gpd.GeoSeries(points).set_crs("EPSG:4326").to_crs(self.assets.crs).union_all()
                )
                intensities = compute_windfields_sparse(
                    track, 
                    centroids=self.cent, 
                    idx_centr_filter=np.arange(self.cent.size), 
                    model='ER11',
                    store_windfields=False)[0]
                wind_speeds = pd.DataFrame(
                    {'centroid_id': intensities.indices, 'wind_speed': intensities.data}
                )
                self.event_times.append(track_time)
                for _, wind_speed in wind_speeds.iterrows():
                    if wind_speed['wind_speed'] > 25 and self.asset_objects:
                        asset = self.asset_objects[int(wind_speed['centroid_id'])]
                        TropicalCycloneEvent(time=track_time, wind_speed=wind_speed['wind_speed'], asset=asset).send(kernel)
                time += np.random.exponential(freq)


    def _simulate_poisson(self, time_horizon: float, kernel):
        """
            Simulates the occurrence of a tropical cyclone event using
            a Poisson model.

        Parameters
        ----------
        time_horizon : float
            The time horizon of the simulation in years
        """
        if self.random_seed:
            np.random.seed(self.random_seed)

        for cat, freq in TC_BASE_FREQUENCIES.items():
            time = np.random.exponential(freq)
            while time < time_horizon:
                track = self.tracks[self.tracks.category == cat].sample().to_dict(orient='records')[0]['sid']
                # Read track using pickle
                with open(f'tc_tracks/{track}.pkl', 'rb') as file_name:
                    track = pickle.load(file_name)
                intensities = compute_windfields_sparse(
                    track, 
                    centroids=self.cent, 
                    idx_centr_filter=np.arange(self.cent.size), 
                    model='ER11',
                    store_windfields=False)[0]
                wind_speeds = pd.DataFrame(
                    {'centroid_id': intensities.indices, 'wind_speed': intensities.data}
                )
                for _, wind_speed in wind_speeds.iterrows():
                    if wind_speed['wind_speed'] > 25:
                        asset = self.asset_objects[int(wind_speed['centroid_id'])]
                        TropicalCycloneEvent(time=time, wind_speed=wind_speed['wind_speed'], asset=asset).send(kernel)
                time += np.random.exponential(freq)

    def simulate(self, time_horizon: float, kernel):
        """Simulate the tcs

        Parameters
        ----------
        time_horizon : float
            The time horizon of the simulation in years
        """
        if self.model == "poisson":
            self._simulate_non_homogeneous_poisson(time_horizon, kernel)


if __name__ == '__main__':
    data = pd.read_excel('data/supply_chains/geo_data.xlsx')
    data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude), crs="EPSG:4326")
    data.loc[:, 'Longitude'] = data.geometry.x
    data.loc[:, 'Latitude'] = data.geometry.y
    sim = TropicalCycloneSim(
        assets=data,
        random_seed=12)
    sim.simulate(
        time_horizon=25*365.25,
        kernel=None
    )
    print(sim.event_times)
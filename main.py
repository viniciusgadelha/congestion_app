"""
author: VINICIUS - CITCEA
contact: vinicius.gadelha@upc.edu
date: 10/2024

Main python file to run the congestion application server. It requires as inputs a pandapower network and forecast
of its loads and generation to run a timeseries powerflow and check for congestions. testeando um cambio
"""
import datetime
import os

import pandapower as pp
from profile_creation import *
from pp_timeseries import timeseries_pf
from pandapower.plotting import pf_res_plotly, simple_plotly
from plot import plot_grid
from minio_api import minio_get
from datetime import timedelta
from forecaster.test_nn import run_forecast

from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
set_mapbox_token('pk.eyJ1IjoidmluaWNpdXNnYWQiLCJhIjoiY2xseXp6d3llMHZ6azNrczZmc3JqdzJhYyJ9.TEYsgM85dVun46w_lpOKrw')

if __name__ == "__main__":

    # variables definition
    # path_net = 'grids/ct217_grid_model_v0.3.xlsx'
    path_net = 'grids/ct941_grid_model_v0.4.xlsx'

    net = pp.from_excel(path_net)
    output_path = 'results/time_series_results'

    # retrieving SM data from MiniO for yesterday, cleaning data and saving in inputs
    # yesterday = datetime.today() - timedelta(days=1)
    # yesterday = yesterday.strftime('%Y_%m_%d')
    # minio_get(yesterday)

    # running forecasting algorithm
    os.chdir('forecaster')
    run_forecast('centralized')
    os.chdir('..')

    # path_forecast = 'inputs/Profiles_load_CT217_one_week_2.xlsx'
    path_forecast = 'inputs/' + yesterday + '.xlsx'

    # creating profiles and matching SM data with pandapower network loads
    load_df, ds, time_range = create_profiles(net, path=path_forecast)

    # run time series power flow
    time_step = range(0, time_range.__len__())
    timeseries_pf(net, time_step, output_path, ds, path_forecast)

    # plotting power flow results of the heaviest hour on map

    pp.runpp(net, numba=False)
    pf_res_plotly(net, figsize=1.5, on_map=True, map_style='streets')
    # pf_res_plotly(net)
    #
    #
    plot_grid(net, tree=True)
    net = pp.plotting.create_generic_coordinates(net, overwrite=True)
    # pf_res_plotly(net, figsize=2.5)

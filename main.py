"""
author: VINICIUS - CITCEA
contact: vinicius.gadelha@upc.edu
date: 10/2024

Main python file to run the congestion application server. It requires as inputs a pandapower network and forecast
of its loads and generation to run a timeseries powerflow and check for congestions. testeando um cambio
"""

import pandapower as pp
from profile_creation import *
from pp_timeseries import timeseries_pf
from pandapower.plotting import pf_res_plotly, simple_plotly
from plot import plot_grid

from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
set_mapbox_token('pk.eyJ1IjoidmluaWNpdXNnYWQiLCJhIjoiY2xseXp6d3llMHZ6azNrczZmc3JqdzJhYyJ9.TEYsgM85dVun46w_lpOKrw')

if __name__ == "__main__":

    # variables definition
    # path_net = 'grids/ct217_grid_model_v0.3.xlsx'
    path_net = 'grids/ct941_grid_model_v0.3.xlsx'
    net = pp.from_excel(path_net)
    # net.sgen.p_mw = 0  # set PVs to zero since we are not using exported energy yet
    net.load.scaling = 3
    output_path = 'results/time_series_results'

    # creating profiles and matching SM data with pandapower network loads
    # load_df, ds, time_range = create_profiles(net)
    load_df, ds, time_range = create_profiles(net, path='inputs/Most_Loaded_Week_CT941.xlsx')

    # run time series power flow
    time_step = range(0, time_range.__len__())
    timeseries_pf(net, time_step, output_path, ds)

    # plotting power flow results of the heaviest hour on map

    net.load.p_mw = load_df.loc[142, :].values/1000
    pp.runpp(net)
    pf_res_plotly(net, figsize=1.5, on_map=True, map_style='streets')


    plot_grid(net)

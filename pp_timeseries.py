import os
import numpy as np
import pandas as pd
import plotly.express as px
import pandapower as pp
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl

np.seterr(divide='ignore', invalid='ignore')


def timeseries_pf(net, time_step, output_path, ds, flex_activation=False):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 1. create controllers (to control P values of the load and the sgen)
    create_controllers(net, ds, flex_activation)

    # 2. the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, time_step, output_dir=output_path)

    # 3. run the main time series function
    run_timeseries(net, time_step, run=pp.runpp)

    plot_timeseries(output_path)

    return


def create_controllers(net, ds, flex_activation=False):

    # for i in net.sgen.index:
    #     ConstControl(net, element='sgen', variable='p_mw', element_index=i,
    #                  data_source=ds, profile_name=["sgen_"+str(net.sgen.name[i])], scale_factor=0.001)

    for i in net.load.index:
        ConstControl(net, element='load', variable='p_mw', element_index=i,
                     data_source=ds, profile_name=["load_" + str(i)], scale_factor=0.001)

    for i in net.load.index:
        ConstControl(net, element='load', variable='q_mvar', element_index=i,
                     data_source=ds, profile_name=["load_q" + str(i)], scale_factor=0.001)

    for i in net.sgen.index:
        ConstControl(net, element='sgen', variable='p_mw', element_index=i,
                     data_source=ds, profile_name=["sgen_" + str(i)], scale_factor=0.001)

    if flex_activation:
        flex_request = pd.read_csv('results/flex_request.csv', index_col=0)
        if not flex_request.empty:
            flex_buses = list(set([int(i.split('_')[-1]) for i in flex_request.columns]))
            for bus in flex_buses:
                pp.create_sgen(net, bus, 0, 0, name="Flex_Bus_Q_" + str(bus))
                ConstControl(net, element='sgen', variable='q_mvar', element_index=net.sgen.loc[net.sgen.bus == bus].index,
                             data_source=ds, profile_name=["Flex_Bus_Q_"+str(bus)], scale_factor=1)

                ConstControl(net, element='sgen', variable='p_mw', element_index=net.sgen.loc[net.sgen.bus == bus].index,
                             data_source=ds, profile_name=["Flex_Bus_P_"+str(bus)], scale_factor=1)
    return


def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop

    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    ow.log_variable('res_bus', 'p_mw')
    ow.log_variable('res_bus', 'q_mvar')
    ow.log_variable('res_line', 'pl_mw')
    ow.log_variable('res_line', 'ql_mvar')
    ow.log_variable('res_line', 'p_from_mw')
    ow.log_variable('res_line', 'q_from_mvar')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_trafo', 'pl_mw')
    ow.log_variable('res_trafo', 'ql_mvar')
    ow.log_variable('res_trafo', 'p_hv_mw')
    ow.log_variable('res_trafo', 'q_hv_mvar')
    ow.log_variable('res_trafo', 'loading_percent')
    return ow


def plot_timeseries(output_dir):

    # data = pd.read_excel('inputs/Profiles_load_CT217_one_week.xlsx', index_col='dataLectura')

    data = pd.read_excel('inputs/Most_Loaded_Week_CT941.xlsx', index_col='dataLectura')

    # voltage results
    vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
    vm_pu = pd.read_excel(vm_pu_file, index_col=0)
    # vm_pu = vm_pu.loc[:, vm_pu.std() > .005]  # filtering in case plots are too dense

    vm_pu.index = data.index  # change the index back to datetime
    fig = px.line(vm_pu, y=vm_pu.columns,
                  labels={"value": "count", "variable": "Bus ID"})
    fig.update_xaxes(title='Hour', fixedrange=True)
    fig.update_yaxes(title='Voltage [p.u.]')
    fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Voltage: %{y}')
    fig.write_html("voltage_results.html")
    fig.show()

    # trafo loading results (not necessary for now)
    # trafo_file = os.path.join(output_dir, "res_trafo", "loading_percent.xlsx")
    # trafo_loading = pd.read_excel(trafo_file, index_col=0)
    # fig = px.line(trafo_loading, x=trafo_loading.index, y=trafo_loading.columns,
    #               labels={"value": "count", "variable": "Transformer ID"})
    # fig.update_xaxes(title='Hour', fixedrange=True)
    # fig.update_yaxes(title='Loading [%]')
    # fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Loading: %{y}')
    # fig.show()

    # line loading results
    ll_file = os.path.join(output_dir, "res_line", "loading_percent.xlsx")
    line_loading = pd.read_excel(ll_file, index_col=0)
    line_loading.index = data.index  # change the index back to datetime
    # line_loading = line_loading.loc[:, line_loading.std() > 1]  # filtering in case plots are too dense
    fig = px.line(line_loading, y=line_loading.columns,
                  labels={"value": "count", "variable": "Line ID"})
    fig.update_xaxes(title='Hour', fixedrange=True)
    fig.update_yaxes(title='Loading [%]')
    fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Loading: %{y}')
    fig.write_html("line_loading.html")
    fig.show()


    # load results
    load_file = os.path.join(output_dir, "res_load", "p_mw.xlsx")
    load = pd.read_excel(load_file, index_col=0)
    load.index = data.index  # change the index back to datetime
    # load = load.loc[:, load.std() > .001]   # filtering in case plots are too dense
    load = load*1000
    fig = px.line(load, y=load.columns,
                  labels={"value": "count", "variable": "Load ID"})
    fig.update_xaxes(title='Hour', fixedrange=True)
    fig.update_yaxes(title='Power [kW]')
    fig.update_traces(mode="markers+lines", hovertemplate='Hour: %{x} <br>Power: %{y}')
    fig.show()
    fig.write_html("load_results.html")
    return

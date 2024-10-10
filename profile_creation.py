# -*- coding: utf-8 -*-
"""
API CONNECTION SCRIPT - Retrieve FORECASTS data from ICOM servers
FEVER FORECAST SERVICE
https://data-services.dsotoolbox.fever.intracom-telecom.com/swagger-ui/index.html?urls.primaryName=forecast-service#/forecast-service-controller/retrieveUsingGET
"""
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pandapower.timeseries import DFData


def create_profiles(net, path='inputs/Profiles_load_CT217_one_week.xlsx', ):

    # loading json datasets with the forecasting data #
    data = pd.read_excel(path, index_col='dataLectura')
    load_df = pd.DataFrame(index=data.index, columns=['load_' + str(i) for i in net.load.index])
    gen_df = pd.DataFrame(index=data.index, columns=['sgen_' + str(i) for i in net.sgen.index])

    received_ids = data.columns
    list_loads_net = list(net.load.SM.astype(str))
    list_gen_net = list()

    data = check_id_matching(net, data, received_ids, list_gen_net, list_loads_net)

    for load_id in data:
        net_index = net.load[net.load.SM == load_id].index[0]
        load_df['load_' + str(net_index)] = data[str(load_id)]

    time_range = data.index
    load_df.reset_index(inplace=True, drop=True)
    ds = DFData(load_df)

    return load_df, ds, time_range


def check_id_matching(net, load_data, received_ids, list_gen_net, list_loads_net, fill=True):

    list_ids_net = list_gen_net + list_loads_net

    if len(set(list_ids_net) - set(received_ids)) > 0:

        mismatch = list(sorted(set(list_ids_net) - set(received_ids)))

        print('WARNING: no data available for the following IDs: ', mismatch)

    if len(set(received_ids) - set(list_ids_net)) > 0:
        mismatch = list(sorted(set(received_ids) - set(list_ids_net)))

        print('WARNING: the following received IDs are mapped but not present in the virtual grid: ', mismatch)

        for i in mismatch:
            if str(i) in load_data.columns:
                load_data.drop(str(i), axis=1, inplace=True)
        print('Exceeding IDs successfully removed')

    return load_data


def fill_missing_ids(mismatch, gen_set, load_set, net):
    profiles = pd.read_excel('inputs/sgen_profile.xlsx', index_col=0, usecols=[0, 3, 7])
    for i in mismatch:

        if i in list(net.sgen.name):
            fill = pd.DataFrame(columns=gen_set.columns)
            fill.timestamp = gen_set.timestamp.sort_values().unique()
            fill.assetId = str(i)
            fill.type = 'generation'
            fill['electricalEnergyExported'] = net.sgen.loc[net.sgen['name'] == i, 'p_mw'].values[0]*profiles.iloc[:, 0]
            fill.fillna(0, inplace=True)
            gen_set = pd.concat([gen_set, fill])

        elif i in list(net.load.name):
            fill = pd.DataFrame(columns=load_set.columns)
            fill.timestamp = load_set.timestamp.sort_values().unique()
            fill.type = 'demand'
            fill.assetId = str(i)
            fill['electricalEnergyImported'] = net.load.loc[net.load['name'] == i, 'p_mw'].values[0]*profiles.iloc[:, 1]
            fill['value'] = net.load.loc[net.load['name'] == i, 'p_mw'].values[0]*profiles.iloc[:, 1]
            fill.fillna(0, inplace=True)
            load_set = pd.concat([load_set, fill])

    gen_set.reset_index(inplace=True, drop=True)
    load_set.reset_index(inplace=True, drop=True)
    print('Missing values have successfully been filled')
    return gen_set, load_set

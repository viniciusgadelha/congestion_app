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


def create_profiles(net, path='inputs/Profiles_load_CT217_one_week_2.xlsx', ):

    # loading json datasets with the forecasting data #
    load_import = pd.read_excel(path, index_col='dataLectura', sheet_name='Import')
    load_export = pd.read_excel(path, index_col='dataLectura', sheet_name='Export')
    load_r1 = pd.read_excel(path, index_col='dataLectura', sheet_name='r1')
    load_r2 = pd.read_excel(path, index_col='dataLectura', sheet_name='r2')
    load_r3 = pd.read_excel(path, index_col='dataLectura', sheet_name='r3')
    load_r4 = pd.read_excel(path, index_col='dataLectura', sheet_name='r4')

    load_reactive = load_r1 - load_r2 + load_r3 - load_r4

    if load_import.isna().sum().any() or load_export.isna().sum().any() or load_reactive.isna().sum().any():
        print('There some NaN values in the SM data, these will be filled with zero.')
        load_import.fillna(0, inplace=True)
        load_export.fillna(0, inplace=True)
        load_reactive.fillna(0, inplace=True)

    load_df = pd.DataFrame(index=load_import.index, columns=['load_' + str(i) for i in net.load.index])
    sgen_df = pd.DataFrame(index=load_import.index, columns=['sgen_' + str(i) for i in net.sgen.index])
    load_df_reactive = pd.DataFrame(index=load_reactive.index, columns=['load_q' + str(i) for i in net.load.index])

    received_ids = load_import.columns
    list_loads_net = list(net.load.SM.astype(str))
    list_gen_net = list(net.sgen.SM.astype(str))

    load_import, load_reactive, load_export = check_id_matching(net, load_import, received_ids, list_gen_net, list_loads_net, load_reactive, load_export)

    for load_id in load_import:
        net_index = net.load[net.load.SM == load_id].index[0]
        load_df['load_' + str(net_index)] = load_import[str(load_id)]

    for sgen_id in load_export:
        net_index = net.sgen[net.sgen.SM == sgen_id].index[0]
        sgen_df['sgen_' + str(net_index)] = load_export[str(sgen_id)]

    for load_id in load_reactive:
        net_index = net.load[net.load.SM == load_id].index[0]
        load_df_reactive['load_q' + str(net_index)] = load_reactive[str(load_id)]

    time_range = load_import.index

    load_df.reset_index(inplace=True, drop=True)
    load_df_reactive.reset_index(inplace=True, drop=True)
    sgen_df.reset_index(inplace=True, drop=True)
    profiles_df = pd.concat([load_df, load_df_reactive], axis=1)
    profiles_df = pd.concat([profiles_df, sgen_df], axis=1)
    ds = DFData(profiles_df)

    return load_df, ds, time_range


def check_id_matching(net, load_data, received_ids, list_gen_net, list_loads_net, load_reactive_data, load_exported_data, fill=True):

    list_ids_net = list_gen_net + list_loads_net

    if len(set(list_ids_net) - set(received_ids)) > 0:

        mismatch = list(sorted(set(list_ids_net) - set(received_ids)))

        print('WARNING: no data available for the following IDs: ', mismatch)

    if len(set(received_ids) - set(list_loads_net)) > 0:
        mismatch = list(sorted(set(received_ids) - set(list_loads_net)))

        print('WARNING: the following received IDs are mapped but not present in the loads of virtual grid: ', mismatch)

        for i in mismatch:
            if str(i) in load_data.columns:
                load_data.drop(str(i), axis=1, inplace=True)
                load_reactive_data.drop(str(i), axis=1, inplace=True)
        print('Exceeding IDs successfully removed from load data')

    if len(set(received_ids) - set(list_gen_net)) > 0:
        mismatch = list(sorted(set(received_ids) - set(list_gen_net)))

        print('WARNING: the following received IDs are mapped but not present in the generators of the virtual grid: ', mismatch)

        for i in mismatch:
            if str(i) in load_exported_data.columns:
                load_exported_data.drop(str(i), axis=1, inplace=True)
        print('Exceeding IDs successfully removed from generator data')

    return load_data, load_reactive_data, load_exported_data


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

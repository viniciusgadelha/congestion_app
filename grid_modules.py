import pandapower as pp
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from pandapower.plotting import simple_plot
import math



###Add here the path to the template file

# path = '../data/Templates/S3.1-Template_Input_Data_v4_Estabanell - vFinal.xlsx'
# path2 = '../data/Templates/S3.1-Template_Input_Data_OEDAS2.xlsx'
# path3 = '../data/Templates/Casee33_Template_PPT.xlsx'

# Creates the net object with the elements from each sheet in the template: lines, transformers and allocates loads and generation.

def create_net(path):
    '''Function to create net object in pandapower (v2.4.0) from excel template

    external grid is automatically added, with default parameters.

    Lines and transformers are created from conventions

    Elements magnitudes follow the convetion of pandapower (check 'Metadata' sheet in template)

    Ex:
    net = create_net('Input_Template_filled.xlsx')
    '''

    net = pp.create_empty_network()
    df_bus = pd.read_excel(path, sheet_name="Bus", index_col=0).dropna()
    df_lines = pd.read_excel(path, sheet_name="Lines", index_col=0).dropna()
    df_trafos = pd.read_excel(path, sheet_name="Transformers", index_col=0).dropna()
    df_loads = pd.read_excel(path, sheet_name="Loads - alloc", index_col=0).dropna()
    df_gens = pd.read_excel(path, sheet_name='Generators - alloc', index_col=0).dropna()

    # Bus Generation
    for index in df_bus.index:
        pp.create_bus(net, vn_kv=df_bus.at[index, "Bus Voltage (kV)"],
                      name=str(df_bus.at[index, "Bus Name"]),
                      # max_vm_pu=df_bus.at[index, "max_vm_pu"],
                      # min_vm_pu=df_bus.at[index, "min_vm_pu"]
                      )

    # Lines Generation from parameters
    for index in df_lines.index:
        # pp.create_line(net, from_bus=df_lines.at[index, "Bus From"], to_bus=df_lines.at[index, "Bus to"],
        #                length_km=df_lines.at[index, "Lenght (km)"], std_type=df_lines.at[index, "type"],
        #                name=df_lines.at[index, "Line ID"])

        pp.create_line_from_parameters(net, from_bus=df_lines.at[index, "Bus from"],
                                       to_bus=df_lines.at[index, "Bus to"],
                                       length_km=df_lines.at[index, "Length (km)"],
                                       r_ohm_per_km=df_lines.at[index, "R (ohm/km)"],
                                       x_ohm_per_km=df_lines.at[index, "X (ohm/km)"],
                                       c_nf_per_km=df_lines.at[index, "C (nF/km)"],
                                       max_i_ka=df_lines.at[index, "Max Ik (kA)"],
                                       name=df_lines.at[index, "Line ID"])

        ###create function to create line from standard type
    #
    # External grid, will always be allocated at bus index 0 (Slack bus)
    # pp.create_ext_grid(net, 0, voltage = 1.00,s_sc_max_mva= 1000, rx_max=0.1,r0x0_max=0.1,x0x_max= 1.0 )
    #
    pp.create_ext_grid(net, 0)

    # Generación trafos

    for index in df_trafos.index:
        pp.create_transformer_from_parameters(net, hv_bus=df_trafos.at[index, "HV Bus"],
                                              lv_bus=df_trafos.at[index, "LV Bus"],
                                              sn_mva=df_trafos.at[index, "Sn (MVA)"],
                                              vn_hv_kv=df_trafos.at[index, "HV (kV)"],
                                              vn_lv_kv=df_trafos.at[index, "LV (kV)"],
                                              vk_percent=df_trafos.at[index, "Vk (%)"],
                                              vkr_percent=df_trafos.at[index, "Vk_r (%)"],
                                              pfe_kw=df_trafos.at[index, "P Losses (kW)"],
                                              i0_percent=df_trafos.at[index, "I0 Losses (%)"],
                                              shift_degree=df_trafos.at[index, "Shift Degree"],
                                              name=str(df_trafos.at[index, "Trafo Name"]),
                                              max_loading_percent=df_trafos.at[index, "Max Loading (%)"])

    # Loads allocation
    for index in df_loads.index:
        # pp.create_load_from_cosphi(net, df_loads.at[index, "Bus"],
        #                            df_loads.at[index, "sn_mva"], cos_phi=1,
        #                            mode="ind")

        pp.create_load(net, df_loads.at[index, "Bus"],
                       df_loads.at[index, "Pd"],
                       df_loads.at[index, "Qd"],
                       name=df_loads.at[index, "Name"])

    for index in df_gens.index:
        pp.create_sgen(net, df_gens.at[index, "Bus"],
                       df_gens.at[index, 'P (MW)'],
                       df_gens.at[index, 'Q (MVAR)'],
                       name=df_gens.at[index, 'Name'])

    return net


# net = pn.case33bw()
# for line in range(32, 36):
#     net.line.at[line, 'in_service']=True

def add_type(net, name, parameters, element='line'):
    '''
    Function to create standard types for lines or transformers, specific standard of the pilot's grid. This function is
    specially useful for lines, since a considerable number of lines are of the same type, the user will only need to add the
    distance of the cables.
    :return:the standard type of transformer or line of the grid.
    '''
    pp.create_std_type(net, parameters, name, element)


def plot_grid(net):
    '''
    Draws a simple pandapower plot of the created net
    :param net:
    :return: simple_plot from pandapower.
    '''
    simple_plot(net)
    # simple_plotly(net)
    # pf_res_plotly(net)


# Construye la matriz Ybus a partir de una red
def Ybus(net):
    from_bus, to_bus, km_line, r_line, x_line, c_line, info = [], [], [], [], [], [], []
    for fbus in net.line["from_bus"]:
        from_bus.append(fbus)
    for tbus in net.line["to_bus"]:
        to_bus.append(tbus)
    for km in net.line["length_km"]:
        km_line.append(km)
    for r in net.line["r_ohm_per_km"]:
        r_line.append(r)
    for x in net.line["x_ohm_per_km"]:
        x_line.append(x)
    for c in net.line["c_nf_per_km"]:
        c_line.append(c)
    for fbus, tbus, km, r, x, c in zip(from_bus, to_bus, km_line, r_line, x_line, c_line):
        info.append([fbus, tbus, km, r, x, c])

    nodes = max(net.bus.index) + 1
    Ybus = np.zeros((nodes, nodes), dtype=complex)
    for busi, buso, km, r, x, c in info:
        Ybus[busi][busi] += (1 / (complex(r, x) * km)) + complex(0, 2 * math.pi * 50 * c * 0.000000001) * km / 2
        Ybus[buso][buso] += (1 / (complex(r, x) * km)) + complex(0, 2 * math.pi * 50 * c * 0.000000001) * km / 2
        Ybus[busi][buso] += (-1 / (complex(r, x) * km))
        Ybus[buso][busi] += (-1 / (complex(r, x) * km))

    return Ybus


# Escribe sobre un excel el DataFrame dado
def write_f(df, path, sheet_name):
    book = load_workbook(path)
    writer = pd.ExcelWriter(path, engine='openpyxl')
    writer.book = book
    df.to_excel(writer, sheet_name=sheet_name)
    writer.save()
    writer.close()


def merge_grids(mv_net, lv_net, pcc):
    """
      merge two pandapower networks. Each net must have its own external grid. The connection bus
      must be a LV bus of any transformer on the MV network.
      :param mv_net: medium voltage network
      :param lv_net: low voltage network to be added
      :param pcc: point of common coupling
      :return net: merged pandapower network
    """

    net = pp.merge_nets(mv_net, lv_net)
    mn_lv_bus = net.trafo.tail(1).lv_bus.values[0]
    mn_hv_bus = net.trafo.tail(1).hv_bus.values[0]
    net.trafo.drop(net.trafo.tail(1).index, inplace=True)
    net.ext_grid.drop(index=1, inplace=True)
    net.bus.drop(index=mn_hv_bus, inplace=True)
    pp.fuse_buses(net, pcc, mn_lv_bus)

    return net


def create_unbalance(net, r1, r2, r3):
    """
        Create a load unbalance in the asymmetric loads of the network. The network must have asymmetric loads and the
        summation of the ratios must be 1.
        :param net: pandapower network
        :param r1: ratio for phase A, float [0-1]
        :param r2: ratio for phase B, float [0-1]
        :param r3: ratio for phase B, float [0-1]
        :return net: pandapower network with unbalances
    """
    if r1 + r2 + r3 != 1:
        return net

    total_power = net.asymmetric_load['p_a_mw'] + net.asymmetric_load['p_b_mw'] + net.asymmetric_load['p_c_mw']
    net.asymmetric_load['p_a_mw'] = total_power * r1
    net.asymmetric_load['p_b_mw'] = total_power * r2
    net.asymmetric_load['p_c_mw'] = total_power * r3

    return net


def single_to_three_phase(net, loads=True, sgen=True):
    """
        transforms all assymetric loads and sgen into three-phase symmetric ones
        :param net: pandapower network
        :param loads: boolean value to enable the load conversion
        :param sgen: boolean value to enable the sgen conversion
        :return net: pandapower network without assymetric sgen/load
    """
    if loads:
        for i in net.asymmetric_load.iterrows():
            i = i[1]
            pp.create_load(net, i.bus, i.p_a_mw + i.p_b_mw + i.p_c_mw, i.q_a_mvar + i.q_b_mvar + i.q_c_mvar,
                           name=i.values[0])
        net.asymmetric_load.drop(net.asymmetric_load.index, inplace=True)

    if sgen:
        for i in net.asymmetric_sgen.iterrows():
            i = i[1]
            pp.create_sgen(net, i.bus, i.p_a_mw + i.p_b_mw + i.p_c_mw, i.q_a_mvar + i.q_b_mvar + i.q_c_mvar,
                           name=i.values[0])
        net.asymmetric_sgen.drop(net.asymmetric_sgen.index, inplace=True)

    return net


def sensitivity(net, power="q", delta=0.001):
    """
        Create a load unbalance in the asymmetric loads of the network. The network must have asymmetric loads and the
        summation of the ratios must be 1.
        :param net: pandapower network
        :param power: choosing of power sensitivity, p for active and q for reactive
        :param delta: change in power to estimate sensitivity
        :return sens_matrix: sensitivity matrix related to delta change in power for every bus
    """
    sens_matrix = pd.DataFrame(index=net.bus.index, columns=net.bus.index)
    sens_matrix_normalized = pd.DataFrame(index=net.bus.index, columns=net.bus.index)
    if len(net.res_bus) < 1:
        pp.runpp(net)
    if power == 'q':
        v0 = net.res_bus.vm_pu
        for i in net.bus.index:
            pp.create_sgen(net, i, p_mw=0, q_mvar=delta)
            try:
                pp.runpp(net)
            except:
                print('Power flow cannot converge by adding this amount of power to bus ' + str(i))
            sens_matrix[sens_matrix.columns[i]] = net.res_bus.vm_pu - v0
            net.sgen.drop(net.sgen.tail(1).index, inplace=True)
        sens_matrix.to_excel('results/sens_matrix_' + power + '_' + str(delta) + '.xlsx')
        for i in net.bus.index:
            sens_matrix_normalized.loc[i, :] = (sens_matrix.loc[i, :] / sens_matrix.loc[i, :].max())
        sens_matrix_normalized.to_excel('results/sens_matrix_normalized_' + power + '_' + str(delta) + '.xlsx')

    if power == 'p':
        v0 = net.res_bus.vm_pu
        for i in net.bus.index:
            pp.create_sgen(net, i, p_mw=delta, q_mvar=0)
            try:
                pp.runpp(net)
            except:
                print('Power flow cannot converge by adding this amount of power to bus ' + str(i))
            sens_matrix[sens_matrix.columns[i]] = net.res_bus.vm_pu - v0
            sens_matrix_normalized[sens_matrix_normalized.columns[i]] = (sens_matrix.loc[:, i] / sens_matrix.loc[i, i])
            net.sgen.drop(net.sgen.tail(1).index, inplace=True)
        sens_matrix.to_excel('results/sens_matrix_' + power + '_' + str(delta) + '.xlsx')
        for i in net.bus.index:
            sens_matrix_normalized.loc[i, :] = (sens_matrix.loc[i, :] / sens_matrix.loc[i, :].max())
        sens_matrix_normalized.to_excel('results/sens_matrix_normalized_' + power + '_' + str(delta) + '.xlsx')
    return sens_matrix


def s_power(element):
    """
        :param element: pandapower element (load or generator)
        :return net: apparent power of the element in kVA
    """

    s = complex(element.p_a_mw + element.p_b_mw + element.p_c_mw,
                element.q_a_mvar + element.q_b_mvar + element.q_c_mvar)

    return abs(s), math.atan2(s.imag, s.real)


def switch_coords(net):
    """
        :param net: pandapower network
        :return net: pandapower network with coordinate change
    """
    net.bus_geodata.columns = ['y', 'x', 'coords']

    return net


def sens_jacobian(net, split=True):
    """
        Create the sensitivity matrix using the jacobian matrix of an input pandapower network
        The sensitivity matrix has the following structure:
                    d(Angle)/dP d(Voltage)/dP
                    d(Angle)/dQ d(Voltage)/dP
        :param split: option to split the sensitivity matrix in 4 quadrants
        :param net: pandapower network
        :return sens_matrix: sensitivity matrix calculated from the inverse jacobian
    """
    pp.runpp(net)

    n_buses = net._pd2ppc_lookups["bus"].shape[0]  # amount of buses not accounting for switches
    n_slack = net.ext_grid.shape[0]  # amount of slack buses connected to external grid
    J = net._ppc["internal"]["J"].todense()  # jacobian matrix of net
    sens = pd.DataFrame((np.linalg.pinv(J)))  # inverse of the jacobian, i.e sensitivity matrix

    if split:
        # extracting each quadrant of the sensitivity matrix
        sens_p_delta = sens.loc[0:n_buses - n_slack - 1, 0:n_buses - n_slack - 1]
        sens_p_v = sens.loc[0:n_buses - n_slack - 1, n_buses - n_slack:2 * (n_buses - n_slack)]
        sens_q_delta = sens.loc[n_buses - n_slack:2 * (n_buses - n_slack), 0:n_buses - n_slack - 1]
        sens_q_v = sens.loc[n_buses - n_slack:2 * (n_buses - n_slack), n_buses - n_slack:2 * (n_buses - n_slack)]
        return sens_p_delta, sens_p_v, sens_q_delta, sens_q_v

    return sens


def pp_results_aggregation(path):

    bus_p_mw = pd.read_excel(path + '/res_bus/p_mw.xlsx')
    bus_q_mvar = pd.read_excel(path + '/res_bus/q_mvar.xlsx')
    bus_vm_pu = pd.read_excel(path + '/res_bus/vm_pu.xlsx')
    bus_va_degree = pd.read_excel(path + '/res_bus/va_degree.xlsx')

    line_loading_percent = pd.read_excel(path + '/res_line/loading_percent.xlsx')
    line_p_from_mw = pd.read_excel(path + '/res_line/p_from_mw.xlsx')
    line_q_from_mvar = pd.read_excel(path + '/res_line/q_from_mvar.xlsx')
    line_ql_mvar = pd.read_excel(path + '/res_line/ql_mvar.xlsx')
    line_pl_mw = pd.read_excel(path + '/res_line/pl_mw.xlsx')

    trafo_loading_percent = pd.read_excel(path + '/res_trafo/loading_percent.xlsx')
    trafo_p_hv_mw = pd.read_excel(path + '/res_trafo/p_hv_mw.xlsx')
    trafo_q_hv_mvar = pd.read_excel(path + '/res_trafo/q_hv_mvar.xlsx')
    trafo_ql_mvar = pd.read_excel(path + '/res_trafo/ql_mvar.xlsx')
    trafo_pl_mw = pd.read_excel(path + '/res_trafo/pl_mw.xlsx')

    load_p_mw = pd.read_excel(path + '/res_load/p_mw.xlsx')

    return

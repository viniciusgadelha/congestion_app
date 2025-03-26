import numpy as np
import pandapower as pp
import pandas as pd
import copy
from pandapower.plotting.plotly.mapbox_plot import _on_map_test, _get_mapbox_token, \
    MapboxTokenMissing
from pandapower.plotting.plotly.mapbox_plot import set_mapbox_token
from pandapower.plotting import simple_plot
from collections.abc import Iterable
from plotly.offline import plot as plot
from plotly import __version__ as plotly_version
from plotly.graph_objs.scatter.marker import ColorBar
from plotly.graph_objs import Figure, Layout
from plotly.graph_objs.layout import XAxis, YAxis
from plotly.graph_objs.scatter import Line, Marker
from plotly.graph_objs.scattermapbox import Line as scmLine
from plotly.graph_objs.scattermapbox import Marker as scmMarker
from grid_modules import switch_coords

import plotly.graph_objs as go

set_mapbox_token('pk.eyJ1IjoidmluaWNpdXNnYWQiLCJhIjoiY2xseXp6d3llMHZ6azNrczZmc3JqdzJhYyJ9.TEYsgM85dVun46w_lpOKrw')


def get_hoverinfo(net, element, precision=3, sub_index=None):
    hover_index = net[element].index
    if element == "bus":
        load_str, sgen_str = [], []
        for b in net.bus.index:
            ln_a = net.asymmetric_load.loc[net.asymmetric_load.bus == b, 'p_a_mw'].sum() + \
                   net.load.loc[net.load.bus == b, "p_mw"].sum() / 3
            ln_b = net.asymmetric_load.loc[net.asymmetric_load.bus == b, 'p_a_mw'].sum() + \
                net.load.loc[net.load.bus == b, "p_mw"].sum() / 3
            ln_c = net.asymmetric_load.loc[net.asymmetric_load.bus == b, 'p_a_mw'].sum() + \
                net.load.loc[net.load.bus == b, "p_mw"].sum() / 3
            # load_str.append("Load_a: {:.3f} MW<br />Load_b: "
            #                 "{:.3f} MW<br />Load_c: {:.3f} MW<br />".format(ln_a, ln_b, ln_c)
            #                 if ln_a != 0 or ln_b != 0 or ln_c != 0. else "")
            total_ln = ln_a + ln_b + ln_c
            load_str.append("Load: {:.3f} MW<br />".format(total_ln)
                            if ln_a != 0 or ln_b != 0 or ln_c != 0. else "")

            gen_a = net.asymmetric_sgen.loc[net.asymmetric_sgen.bus == b, 'p_a_mw'].sum() + \
                net.sgen.loc[net.sgen.bus == b, "p_mw"].sum() / 3
            gen_b = net.asymmetric_sgen.loc[net.asymmetric_sgen.bus == b, 'p_a_mw'].sum() + \
                net.sgen.loc[net.sgen.bus == b, "p_mw"].sum() / 3
            gen_c = net.asymmetric_sgen.loc[net.asymmetric_sgen.bus == b, 'p_a_mw'].sum() + \
                net.sgen.loc[net.sgen.bus == b, "p_mw"].sum() / 3
            total_gen = gen_a + gen_b + gen_c
            sgen_str.append("Gen: {:.3f} MW<br />".format(total_gen)
                            if gen_a != 0 or gen_b != 0 or gen_c != 0. else "")
        hoverinfo = (
                "Index: " + net.bus.index.astype(str) + '<br />' +
                "Name: " + net.bus['name'].astype(str) + '<br />' +
                "Zone: " + net.bus['zone'].astype(str) + '<br />' +
                'V_n: ' + net.bus['vn_kv'].round(precision).astype(str) + ' kV' + '<br />' + load_str + sgen_str) \
            .tolist()
    elif element == "line":
        hoverinfo = (
                "Index: " + net.line.index.astype(str) + '<br />' +
                "Name: " + net.line['name'].astype(str) + '<br />' +
                'Length: ' + net.line['length_km'].round(precision).astype(str) + ' km' + '<br />' +
                'R: ' + (net.line['length_km'] * net.line['r_ohm_per_km']).round(precision).astype(str)
                + ' Ohm' + '<br />'
                + 'X: ' + (net.line['length_km'] * net.line['x_ohm_per_km']).round(precision).astype(str)
                + ' Ohm' + '<br />').tolist()
    elif element == "trafo":
        hoverinfo = (
                "Index: " + net.trafo.index.astype(str) + '<br />' +
                "Name: " + net.trafo['name'].astype(str) + '<br />' +
                'S_n: ' + net.trafo['sn_mva'].round(precision).astype(str) + ' MVA' + '<br />' +
                'V_n HV: ' + net.trafo['vn_hv_kv'].round(precision).astype(str) + ' kV' + '<br />' +
                'V_n LV: ' + net.trafo['vn_lv_kv'].round(precision).astype(str) + ' kV' + '<br />').tolist()
    elif element == "switch":
        hoverinfo = (
                "Name: " + net.switch['name'].astype(str) + '<br />' +
                'Close: ' + net.switch['closed'].astype(str) + '<br />' +
                'Type: ' + net.switch['type'].astype(str) + '<br />').tolist()
    elif element == "ext_grid":
        hoverinfo = (
                "Name: " + net.ext_grid['name'].astype(str) + '<br />' +
                'V_m: ' + net.ext_grid['vm_pu'].round(precision).astype(str) + ' p.u.' + '<br />' +
                'V_a: ' + net.ext_grid['va_degree'].round(precision).astype(str) + ' °' + '<br />').tolist()
        hover_index = net.ext_grid.bus.tolist()

    else:
        return None
    hoverinfo = pd.Series(index=hover_index, data=hoverinfo, dtype='object')
    if sub_index is not None:
        hoverinfo = hoverinfo.loc[list(sub_index)]
    return hoverinfo


def create_trafo_trace(net, trafos=None, color='green', width=7, hoverinfo=None, cmap=None,
                       trace_name='trafos', cmin=None, cmax=None, cmap_vals=None,
                       use_line_geodata=None):
    """
    Creates a plotly trace of pandapower trafos.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The trafos for which the collections are created.
        If None, all trafos in the network are considered.

        **width** (int, 5) - line width

        **infofunc** (pd.Series, None) - hoverinfo for trafo elements. Indices should correspond
            to the pandapower element indices

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "green") - color of lines in the trace

        **cmap** (bool, False) - name of a colormap which exists within plotly (Greys, YlGnBu,
            Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot,
            Blackbody, Earth, Electric, Viridis)

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum


    """

    # defining lines to be plot
    trafos = net.trafo.index.tolist() if trafos is None else list(trafos)
    if len(trafos) == 0:
        return []

    trafo_buses_with_geodata = net.trafo.hv_bus.isin(net.bus_geodata.index) & \
                               net.trafo.lv_bus.isin(net.bus_geodata.index)

    trafos_mask = net.trafo.index.isin(trafos)
    trafos_to_plot = net.trafo[trafo_buses_with_geodata & trafos_mask]

    if hoverinfo is not None:
        if not isinstance(hoverinfo, pd.Series) and isinstance(hoverinfo, Iterable) and \
                len(hoverinfo) == len(trafos):
            hoverinfo = pd.Series(index=trafos, data=hoverinfo)
        assert isinstance(hoverinfo, pd.Series), \
            "infofunc should be a pandas series with the net.trafo.index to the infofunc contents"
        hoverinfo = hoverinfo.loc[trafos_to_plot.index]
    trafo_traces = []

    for col_i, (idx, trafo) in enumerate(trafos_to_plot.iterrows()):
        trafo_trace = dict(type='scatter', text=[], line=Line(width=width, color=color),
                           hoverinfo='text', mode='lines', name=trace_name)

        trafo_trace['text'] = trafo['name'] if hoverinfo is None else hoverinfo.loc[idx]

        from_bus = net.bus_geodata.loc[trafo.hv_bus, 'x']
        to_bus = net.bus_geodata.loc[trafo.lv_bus, 'x']
        trafo_trace['x'] = [from_bus, (from_bus + to_bus) / 2, to_bus]

        from_bus = net.bus_geodata.loc[trafo.hv_bus, 'y']
        to_bus = net.bus_geodata.loc[trafo.lv_bus, 'y']
        trafo_trace['y'] = [from_bus, (from_bus + to_bus) / 2, to_bus]

        trafo_traces.append(trafo_trace)
    return trafo_traces


def create_switch_trace(net, switches=None, color='red', width=7, hoverinfo=None, trace_name='switches'):
    """
    Creates a plotly trace of pandapower trafos.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The trafos for which the collections are created.
        If None, all trafos in the network are considered.

        **width** (int, 5) - line width

        **infofunc** (pd.Series, None) - hoverinfo for trafo elements. Indices should correspond
            to the pandapower element indices

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "green") - color of lines in the trace

        **cmap** (bool, False) - name of a colormap which exists within plotly (Greys, YlGnBu,
            Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot,
            Blackbody, Earth, Electric, Viridis)

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum


    """

    # defining lines to be plot
    switches = net.switch.index.tolist() if switches is None else list(switches)
    if len(switches) == 0:
        return []

    switches_buses_with_geodata = net.switch.bus.isin(net.bus_geodata.index) & \
                                  net.switch.element.isin(net.bus_geodata.index)

    switches_mask = net.switch.index.isin(switches)
    switches_to_plot = net.switch[switches_buses_with_geodata & switches_mask]

    if hoverinfo is not None:
        if not isinstance(hoverinfo, pd.Series) and isinstance(hoverinfo, Iterable) and \
                len(hoverinfo) == len(switches):
            hoverinfo = pd.Series(index=switches, data=hoverinfo)
        assert isinstance(hoverinfo, pd.Series), \
            "infofunc should be a pandas series with the net.trafo.index to the infofunc contents"
        hoverinfo = hoverinfo.loc[switches_to_plot.index]
    switch_traces = []

    for col_i, (idx, switch) in enumerate(switches_to_plot.iterrows()):
        switch_trace = dict(type='scatter', text=[], line=Line(width=width, color=color),
                            hoverinfo='text', mode='lines', name=trace_name)

        switch_trace['text'] = switch['name'] if hoverinfo is None else hoverinfo.loc[idx]

        from_bus = net.bus_geodata.loc[switch.bus, 'x']
        to_bus = net.bus_geodata.loc[switch.element, 'x']
        switch_trace['x'] = [from_bus, (from_bus + to_bus) / 2, to_bus]

        from_bus = net.bus_geodata.loc[switch.bus, 'y']
        to_bus = net.bus_geodata.loc[switch.element, 'y']
        switch_trace['y'] = [from_bus, (from_bus + to_bus) / 2, to_bus]

        switch_traces.append(switch_trace)
    return switch_traces


def create_bus_trace(net, buses=None, size=10, patch_type="circle", color="blue", hoverinfo=None,
                     trace_name='buses', legendgroup=None, cmap=None, cmap_vals=None,
                     cbar_title=None, cmin=None, cmax=None, cpos=1.0, colormap_column="vm_pu"):
    """
    Creates a plotly trace of pandapower buses.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **buses** (list, None) - The buses for which the collections are created.
        If None, all buses in the network are considered.

        **size** (int, 5) - patch size

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "square" for a rectangle
                - "diamond" for a diamond
                - much more pathc types at https://plot.ly/python/reference/#scatter-marker

        **infofunc** (pd.Series, None) - hoverinfo for bus elements. Indices should correspond to
            the pandapower element indices

        **trace_name** (String, "buses") - name of the trace which will appear in the legend

        **color** (String, "blue") - color of buses in the trace

        **cmap** (String, None) - name of a colormap which exists within plotly
            (Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow,
            Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis) alternatively a custom
            discrete colormap can be used

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

        **cpos** (float, 1.1) - position of the colorbar

        **colormap_column** (str, "vm_pu") - set color of bus according to this variable

    """

    bus_trace = dict(type='scatter', text=[], mode='markers', hoverinfo='text', name=trace_name,
                     marker=dict(color=color, size=size, symbol=patch_type))

    if hoverinfo is None:
        bus_trace = dict(type='scatter', text=[], mode='markers', hoverinfo='skip', name=trace_name,
                         marker=dict(color=color, size=size, symbol=patch_type))

    buses = net.bus.index.tolist() if buses is None else list(buses)
    bus_plot_index = [b for b in buses if b in list(set(buses) & set(net.bus_geodata.index))]

    bus_trace['x'], bus_trace['y'] = (net.bus_geodata.loc[bus_plot_index, 'x'].tolist(),
                                      net.bus_geodata.loc[bus_plot_index, 'y'].tolist())

    if not isinstance(hoverinfo, pd.Series) and isinstance(hoverinfo, Iterable) and \
            len(hoverinfo) == len(buses):
        hoverinfo = pd.Series(index=buses, data=hoverinfo)

    bus_trace['text'] = net.bus.loc[bus_plot_index, 'name'] if hoverinfo is None else \
        hoverinfo.loc[buses]

    if legendgroup:
        bus_trace['legendgroup'] = legendgroup

    bus_trace['marker'] = Marker(size=size,
                                 color=color,
                                 symbol=patch_type
                                 )

    # bus_trace['marker']['title']['colorbar']['side'] = 'right'

    return [bus_trace]


def create_line_trace(net, lines=None, use_line_geodata=False, respect_switches=False, width=1.0,
                      color='grey', hoverinfo=None, trace_name='lines', legendgroup=None,
                      cmap=None, cbar_title=None, show_colorbar=True, cmap_vals=None, cmin=None,
                      cmax=None, cpos=1.1):
    """
    Creates a plotly trace of pandapower lines.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **lines** (list, None) - The lines for which the collections are created.
        If None, all lines in the network are considered.

        **width** (int, 1) - line width

        **respect_switches** (bool, False) - flag for consideration of disconnected lines

        **infofunc** (pd.Series, None) - hoverinfo for line elements. Indices should correspond to
            the pandapower element indices

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "grey") - color of lines in the trace

        **legendgroup** (String, None) - defines groups of layers that will be displayed in a legend
        e.g. groups according to voltage level (as used in `vlevel_plotly`)

        **cmap** (String, None) - name of a colormap which exists within plotly if set to True default `Jet`
        colormap is used, alternative colormaps : Greys, YlGnBu, Greens, YlOrRd,
        Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis

        **cmap_vals** (list, None) - values used for coloring using colormap

        **show_colorbar** (bool, False) - flag for showing or not corresponding colorbar

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

        **cpos** (float, 1.1) - position of the colorbar

        """

    # defining lines to be plot
    lines = net.line.index.tolist() if lines is None else list(lines)
    if len(lines) == 0:
        return []

    if hoverinfo is not None:
        if not isinstance(hoverinfo, pd.Series) and isinstance(hoverinfo, Iterable) and \
                len(hoverinfo) == len(lines):
            hoverinfo = pd.Series(index=lines, data=hoverinfo)
        if len(hoverinfo) != len(lines) and len(hoverinfo) != len(net.line):
            raise UserWarning("Different amount of hover info than lines to plot")
        assert isinstance(hoverinfo, pd.Series), \
            "infofunc should be a pandas series with the net.line.index to the infofunc contents"

    no_go_lines = set()
    if respect_switches:
        no_go_lines = set(lines) & set(net.switch.element[(net.switch.et == "l") &
                                                          (net.switch.closed == 0)])

    lines_to_plot = net.line.loc[list(set(net.line.index) & (set(lines) - no_go_lines))]
    no_go_lines_to_plot = None

    use_line_geodata = use_line_geodata if net.line_geodata.shape[0] > 0 else False
    if use_line_geodata:
        lines_to_plot = lines_to_plot.loc[set(lines_to_plot.index) & set(net.line_geodata.index)]
    else:
        lines_with_geodata = lines_to_plot.from_bus.isin(net.bus_geodata.index) & \
                             lines_to_plot.to_bus.isin(net.bus_geodata.index)
        lines_to_plot = lines_to_plot.loc[lines_with_geodata]

    line_traces = []
    for col_i, (idx, line) in enumerate(lines_to_plot.iterrows()):
        line_color = color

        line_trace = dict(type='scatter', text=[], hoverinfo='text', mode='lines', name=trace_name,
                          line=Line(width=width, color=color))

        line_trace['x'], line_trace['y'] = _get_line_geodata_plotly(net, lines_to_plot.loc[idx:idx])

        line_trace['line']['color'] = line_color

        line_trace['text'] = hoverinfo.loc[idx]

        line_traces.append(line_trace)

    if len(no_go_lines) > 0:
        no_go_lines_to_plot = net.line.loc[no_go_lines]
        for idx, line in no_go_lines_to_plot.iterrows():
            line_color = color
            line_trace = dict(type='scatter',
                              text=[], hoverinfo='text', mode='lines', name='disconnected lines',
                              line=Line(width=width / 2, color='grey', dash='dot'))

            line_trace['x'], line_trace['y'] = _get_line_geodata_plotly(net, no_go_lines_to_plot.loc[idx:idx])

            line_trace['line']['color'] = line_color
            try:
                line_trace['text'] = hoverinfo.loc[idx]
            except (KeyError, IndexError, AttributeError):
                line_trace["text"] = line['name']

            line_traces.append(line_trace)

            if legendgroup:
                line_trace['legendgroup'] = legendgroup

    return line_traces


def _get_line_geodata_plotly(net, lines):
    xs = []
    ys = []

    from_bus = net.bus_geodata.loc[lines.from_bus, 'x'].tolist()
    to_bus = net.bus_geodata.loc[lines.to_bus, 'x'].tolist()
    # center point added because of the hovertool
    center = (np.array(from_bus) + np.array(to_bus)) / 2
    none_list = [None] * len(from_bus)
    xs = np.array([from_bus, center, to_bus, none_list]).T.flatten().tolist()

    from_bus = net.bus_geodata.loc[lines.from_bus, 'y'].tolist()
    to_bus = net.bus_geodata.loc[lines.to_bus, 'y'].tolist()
    # center point added because of the hovertool
    center = (np.array(from_bus) + np.array(to_bus)) / 2
    none_list = [None] * len(from_bus)
    ys = np.array([from_bus, center, to_bus, none_list]).T.flatten().tolist()

    # [:-1] is because the trace will not appear on maps if None is at the end
    return xs[:-1], ys[:-1]


def draw_traces(traces, on_map=False, map_style='basic', showlegend=True,
                filename="topology.html"):
    """
    plots all the traces (which can be created using :func:`create_bus_trace`, :func:`create_line_trace`,
    :func:`create_trafo_trace`)
    to PLOTLY (see https://plot.ly/python/)

    INPUT:
        **traces** - list of dicts which correspond to plotly traces
        generated using: `create_bus_trace`, `create_line_trace`, `create_trafo_trace`

    OPTIONAL:
        **on_map** (bool, False) - enables using mapbox plot in plotly

        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **showlegend** (bool, 'True') - enables legend display

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the
            network geodata any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **filename** (str, "temp-plot.html") - plots to a html file called filename

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object

    """

    if on_map:
        # change traces for mapbox
        # change trace_type to scattermapbox and rename x to lat and y to lon
        for trace in traces:
            trace['lat'] = trace.pop('x')
            trace['lon'] = trace.pop('y')
            trace['type'] = 'scattermapbox'
            if "line" in trace and isinstance(trace["line"], Line):
                # scattermapboxplot lines do not support dash for some reason, make it a red line instead
                if "dash" in trace["line"]._props:
                    _prps = dict(trace["line"]._props)
                    _prps.pop("dash", None)
                    _prps["color"] = "red"
                    trace["line"] = scmLine(_prps)
                else:
                    trace["line"] = scmLine(dict(trace["line"]._props))
            elif "marker" in trace and isinstance(trace["marker"], Marker):
                trace["marker"] = scmMarker(trace["marker"]._props)

    # setting Figure object
    fig = Figure(data=traces,  # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=showlegend,
                     autosize=True,
                     hovermode='closest',
                     margin=dict(b=5, l=5, r=5, t=5),
                     annotations=[dict(
                         text="",
                         showarrow=False,
                         xref="paper", yref="paper",
                         x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False),
                     # legend=dict(x=0, y=1.0)
                 ), )
    if on_map:
        mapbox_access_token = _get_mapbox_token()
        center_index = next((index for (index, d) in enumerate(traces) if d["name"] == "buses"), None)
        fig['layout']['mapbox'] = dict(accesstoken=mapbox_access_token,
                                       bearing=0,
                                       center=dict(lat=pd.Series(traces[center_index]['lat']).dropna().mean(),
                                                   lon=pd.Series(traces[center_index]['lon']).dropna().mean()),
                                       style=map_style,
                                       pitch=0,
                                       zoom=11)

    plot(fig, filename=filename)

    return fig


def plot_grid(net, on_map=True, tree=False):
    net = switch_coords(net)
    net_to_plot = copy.deepcopy(net)
    if tree:
        on_map = False
        net_to_plot = pp.plotting.create_generic_coordinates(net_to_plot, overwrite=True)

    display_info = get_hoverinfo(net_to_plot, element="ext_grid")
    ext_grid_trace = create_bus_trace(net_to_plot, buses=net_to_plot.ext_grid.bus,
                                      color="yellow", size=20,
                                      patch_type='diamond', trace_name='external_grid', hoverinfo=display_info)

    asym_load_trace = create_bus_trace(net_to_plot, buses=net_to_plot.asymmetric_load.bus,
                                      color="orange", size=20,
                                      patch_type='circle', trace_name='asym_load')

    load_trace = create_bus_trace(net_to_plot, buses=net_to_plot.load.bus,
                                      color="purple", size=15,
                                      patch_type='circle', trace_name='load')

    sgen_trace = create_bus_trace(net_to_plot, buses=net_to_plot.sgen.bus,
                                      color="green", size=15,
                                      patch_type='circle', trace_name='sgen')

    aysm_sgen_trace = create_bus_trace(net_to_plot, buses=net_to_plot.asymmetric_sgen.bus,
                                      color="#E82C30", size=20,
                                      patch_type='circle', trace_name='asym_sgen')

    display_info = get_hoverinfo(net_to_plot, element="switch")
    switch_plot = create_switch_trace(net_to_plot, hoverinfo=display_info)

    display_info = get_hoverinfo(net_to_plot, element="trafo")
    trafo_plot = create_trafo_trace(net_to_plot, hoverinfo=display_info)

    display_info = get_hoverinfo(net_to_plot, element="bus")
    bus_plot = create_bus_trace(net_to_plot, hoverinfo=display_info)

    display_info = get_hoverinfo(net_to_plot, element="line")
    line_plot = create_line_trace(net_to_plot, hoverinfo=display_info)

    # adding final ext_grid marker
    marker_type = 'circle' if on_map else 'square'  # workaround because doesn't appear on mapbox if square
    hoverinfo = get_hoverinfo(net_to_plot, element="ext_grid")
    ext_grid_trace = create_bus_trace(net_to_plot, buses=net_to_plot.ext_grid.bus,
                                      color='yellow', size=20,
                                      patch_type=marker_type, trace_name='external_grid', hoverinfo=hoverinfo)


    # draw all the traces previously established
    draw_traces(asym_load_trace + load_trace + sgen_trace + ext_grid_trace + aysm_sgen_trace
                + line_plot + bus_plot + trafo_plot + switch_plot, on_map=on_map, map_style='streets')
    return

# create a code to export a plotly express figure


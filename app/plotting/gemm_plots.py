from xml.parsers.expat import model
import plotly.graph_objects as go
from pathlib import Path
from typing import List
import numpy as np

from app.data_handling.readers import *


def get_x_axis_labels(log_file_data: pd.DataFrame, simple=False) -> List[str]:
    """
    Generate x-axis labels for the GEMM operations based on the log file data.
    """
    x = "m:" + log_file_data['m'].astype(str) + "-n:" + log_file_data['n'].astype(str) + "-k:" + log_file_data['k'].astype(str) + "-tA:" + log_file_data["transa"].astype(str) + "-tB:" + log_file_data["transb"].astype(str)
    
    if "stat" in log_file_data.columns:
        x += "-stat:" + log_file_data["stat"].astype(str)

    if simple:
        x = (log_file_data["m"].astype(str) + "x" + log_file_data["n"].astype(str) + "x" + log_file_data["k"].astype(str)).to_list()

    
    return x


def plot_gemm_times(log_file_data: pd.DataFrame, title="GEMM Cycles") -> go.Figure:
    fig = go.Figure()
    x = get_x_axis_labels(log_file_data)
    fig.add_trace(go.Bar(
        x=list(range(len(log_file_data))), 
        y=log_file_data["dev_time"], 
        name='dev_time',
        offsetgroup="acc",
        text=log_file_data["dev_time"],
        texttemplate="%{text:.2f} s",
    ))
    fig.add_trace(go.Bar(
        x=list(range(len(log_file_data))), 
        y=log_file_data["host_time"], 
        name='host_time',
        offsetgroup="noacc",
        text=log_file_data["host_time"],
        texttemplate="%{text:.2f} s",
    ))
    fig.update_layout(xaxis=dict(ticktext=x, 
                                 tickvals=list(range(len(log_file_data))),))
    fig.update_yaxes(type="linear", title_text="Time (s)")

    fig.update_layout(title=title, xaxis_title="GEMM Operation", yaxis_title="Time (s)")
    return fig  


def plot_gemm_speedup(log_file_data: pd.DataFrame, title="GEMM Speedup") -> go.Figure:
    fig = go.Figure()
    x = get_x_axis_labels(log_file_data)

    fig.add_trace(go.Bar(
        x=list(range(len(log_file_data))),
        y=log_file_data["host_time"] / log_file_data["dev_time"], 
        name='Speedup',
        offsetgroup="speedup",
        text=log_file_data["host_time"] / log_file_data["dev_time"],
        texttemplate="%{text:.2f}x",
    ))
    fig.update_layout(xaxis=dict(ticktext=x, tickvals=list(range(len(log_file_data)))))
    fig.update_layout(title=title, xaxis_title="GEMM Operation", yaxis_title="Speedup")
    return fig


def plot_gemm_times__with_speedup(log_file_data: pd.DataFrame, title="Gemm times with speedup") -> go.Figure:
    fig = go.Figure()
    x = get_x_axis_labels(log_file_data, simple=True)

    speedup = log_file_data["host_time"] / log_file_data["dev_time"]

    fig.add_trace(go.Bar(
        x=list(range(len(log_file_data))), 
        y=log_file_data["host_time"], 
        name='CVA6',
        offsetgroup="noacc",
    ))


    fig.add_trace(go.Bar(
        x=list(range(len(log_file_data))), 
        y=log_file_data["dev_time"], 
        name='CVA6+Snitch',
        offsetgroup="acc",
        text=speedup,
        texttemplate="%{text:.2f}x",
        textposition="outside",
    ))

    # Add a dummy trace for the speedup legend entry
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        name="Speedup",
        mode="markers",
        marker=dict(symbol="circle", color="black", size=5),
        showlegend=True,
        hoverinfo="skip",
    ))

    fig.update_layout(xaxis=dict(ticktext=x, 
                                 tickvals=list(range(len(log_file_data))),))
    fig.update_yaxes(type="linear", title_text="Time (s)", minorloglabels='none', exponentformat="power")
    fig.update_xaxes(title_text="Problem size (m x n x k)")
    return fig

def plot_gemm_times_with_speedup_triple(log_file_data: pd.DataFrame, log_file_data_iommu: pd.DataFrame, title="Gemm times with speedup") -> go.Figure:
    fig = go.Figure()
    x = get_x_axis_labels(log_file_data, simple=True)

    speedup = log_file_data["host_time"] / log_file_data["dev_time"]
    speedup_iommu = log_file_data["host_time"] / log_file_data_iommu["dev_time"]
    fig.add_trace(go.Bar(
        x=list(range(len(log_file_data))), 
        y=log_file_data["host_time"], 
        name='CVA6',
        offsetgroup="noacc",
    ))


    fig.add_trace(go.Bar(
        x=list(range(len(log_file_data))), 
        y=log_file_data["dev_time"], 
        name='CVA6+Snitch',
        offsetgroup="acc",
        text=speedup,
        texttemplate="%{text:.2f}x",
        textposition="outside",
    ))


    fig.add_trace(go.Bar(
        x=list(range(len(log_file_data_iommu))), 
        y=log_file_data_iommu["dev_time"], 
        name='CVA6+Snitch+IOMMU',
        offsetgroup="iommu",
        text=speedup_iommu,
        texttemplate="%{text:.2f}x",
        textposition="outside",
    ))


    # Add a dummy trace for the speedup legend entry
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        name="Speedup",
        mode="markers",
        marker=dict(symbol="circle", color="black", size=5),
        showlegend=True,
        hoverinfo="skip",
    ))

    fig.update_layout(xaxis=dict(ticktext=x, 
                                 tickvals=list(range(len(log_file_data))),))
    fig.update_yaxes(type="linear", title_text="Time (s)", minorloglabels='none', exponentformat="power")
    fig.update_xaxes(title_text="Problem size (m x n x k)")
    return fig



def plot_gemm_times_with_speedup_and_timestamps(log_file_data: pd.DataFrame, title="Gemm times with speedup") -> go.Figure:
    fig = go.Figure()
    x = get_x_axis_labels(log_file_data, simple=True)

    speedup = log_file_data["host_time"] / log_file_data["dev_time"]

    fig.add_trace(go.Bar(
        x=x, 
        y=log_file_data["host_time"], 
        name='CVA6',
        offsetgroup="noacc",
    ))

    with_iommu = True if "data_in_copy_map" in log_file_data.columns else False
    remainder = log_file_data["dev_time"]
    if not with_iommu:
        fig.add_trace(go.Bar(
            x=x,
            y=log_file_data["data_in_copy"]+log_file_data["data_out_copy"],
            name='data copy',
            offsetgroup="acc",
        ))

        remainder -= (log_file_data["data_out_copy"]+log_file_data["data_in_copy"])

    else:
        fig.add_trace(go.Bar(
            x=x,
            y=log_file_data["data_in_copy_map"],
            name='IOMMU mapping',
            offsetgroup="acc",
        ))
        remainder -= log_file_data["data_in_copy_map"]
    fig.add_trace(go.Bar(
        x=x,
        y=log_file_data["enter_omp"]+log_file_data["offload_return"],
        name='offloading',
        offsetgroup="acc",
    ))
    remainder -= (log_file_data["enter_omp"]+log_file_data["offload_return"])

    fig.add_trace(go.Bar(
        x=x,
        y=log_file_data["offload_wait"],
        name='offload wait',
        offsetgroup="acc",
    ))
    remainder -= log_file_data["offload_wait"]


    fig.add_trace(go.Bar(
        x=x, 
        y=remainder,
        name='other',
        offsetgroup="acc",
        text=speedup,
        texttemplate="%{text:.2f}x",
        textposition="outside",
    ))


    # Add a dummy trace for the speedup legend entry
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        name="Speedup",
        mode="markers",
        marker=dict(symbol="circle", color="black", size=5),
        showlegend=True,
        hoverinfo="skip",
    ))

    fig.update_layout(barmode='stack')
    fig.update_layout(xaxis=dict(ticktext=x, 
                                 tickvals=list(range(len(log_file_data))),))
    fig.update_yaxes(type="linear", title_text="Time (s)", minorloglabels='none', exponentformat="power")
    fig.update_xaxes(title_text="Problem size (m x n x k)")
    return fig



def plot_gemm_times_with_speedup_and_timestamps_triple(log_file_data: pd.DataFrame, log_file_data_iommu: pd.DataFrame, title="Gemm times with speedup") -> go.Figure:
    fig = go.Figure()
    x = get_x_axis_labels(log_file_data, simple=True)

    speedup = log_file_data["host_time"] / log_file_data["dev_time"]
    speedup_iommu = log_file_data["host_time"] / log_file_data_iommu["dev_time"]

    fig.add_trace(go.Bar(
        x=x, 
        y=log_file_data["host_time"], 
        name='CVA6',
        offsetgroup="noacc",
    ))

    with_iommu = True if "data_in_copy_map" in log_file_data.columns else False
    remainder = log_file_data["dev_time"]
    if not with_iommu:
        fig.add_trace(go.Bar(
            x=x,
            y=log_file_data["data_in_copy"]+log_file_data["data_out_copy"],
            name='data copy',
            offsetgroup="acc",
        ))

        remainder -= (log_file_data["data_out_copy"]+log_file_data["data_in_copy"])

    else:
        fig.add_trace(go.Bar(
            x=x,
            y=log_file_data["data_in_copy_map"],
            name='IOMMU mapping',
            offsetgroup="acc",
        ))
        remainder -= log_file_data["data_in_copy_map"]
    fig.add_trace(go.Bar(
        x=x,
        y=log_file_data["enter_omp"]+log_file_data["offload_return"],
        name='offloading',
        offsetgroup="acc",
    ))
    remainder -= (log_file_data["enter_omp"]+log_file_data["offload_return"])

    fig.add_trace(go.Bar(
        x=x,
        y=log_file_data["offload_wait"],
        name='offload wait',
        offsetgroup="acc",
    ))
    remainder -= log_file_data["offload_wait"]


    fig.add_trace(go.Bar(
        x=x, 
        y=remainder,
        name='other',
        offsetgroup="acc",
        text=speedup,
        texttemplate="%{text:.2f}x",
        textposition="outside",
    ))





    # with iommu

    from plotly.io import templates
    default_colors = templates[templates.default].layout.colorway

    remainder = log_file_data_iommu["dev_time"]
    fig.add_trace(go.Bar(
        x=x,
        y=log_file_data_iommu["data_in_copy_map"],
        name='IOMMU mapping',
        offsetgroup="iommu",
        marker_color=default_colors[7]
    ))
    remainder -= log_file_data_iommu["data_in_copy_map"]
    fig.add_trace(go.Bar(
        x=x,
        y=log_file_data_iommu["enter_omp"]+log_file_data_iommu["offload_return"],
        name='offloading',
        offsetgroup="iommu",
        showlegend=False,
        marker_color=default_colors[2]
    ))
    remainder -= (log_file_data_iommu["enter_omp"]+log_file_data_iommu["offload_return"])

    fig.add_trace(go.Bar(
        x=x,
        y=log_file_data_iommu["offload_wait"],
        name='offload wait',
        offsetgroup="iommu",
        showlegend=False,
        marker_color=default_colors[3]
    ))
    remainder -= log_file_data_iommu["offload_wait"]


    fig.add_trace(go.Bar(
        x=x, 
        y=remainder,
        name='other',
        offsetgroup="iommu",
        text=speedup_iommu,
        texttemplate="%{text:.2f}x",
        textposition="outside",
        showlegend=False,
        marker_color=default_colors[4],
    ))

    # Add a dummy trace for the speedup legend entry
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        name="Speedup",
        mode="markers",
        marker=dict(symbol="circle", color="black", size=5),
        showlegend=True,
        hoverinfo="skip",
    ))



    fig.update_layout(barmode='stack')
    fig.update_layout(xaxis=dict(ticktext=x, 
                                 tickvals=list(range(len(log_file_data))),))
    fig.update_yaxes(type="linear", title_text="Time (s)", minorloglabels='none', exponentformat="power")
    fig.update_xaxes(title_text="Problem size (m x n x k)")
    return fig


def plot_gemm_timestamps(hero_timestamps: pd.DataFrame, title="GEMM Timestamps") -> go.Figure:
    # Get the operation name of the data_in_copy operaton. IN the case of using IOMMU,
    # this is called "data_in_copy_map" instead of "data_in_copy".
    operations = hero_timestamps["operation"].to_list()
    data_in_op = ""
    for op in operations:
        if op.startswith("data_in_copy"):
            data_in_op = op
            break

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=hero_timestamps[hero_timestamps["operation"] == data_in_op]["time"], 
        name=data_in_op,
    ))

    fig.add_trace(go.Bar(
        y=hero_timestamps[hero_timestamps["operation"] == "enter_omp"]["time"], 
        name='Enter OMP',
    ))  
    fig.add_trace(go.Bar(
        y=hero_timestamps[hero_timestamps["operation"] == "offload_wait"]["time"], 
        name='Offload Wait',
    ))
    fig.add_trace(go.Bar(
        y=hero_timestamps[hero_timestamps["operation"] == "offload_return"]["time"], 
        name='Offload Return',
    ))
    fig.add_trace(go.Bar(
        y=hero_timestamps[hero_timestamps["operation"] == "data_out_copy"]["time"], 
        name='data_out_copy',
    ))


    fig.update_layout(title=title, xaxis_title="Index", yaxis_title="Timestamp (s)", barmode="stack")
    return fig


def plot_gemm_timestamps_v2(gemm_combined: pd.DataFrame, title="GEMM Timestamps") -> go.Figure:
    x = get_x_axis_labels(gemm_combined, simple=True)
    fig = go.Figure()
    with_iommu = True if "data_in_copy_map" in gemm_combined.columns else False

    if not with_iommu:
        fig.add_trace(go.Bar(
            x=x,
            y=gemm_combined["data_in_copy"],
            name='data_in_copy',
        ))
    else:
        fig.add_trace(go.Bar(
            x=x,
            y=gemm_combined["data_in_copy_map"],
            name='Page mapping',
        ))
    fig.add_trace(go.Bar(
        x=x,
        y=gemm_combined["enter_omp"]+gemm_combined["offload_return"],
        name='Offloading',
    ))

    fig.add_trace(go.Bar(
        x=x,
        y=gemm_combined["offload_wait"],
        name='Compute',
    ))

    if not with_iommu:
        fig.add_trace(go.Bar(
            x=x,
            y=gemm_combined["data_out_copy"],
            name='data_out_copy',
        ))
    fig.update_layout(title=title, xaxis_title="Index", yaxis_title="Timestamp (s)", barmode="stack")
    return fig


def plot_gemm_ipc(log_file_data: pd.DataFrame, title="GEMM IPC") -> go.Figure:

    fig = go.Figure()
    x = get_x_axis_labels(log_file_data)

    instruction_count = log_file_data["m"] * log_file_data["n"] * log_file_data["k"]

    # Add total cycles as bars
    fig.add_trace(go.Scatter(
        x=list(range(len(log_file_data))),
        y=instruction_count/log_file_data["tot"], 
        name='Tot',
        mode="markers"
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(log_file_data))),
        y=instruction_count/(log_file_data["host_time"]*30e6),
        name="host_time",
        mode="markers"
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(log_file_data))),
        y=instruction_count/(log_file_data["dev_time"]*30e6),
        name="dev_time",
        mode="markers"
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(log_file_data))),
        y=instruction_count/(log_file_data["compute"]),
        name="compute",
        mode="markers"
    ))
    fig.update_layout(xaxis=dict(ticktext=x, tickvals=list(range(len(log_file_data)))))
    fig.update_layout(xaxis_type="category", title=title)
    fig.update_yaxes(title_text="fmadd/Cycle")
    fig.update_traces(mode='markers', marker_size=15)
    return fig


def plot_gemm_device_cycles(data:pd.DataFrame, title="GEMM Device Cycles")-> go.Figure:
    x = get_x_axis_labels(data)

    fig = go.Figure()
    # Add total cycles as bars
    # fig.add_trace(go.Bar(
    #     x=data["x"],
    #     y=data["host_time"]*30e6,
    #     name="host_time",
    #     offsetgroup="host_time"
    # ))
    # fig.add_trace(go.Bar(
    #     x=data["x"],
    #     y=data["dev_time"]*30e6,
    #     name="dev_time",
    #     offsetgroup="dev_time"
    # ))

    fig.add_trace(go.Bar(
        x=list(range(len(x))),
        y=data["tot"],
        name='Tot',
        offsetgroup="tot"
    ))
    fig.add_trace(go.Bar(
        x=list(range(len(x))), 
        y=data["dma"], 
        name='dma',
        offsetgroup="dev"
    ))
    fig.add_trace(go.Bar(
        x=list(range(len(x))),
        y=data["issue"],
        name='issue',
        offsetgroup="dev"
    ))
    fig.add_trace(go.Bar(
        x=list(range(len(x))),
        y=data["compute"], 
        name='compute',
        offsetgroup="dev"
    ))
    fig.update_layout(xaxis=dict(ticktext=x, tickvals=list(range(len(x)))))
    fig.update_layout(barmode='stack', xaxis_type="category", title=title)
    fig.update_yaxes(title_text="Cycles")

    return fig


def gemm_get_arithmetic_intensity(m,n,k, float_size, tile_m=32, tile_n=48, tile_k=64):

    # Calculate the number of tiles in each dimension
    num_tiles_m = (m + tile_m - 1) // tile_m
    num_tiles_n = (n + tile_n - 1) // tile_n
    num_tiles_k = (k + tile_k - 1) // tile_k

    # Calculate the total number of operations
    total_operations = m * n * k

    # Calculate the total number of memory accesses
    total_memory_accesses = (m*n*2 + m*k*num_tiles_n + n*k*num_tiles_m) * float_size


    # Calculate the arithmetic intensity
    arithmetic_intensity = total_operations / total_memory_accesses

    return arithmetic_intensity

def plot_gemm_roofline(log_file_data: pd.DataFrame, float_size=8, title="GEMM Roofline on device") -> go.Figure:
    fig = go.Figure()
    hovertext = None
    try:
        hovertext = "m:" + log_file_data['m'].astype(str) + "-n:" + log_file_data['n'].astype(str) + "-k:" + log_file_data['k'].astype(str) + "-tA:" + log_file_data["transa"].astype(str) + "-tB:" + log_file_data["transb"].astype(str)
    except KeyError:
        try:
            hovertext = "m:" + log_file_data['m'].astype(str) + "-n:" + log_file_data['n'].astype(str) + "-k:" + log_file_data['k'].astype(str)
        except KeyError:
            hovertext = log_file_data.index.astype(str)

    # theoretical_initial_load_time = np.minimum(np.ceil(log_file_data["m"] / 32)*(32*64+48*64+32*48)*float_size/8, (log_file_data["dma"]+log_file_data["issue"]))  # in cycles
    # print(f"Theoretical initial load time: {theoretical_initial_load_time} cycles")
    fig.add_trace(go.Scatter(
        x=gemm_get_arithmetic_intensity(log_file_data["m"], log_file_data["n"], log_file_data["k"], float_size=float_size),  # Convert to MFlops
        y=(log_file_data["m"] * log_file_data["n"] * log_file_data["k"]) / (log_file_data["tot"]),
        mode="markers",
        name="On Device Performance",
        hovertext=hovertext,
        marker=dict(size=15, symbol='triangle-up')
    ))

    # fig.add_trace(go.Scatter(
    #     x=gemm_get_arithmetic_intensity(log_file_data["m"], log_file_data["n"], log_file_data["k"], float_size=float_size),  # Convert to MFlops
    #     y=(log_file_data["m"] * log_file_data["n"] * log_file_data["k"]) / log_file_data["compute"],
    #     mode="markers",
    #     name="compute",
    #     marker=dict(size=10)
    # ))


    peak_performance = 8*(8/float_size) # fmadd/cycle
    bandwidth = 8 # B/cycle

    ai_limit = gemm_get_arithmetic_intensity(10000,10000,10000,float_size=float_size) + 0.5  # Add a small margin to the limit

    x = np.linspace(0, ai_limit, 1000)
    y = np.minimum(x * bandwidth, peak_performance)
    y2 = np.minimum(x * bandwidth/2, peak_performance)

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"Roofline {bandwidth} B / cycle",
        # line=dict(color='red')
    ))

    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y2,
    #     mode="lines",
    #     name=f"Roofline {bandwidth/2} B / cycle",
    #     line=dict(dash='dash')
    # ))

    fig.update_layout(
        title=title,
        xaxis_title="Arithmetic Intensity (fmadd / DMA Byte transferred)",
        yaxis_title="Performance (fmadd / cycle)",
        xaxis=dict(type="linear", range=[0, ai_limit]),
        yaxis=dict(type="linear", range=[0, peak_performance+1]),

        showlegend=True
    )


    return fig

def plot_gemm_percentage(log_file_data: pd.DataFrame, title="GEMM Percentage") -> go.Figure:
    fig = go.Figure()
    x = get_x_axis_labels(log_file_data, simple=True)

    dma_percentage = log_file_data["dma"] / log_file_data["tot"] * 100
    issue_percentage = log_file_data["issue"] / log_file_data["tot"] * 100
    compute_percentage = log_file_data["compute"] / log_file_data["tot"] * 100

    fig.add_trace(go.Scatter(
        x=list(range(len(log_file_data))),
        y=dma_percentage,
        name='DMA wait',
        mode='lines+markers',
        # line=dict(color=default_template.layout.colorway[0]),
        text=dma_percentage,
        texttemplate="%{text:.2f}%",
        stackgroup="percentage",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(log_file_data))),
        y=issue_percentage,
        name='DMA Issue',
        mode='lines+markers',
        # line=dict(color='green'),
        text=issue_percentage,
        texttemplate="%{text:.2f}%",
        stackgroup="percentage",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(log_file_data))),
        y=compute_percentage,
        name='Compute',
        mode='lines+markers',
        # line=dict(color='red'),
        text=compute_percentage,
        texttemplate="%{text:.2f}%",
        stackgroup="percentage",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(log_file_data))),
        y=np.ones_like(dma_percentage) * 100,
        mode='lines',
        name='100% Line',
        line=dict(color='black', dash='dash'),
    ))
    fig.update_layout(xaxis=dict(ticktext=x, tickvals=list(range(len(log_file_data)))))
    fig.update_layout(title=title, xaxis_title="GEMM Operation", yaxis_title="Percentage (%)")
    return fig
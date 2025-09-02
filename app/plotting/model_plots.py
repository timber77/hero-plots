import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

from app.data_handling.readers import *
from app.plotting.custom_sankey import Model_Sankey

def plot_time_total(hw_platforms: List[str], model: str) -> go.Figure:
    """
    Plot the total time taken by each hardware platform for a given model.
    """
    fig = go.Figure()
    for hw in hw_platforms:
        log_file = Path(f"./measurements/{hw}/{model}/benchmark.txt")
        df = get_summary_by_node_type_benchmark_tool(log_file)
        total_time = df["avg_ms"].sum() / 1000  # Convert to seconds
        fig.add_trace(go.Bar(
            x=[hw],
            y=[total_time],
            name=hw,
            width=0.4
        ))  
    fig.update_layout(
        title=f"Total Time for {model}",
        xaxis_title="Hardware Platform",
        yaxis_title="Time (s)",
        width=800,
        height=600
    )
    return fig

def plot_time_total_v2(hw_platforms: List[str], models: List[str]) -> go.Figure:
    fig = go.Figure()

    for hw in hw_platforms:
        x = []
        y = []
        for model in models:
            log_file = Path(f"./measurements/{hw}/{model}/benchmark.txt")
            df = get_summary_by_node_type_benchmark_tool(log_file)
            total_time = df["avg_ms"].sum() / 1000  # Convert to seconds
            x.append(model)
            y.append(total_time)
            
        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name=hw,
            # width=0.2,  # Adjust width based on number of hardware platforms
            offsetgroup=hw,
            text=y,
            texttemplate="%{text:.2f} s",
            ))  
    fig.update_layout(
        title=f"Total Time",
        xaxis_title="Model",
        yaxis_title="Time (s)",
        width=800,
        height=600
    )
    return fig


def plot_time_total_v2_with_speedup(hw_platforms: List[str], models: List[str]) -> go.Figure:
    fig = go.Figure()
    reference_y = []
    for i, hw in enumerate(hw_platforms):
        x = []
        y = []
        speedup = []
        for model in models:
            log_file = Path(f"./measurements/{hw}/{model}/benchmark.txt")
            df = get_summary_by_node_type_benchmark_tool(log_file)
            total_time = df["avg_ms"].sum() / 1000  # Convert to seconds
            x.append(model)
            y.append(total_time)
        if i == 0:  # Reference hardware
            reference_y = y

        speedup = [ref / new for ref, new in zip(reference_y, y)]
        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name=hw,
            # width=0.2,  # Adjust width based on number of hardware platforms
            offsetgroup=hw,
            text=[f"{speedup:.2f}x" if speedup != 1.0 else "baseline" for speedup in speedup],
            textposition="outside",
            ))  
    fig.update_layout(
        title=f"Inference Time and Speedup",
        xaxis_title="Model",
        yaxis_title="Time (s)",
    )
    return fig


def plot_time_total_per_node_type(hw_platforms: List[str], model: str) -> go.Figure:
    """
    Plot the total time taken by each hardware platform for a given model, broken down by node type.
    """
    fig = go.Figure()
    dfs = []
    x = []
    for hw in hw_platforms:
        log_file = Path(f"./measurements/{hw}/{model}/benchmark.txt")
        df = get_summary_by_node_type_benchmark_tool(log_file)
        dfs.append(df)
        x.append(hw)
    # We go reverse to go from least time to most time as the tool already sorts it descendingly
    for node_type in reversed(df["Node_type"].unique()):
        values = []
        for df in dfs:
            node_df = df[df["Node_type"] == node_type]
            if not node_df.empty:
                values.append(node_df["avg_ms"].values[0] / 1000)
        node_df = df[df["Node_type"] == node_type]
        fig.add_trace(go.Bar(
            x=x, 
            y=values, 
            name=node_type, 
            width=0.4,
        ))
    
    fig.update_layout(barmode='stack', title=dict(text=f"Total Time for {model} by Node Type"))
    fig.update_xaxes(title_text="Hardware Platform")
    fig.update_yaxes(title_text="Time (s)", type="linear")
    fig.update_layout(width=800, height=600)
    return fig



def plot_time_per_layer(hw_platforms: List[str], model: str, only_conv:bool=False) -> go.Figure:
    """
    Plot the time taken per layer for each hardware platform for a given model.
    If only_conv is True, only CONV layers are plotted.
    """


    fig = go.Figure()
    for hw in hw_platforms:
        log_file = Path(f"./measurements/{hw}/{model}/benchmark.txt")
        df = get_run_order_benchmark_tool(log_file)
        if only_conv:
            df = df[df["node_type"].str.contains("CONV")]
        fig.add_trace(go.Bar(
            y=df["avg_ms"] / 1000,  # Convert to seconds
            name=hw,
        ))  


    if only_conv:
        gemm_dims = pd.read_csv(Path(f"./metadata/{model}/gemm_dims.csv"))
        ticktext = [f"{row['m']}x{row['n']}x{row['k']}" for _, row in gemm_dims.iterrows()]
    else:
        ticktext = df["node_type"].tolist()

    fig.update_layout(
        title=f"Time per Layer for {model}",
        xaxis_title="Layer",
        yaxis_title="Time (s)",
        xaxis=dict(ticktext=ticktext, tickvals=list(range(len(df["node_type"])))),
    )
    return fig

def plot_time_detailed_per_layer(hw_platform: str, model: str, only_conv:bool=True) -> go.Figure:
    """
    Plot the detailed time breakdown per layer for a given hardware platform and model.
    If only_conv is True, only CONV layers are plotted (False not supported atm). 
    """
    benchmark_file = Path(f"./measurements/{hw_platform}/{model}/benchmark.txt")
    hero_timestamps_file = Path(f"./measurements/{hw_platform}/{model}/hero_timestamps.txt")
    device_cycles_file = Path(f"./measurements/{hw_platform}/{model}/device_cycles.txt")
    breakdown_file = Path(f"./measurements/{hw_platform}/{model}/conv_breakdown.txt")


    bench_df = get_run_order_benchmark_tool(benchmark_file)
    hero_timestamps_df = get_hero_timestamps(hero_timestamps_file)
    device_cycles_df = read_snitch_device_cycles(device_cycles_file)
    breakdown_df = read_conv_breakdown(breakdown_file)
    if only_conv:
        bench_df = bench_df[bench_df["node_type"].str.contains("CONV")]


    operations = hero_timestamps_df["operation"].unique().tolist()
    data_in_op = ""
    for op in operations:
        if op.startswith("data_in_copy"):
            data_in_op = op
            break

    # Contains all the detailed measurements
    details = [{"name": "compute", "values": device_cycles_df["compute"].values},
                {"name": "dma wait", "values": device_cycles_df["dma"].values},
                {"name": "dma issue", "values": device_cycles_df["issue"].values},
                {"name": "im2col", "values": breakdown_df["im2col"].values},
                {"name": "data in copy", "values": hero_timestamps_df[hero_timestamps_df["operation"] == data_in_op]["time"].values},
                {"name": "enter omp", "values": hero_timestamps_df[hero_timestamps_df["operation"] == "enter_omp"]["time"].values},
                {"name": "offload return", "values": hero_timestamps_df[hero_timestamps_df["operation"] == "offload_return"]["time"].values},
            ]
    
    if data_in_op == "data_in_copy":
        details.append({"name": "data out copy", "values": hero_timestamps_df[hero_timestamps_df["operation"] == "data_out_copy"]["time"].values})

    #Calculate the time that is not measured by one of the detailed measurements. Ie the difference between the total time and the sum of the detailed measurements
    
    total_times = bench_df["avg_ms"] / 1000  # Convert to seconds
    unknown_times = total_times
    for detail in details:
        unknown_times -= detail["values"]

    fig = go.Figure()
    for detail in details:
        fig.add_trace(go.Bar(
            y=detail["values"],  # Convert to seconds
            name=detail["name"],
            width=0.4
        ))
    fig.add_trace(go.Bar(
        y=unknown_times,
        name="unknown",
        width=0.4
    ))  
    fig.update_layout(
        title=f"Detailed Time Breakdown per Layer for {model} on {hw_platform}",
        xaxis_title="Layer",
        yaxis_title="Time (s)",
        width=800,
        height=600,
        barmode='stack',
        xaxis=dict(ticktext=bench_df["node_type"], tickvals=list(range(len(bench_df["node_type"]))))
    )
    fig.update_yaxes(tickformat=".2f")
    return fig




def plot_time_detailed_per_layer_percentage(hw_platform: str, model: str, only_conv:bool=True) -> go.Figure:
    """
    Plot the detailed time breakdown per layer for a given hardware platform and model.
    If only_conv is True, only CONV layers are plotted (False not supported atm). 
    """
    benchmark_file = Path(f"./measurements/{hw_platform}/{model}/benchmark.txt")
    hero_timestamps_file = Path(f"./measurements/{hw_platform}/{model}/hero_timestamps.txt")
    device_cycles_file = Path(f"./measurements/{hw_platform}/{model}/device_cycles.txt")
    breakdown_file = Path(f"./measurements/{hw_platform}/{model}/conv_breakdown.txt")


    bench_df = get_run_order_benchmark_tool(benchmark_file)
    hero_timestamps_df = get_hero_timestamps(hero_timestamps_file)
    device_cycles_df = read_snitch_device_cycles(device_cycles_file)
    breakdown_df = read_conv_breakdown(breakdown_file)
    if only_conv:
        bench_df = bench_df[bench_df["node_type"].str.contains("CONV")]


    operations = hero_timestamps_df["operation"].unique().tolist()
    data_in_op = ""
    for op in operations:
        if op.startswith("data_in_copy"):
            data_in_op = op
            break

    # Contains all the detailed measurements
    details = [{"name": "compute", "values": device_cycles_df["compute"].values},
                {"name": "dma wait", "values": device_cycles_df["dma"].values},
                {"name": "dma issue", "values": device_cycles_df["issue"].values},
                {"name": "im2col", "values": breakdown_df["im2col"].values},
                {"name": "data in copy", "values": hero_timestamps_df[hero_timestamps_df["operation"] == data_in_op]["time"].values},
                {"name": "enter omp", "values": hero_timestamps_df[hero_timestamps_df["operation"] == "enter_omp"]["time"].values},
                {"name": "offload return", "values": hero_timestamps_df[hero_timestamps_df["operation"] == "offload_return"]["time"].values},
            ]
    
    if data_in_op == "data_in_copy":
        details.append({"name": "data out copy", "values": hero_timestamps_df[hero_timestamps_df["operation"] == "data_out_copy"]["time"].values})

    #Calculate the time that is not measured by one of the detailed measurements. Ie the difference between the total time and the sum of the detailed measurements
    
    total_times = bench_df["avg_ms"] / 1000  # Convert to seconds
    unknown_times = total_times.copy()
    for detail in details:
        unknown_times -= detail["values"]

    fig = go.Figure()
    for detail in details:
        fig.add_trace(go.Scatter(
            y=detail["values"]/total_times,  # Convert to seconds
            name=detail["name"],
            mode='lines+markers',
            stackgroup="percentage",
        ))

    fig.add_trace(go.Scatter(
        y=unknown_times/total_times,
        name="unknown",
        mode='lines+markers',
        stackgroup="percentage",
    ))
    fig.update_layout(
        title=f"Detailed Time Breakdown per Layer for {model} on {hw_platform}",
        xaxis_title="Layer",
        yaxis_title="Time (s)",
        width=800,
        height=600,
        barmode='stack',
        xaxis=dict(ticktext=bench_df["node_type"], tickvals=list(range(len(bench_df["node_type"]))))
    )
    fig.update_yaxes(tickformat=".2f")
    return fig







def plot_speedup_total(reference_hw: str, accelerated_hw:str, models: List[str]) -> go.Figure:
    """
    Plot the speedup of an accelerated hardware platform compared to a reference hardware platform for a list of models.
    """
    fig = go.Figure()
    speedups = []
    for m in models:
        ref_log_file = Path(f"./measurements/{reference_hw}/{m}/benchmark.txt")
        acc_log_file = Path(f"./measurements/{accelerated_hw}/{m}/benchmark.txt")

        ref_df = get_summary_by_node_type_benchmark_tool(ref_log_file)
        acc_df = get_summary_by_node_type_benchmark_tool(acc_log_file)

        ref_total_time = ref_df["avg_ms"].sum() / 1000  # Convert to seconds
        acc_total_time = acc_df["avg_ms"].sum() / 1000  # Convert to seconds

        speedup = ref_total_time / acc_total_time
        speedups.append(speedup)

    fig.add_trace(go.Bar(
        x=models,
        y=speedups,
        text=[f"{s:.2f}x" for s in speedups],
        name="Speedup",
        width=0.4
    ))

    fig.update_layout(
        title=f"Speedup of {accelerated_hw} over {reference_hw}",
        xaxis_title="Model",
        yaxis_title="Speedup (x)",
        barmode='group',
        width=800,
        height=600
    )
    
    return fig

def plot_speedup_per_layer(reference_hw: str, accelerated_hw:str, model:str, only_conv:bool=False, title="") -> go.Figure:
    """
    Plot the speedup of an accelerated hardware platform compared to a reference hardware platform for each layer of a model.
    """
    fig = go.Figure()
    
    ref_log_file = Path(f"./measurements/{reference_hw}/{model}/benchmark.txt")
    acc_log_file = Path(f"./measurements/{accelerated_hw}/{model}/benchmark.txt")

    ref_df = get_run_order_benchmark_tool(ref_log_file)
    acc_df = get_run_order_benchmark_tool(acc_log_file)

    if only_conv:
        ref_df = ref_df[ref_df["node_type"].str.contains("CONV")]
        acc_df = acc_df[acc_df["node_type"].str.contains("CONV")]

    ref_times = ref_df["avg_ms"] / 1000  # Convert to seconds
    acc_times = acc_df["avg_ms"] / 1000  # Convert to seconds

    speedups = ref_times / acc_times

    fig.add_trace(go.Bar(
        y=speedups,
        # text=[f"{s:.2f}x" for s in speedups],
        name="Speedup",
        width=0.4
    ))

    if only_conv:
        gemm_dims = pd.read_csv(Path(f"./metadata/{model}/gemm_dims.csv"))
        ticktext = [f"{row['m']}x{row['n']}x{row['k']}" for _, row in gemm_dims.iterrows()]
    else:
        ticktext = ref_df["node_type"].tolist()


    fig.update_layout(
        title=title,
        xaxis_title="Layer" if not only_conv else "GEMM Dimensions (MxNxK)",
        yaxis_title="Speedup (x)",
        xaxis=dict(ticktext=ticktext, tickvals=list(range(len(acc_df["node_type"]))))
    )
    
    return fig
    

def plot_timing_breakdown_conv(hw_platforms: List[str], model: str) -> go.Figure:
    """
    Plot the timing breakdown for convolution operations for a given model on different hardware platforms.
    """
    fig = go.Figure()
    x = []
    dfs = []
    for hw in hw_platforms:
        log_file = Path(f"./measurements/{hw}/{model}/conv_breakdown.txt")
        df = read_conv_breakdown(log_file)
        dfs.append(df)
        x.append(hw)  

    for col in ["im2col", "cpu_backend_gemm"]:
        values = []
        for df in dfs:
            if not df.empty:
                values.append(df[col].sum())
            else:
                values.append(0)
        fig.add_trace(go.Bar(
            x=x, 
            y=values, 
            name=col, 
            width=0.4,
        ))  
    fig.update_layout(barmode='stack', title=dict(text="Convolution Timing Breakdown"))
    fig.update_xaxes(title_text="Hardware Platform")
    fig.update_yaxes(title_text="Time (s)", type="linear")
    fig.update_layout(width=800, height=600)
    return fig



def plot_timing_breakdown_sankey(hardware:str, model:str, stop = None, normalization = 1) -> go.Figure:
    """
    Plot a Sankey diagram for the timing breakdown.
    """

    benchmark_file = Path(f"./measurements/{hardware}/{model}/benchmark.txt")
    conv_breakdown_file = Path(f"./measurements/{hardware}/{model}/conv_breakdown.txt")
    hero_timestamps = Path(f"./measurements/{hardware}/{model}/hero_timestamps.txt")
    device_cycles = Path(f"./measurements/{hardware}/{model}/device_cycles.txt")

    benchmark_df = get_summary_by_node_type_benchmark_tool(benchmark_file)
    conv_breakdown_df = read_conv_breakdown(conv_breakdown_file)


    model_sankey:Model_Sankey = Model_Sankey(hardware, model, 5)

    # Per layer type
    for node_type in benchmark_df["Node_type"]:
        value = benchmark_df[benchmark_df["Node_type"] == node_type]["avg_ms"].values[0] / 1000
        model_sankey.add_new(node_type, value, source_name="Total", horizontal_layer=1)


    # Conv breakdown
    conv_val = conv_breakdown_df["cpu_backend_gemm"].sum()
    model_sankey.add_new("CPU Backend GEMM", conv_val, source_name="CONV_2D", horizontal_layer=2)

    im2col_val = conv_breakdown_df["im2col"].sum()
    model_sankey.add_new("Im2Col", im2col_val, source_name="CONV_2D", horizontal_layer=2)

    # hero_timestamps
    if hero_timestamps.exists():
        hero_timestamps_df = get_hero_timestamps(hero_timestamps)
        operations = hero_timestamps_df["operation"].unique().tolist()
        operations.remove("exit_omp")

        data_in_op = ""
        for op in operations:
            if op.startswith("data_in_copy"):
                data_in_op = op
                break

        # Reorder operations so that offload_wait is always first
        operations_ordered = []
        for op in operations:
            if op == "offload_wait":
                operations_ordered.insert(0, op)
            else:
                operations_ordered.append(op)
        operations = operations_ordered

        # Add hero timestamps
        for op in operations:
            value = hero_timestamps_df[hero_timestamps_df["operation"] == op]["time"].sum()
            model_sankey.add_new(op, value, source_name="CPU Backend GEMM", horizontal_layer=3)

        if device_cycles.exists():
            device_cycle_df = read_snitch_device_cycles(device_cycles)
            # Add device cycles
            model_sankey.add_new("compute", device_cycle_df["compute"].sum(), source_name="offload_wait", horizontal_layer=4)
            model_sankey.add_new("dma wait", device_cycle_df["dma"].sum(), source_name="offload_wait", horizontal_layer=4)
            model_sankey.add_new("dma issue", device_cycle_df["issue"].sum(), source_name="offload_wait", horizontal_layer=4)
            
    # style nodes 

    tot = model_sankey._get_node_value("Total")
    if normalization != 1:
        model_sankey.normalize_values(normalization)


    model_sankey.realign_nodes()

    match stop:
        case "node_types":
            model_sankey.hide_horizontal_layer(2)
            model_sankey.hide_horizontal_layer(3)
            model_sankey.hide_horizontal_layer(4)
        case "conv_breakdown":
            model_sankey.hide_horizontal_layer(3)
            model_sankey.hide_horizontal_layer(4)
        case "hero_timestamps":
            model_sankey.hide_horizontal_layer(4)
        case _:
            pass




    fig = model_sankey.create_figure()

    icicle = model_sankey.create_icicle()

    return fig, tot, icicle

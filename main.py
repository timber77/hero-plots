import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path


def read_log(log_file:Path)-> pd.DataFrame:
    """
    Read the log file and return a pandas dataframe.
    """
    data = pd.read_csv(log_file, sep=',', header=0)
    # Add a new column with the x axis values
    data["x"] = "m:" + data['m'].astype(str) + "-n:" + data['n'].astype(str) + "-k:" + data['k'].astype(str) + "-tA:" + data["transa"].astype(str) + "-tB:" + data["transb"].astype(str) 
    return data

def plot_cycles(log_file:Path)-> go.Figure:
    data = read_log(log_file)

    fig = go.Figure()
    # Add total cycles as bars
    fig.add_trace(go.Bar(
        x=data["x"], 
        y=data["tot"], 
        name='Tot',
        offsetgroup="tot"
    ))
    fig.add_trace(go.Bar(
        x=data["x"], 
        y=data["dma"], 
        name='dma',
        offsetgroup="dev"
    ))
    fig.add_trace(go.Bar(
        x=data["x"], 
        y=data["issue"], 
        name='issue',
        offsetgroup="dev"
    ))
    fig.add_trace(go.Bar(
        x=data["x"], 
        y=data["compute"], 
        name='compute',
        offsetgroup="dev"
    ))
    fig.add_trace(go.Bar(
        x=data["x"],
        y=data["host_time"]*2e9,
        name="host_time",
        offsetgroup="host_time"
    ))
    fig.add_trace(go.Bar(
        x=data["x"],
        y=data["dev_time"]*50e6,
        name="dev_time",
        offsetgroup="dev_time"
    ))
    fig.update_layout(barmode='stack', xaxis_type="category", title=dict(text=log_file.name))
    fig.update_yaxes(title_text="Cycles")
    return fig



def plot_ipc(log_file:Path)-> go.Figure:

    data = read_log(log_file)

    fig = go.Figure()

    instruction_count = data["m"] * data["n"] * data["k"]

    # Add total cycles as bars
    fig.add_trace(go.Scatter(
        x=data["x"], 
        y=instruction_count/data["tot"], 
        name='Tot',
        mode="markers"
    ))
    fig.add_trace(go.Scatter(
        x=data["x"],
        y=instruction_count/(data["host_time"]*2e9),
        name="host_time",
        mode="markers"
    ))
    fig.add_trace(go.Scatter(
        x=data["x"],
        y=instruction_count/(data["dev_time"]*50e6),
        name="dev_time",
        mode="markers"
    ))
    fig.add_trace(go.Scatter(
        x=data["x"],
        y=instruction_count/(data["compute"]),
        name="compute",
        mode="markers"
    ))

    fig.update_layout(xaxis_type="category", title=dict(text=log_file.name))
    fig.update_yaxes(title_text="IPC")
    fig.update_traces(mode='markers', marker_size=15)
    return fig



def compare_dev_ipcs(file1, file2)-> go.Figure:
    """ 
    Compare the IPC of two different runs of a different kernel version. Make sure the rest of the parameters are the same.
    """
    data1 = read_log(file1)
    data2 = read_log(file2)

    fig = go.Figure()

    instruction_count = data1["m"] * data1["n"] * data1["k"]

    # Add total cycles as bars
    fig.add_trace(go.Scatter(
        x=data1["x"], 
        y=instruction_count/data1["tot"], 
        name='SSR Repeat + 2D Loop all k',
        mode="markers"
    ))
    fig.add_trace(go.Scatter(
        x=data2["x"], 
        y=instruction_count/data2["tot"], 
        name='SSR Repeat + 2D Loop kmax 256',
        mode="markers"
    ))


    fig.update_layout(xaxis_type="category", title=dict(text="IPC comparison"))
    fig.update_yaxes(title_text="IPC")
    fig.update_traces(mode='markers', marker_size=15)

    return fig




if __name__ == "__main__":

    LOG_FOLDER = Path("/scratch/msc25f15/mnt/milkv-01/measurements/")

    fig = plot_cycles(LOG_FOLDER / "gemm.csv")
    fig.write_html("cycles.html")
    fig.show()

    fig = plot_ipc(LOG_FOLDER / "gemm.csv")
    fig.write_html("ipc.html")
    fig.show()

    fig = compare_dev_ipcs(LOG_FOLDER/"gemm.csv", LOG_FOLDER/"gemm_opti_repeat.csv")
    fig.show()

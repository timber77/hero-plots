import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

COLORSEQ = px.colors.qualitative.Vivid


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
        y=data["host_time"]*30e6,
        name="host_time",
        offsetgroup="host_time"
    ))
    fig.add_trace(go.Bar(
        x=data["x"],
        y=data["dev_time"]*30e6,
        name="dev_time",
        offsetgroup="dev_time"
    ))

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
    fig.update_layout(barmode='stack', xaxis_type="category", title=dict(text=log_file.name))
    fig.update_yaxes(title_text="Cycles")
    fig.update_layout(colorway=COLORSEQ)
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
        y=instruction_count/(data["host_time"]*30e6),
        name="host_time",
        mode="markers"
    ))
    fig.add_trace(go.Scatter(
        x=data["x"],
        y=instruction_count/(data["dev_time"]*30e6),
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
    fig.update_layout(colorway=COLORSEQ)
    fig.update_yaxes(title_text="fmadd/Cycle")
    fig.update_traces(mode='markers', marker_size=15)
    return fig



def compare_dev_ipcs(files:List)-> go.Figure:
    """ 
    Compare the fmadd/cycle of two different runs of a different kernel version. Make sure the rest of the parameters are the same.
    """

    fig = go.Figure()


    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist")
        data = read_log(file)
        instruction_count = data["m"] * data["n"] * data["k"]

        fig.add_trace(go.Scatter(
            x=data["x"], 
            y=instruction_count/data["tot"], 
            name=file.name,
            mode="markers"
        ))

    fig.update_layout(xaxis_type="category", title=dict(text="fmadd/cycle comparison"))
    fig.update_yaxes(title_text="fmadd/cycle")
    fig.update_traces(mode='markers', marker_size=15)
    fig.update_layout(colorway=COLORSEQ)

    return fig




if __name__ == "__main__":

    # LOG_FOLDER = Path("/scratch/msc25f15/mnt/milkv-01/measurements/")
    LOG_FOLDER = Path("./measurements")

    fig = plot_cycles(LOG_FOLDER / "gemm_cva6.csv")
    fig.write_html("cycles.html")
    fig.show()

    fig = plot_ipc(LOG_FOLDER / "gemm_cva6.csv")
    fig.write_html("fmadd_cycle.html")
    fig.show()
    files = [
        LOG_FOLDER / "gemm_cva6.csv"
    ]
    fig = compare_dev_ipcs(files)
    fig.write_html("fmadd_cycle_compare.html")
    fig.show()

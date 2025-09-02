
import re
from typing import List
import pandas as pd
from pathlib import Path


def _match_line_in_lines(lines: List[str], match_str: str) -> int:
    """
    Match a line in the lines list that contains the match_str.
    Returns all indices where the match_str is found.
    If no match is found, returns an empty list.
    """
    indices = []
    for i, line in enumerate(lines):
        if match_str in line:
            indices.append(i)
    return indices

def _get_next_empty_line(lines: List[str], start_index: int) -> int:
    """
    Get the next empty line in the lines list starting from start_index.
    Returns the index of the next empty line or -1 if no empty line is found.
    """
    for i in range(start_index, len(lines)):
        if lines[i].strip() == "":
            return i
    return -1

def _replace_whitespace_in_square_brackets(text: str, replacement = "_") -> str:
    """
    Replace whitespace in square brackets with underscores.
    This is useful for parsing log files where square brackets are used.
    """
    return re.sub(r'\[(.*?)\]', lambda m: '[' + m.group(1).replace(' ', replacement) + ']', text)



def read_conv_breakdown(log_file:Path)-> pd.DataFrame:
    """
    Read the conv breakdown log file and return a pandas dataframe.
    The dataframe will have three columns: im2col, cpu_backend_gemm, and total_conv.
    Each column will contain the time spent in the respective operation in seconds.
    """
    breakdowns = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        conv_breakdown = {"im2col":0, "cpu_backend_gemm":0, "total_conv":0}
        for line in lines:
            line_split = line.split(" ")
            if len(line_split) < 3:
                continue

            match line_split[1]:
                case "Im2col:":
                    conv_breakdown["im2col"] = float(line_split[2])/1000000
                case "cpu_backend_gemm::Gemm:":
                    conv_breakdown["cpu_backend_gemm"] = float(line_split[2])/1000000
                case "Total_Conv:":
                    conv_breakdown["total_conv"] = float(line_split[2])/1000000
                    breakdowns.append(conv_breakdown)
                    conv_breakdown = {"im2col":0, "cpu_backend_gemm":0, "total_conv":0}
                case _:
                    continue
    return pd.DataFrame(breakdowns, columns=["im2col", "cpu_backend_gemm", "total_conv"])


def get_summary_by_node_type_benchmark_tool(log_file:Path)-> pd.DataFrame:
    """
    Read the benchmark tool output log file and return a pandas dataframe.
    """
    start, end = 0, 0
    with open(log_file, 'r') as f:
        lines = f.readlines()
        summary_indices = _match_line_in_lines(lines, "Summary by node type")
        if len(summary_indices) == 0:
            raise ValueError("No summary by node type found in the log file.")
        summary_index = summary_indices[-1] # Take the last one which is about allocating tensors
        end_section_index = _get_next_empty_line(lines, summary_index)
        if end_section_index == -1:
            raise ValueError("No end of summary section found in the log file.")
        start = summary_index + 1
        end = end_section_index
    start += 1 # Skip the header line
    df = pd.read_csv(log_file, skiprows=start, nrows=end-start, sep=" ", header=None, skipinitialspace=True, engine='python')
    # Manually get and set the column names due to a space inside one of the column names
    with open(log_file, 'r') as f:
        header_line = _replace_whitespace_in_square_brackets(f.readlines()[start-1].strip()).replace("[", "").replace("]", "")
        header_columns = header_line.split()
    df.columns = header_columns
    return df

def get_run_order_benchmark_tool(log_file:Path)-> pd.DataFrame:
    """
    Read the benchmark tool output log file and return a pandas dataframe.
    """
    start, end = 0, 0
    with open(log_file, 'r') as f:
        lines = f.readlines()
        run_order_indices = _match_line_in_lines(lines, "Run Order")
        if len(run_order_indices) == 0:
            raise ValueError("No run order found in the log file.")
        run_order_index = run_order_indices[-1] # Take the last one, which is the actual run order
        end_section_index = _get_next_empty_line(lines, run_order_index)
        if end_section_index == -1:
            raise ValueError("No end of run order section found in the log file.")
        start = run_order_index+1
        end = end_section_index
    start+=1 # Skip the header line
    df = pd.read_csv(log_file, skiprows=start, nrows=end-start, sep=" ", header=None, skipinitialspace=True, engine='python')
    # Manually get and set the column names due to a space inside one of the column names
    with open(log_file, 'r') as f:
        header_line = _replace_whitespace_in_square_brackets(f.readlines()[start-1].strip()).replace("[", "").replace("]", "")
        header_columns = header_line.split()
    df.columns = header_columns
    return df

def get_hero_timestamps(log_file:Path)-> pd.DataFrame:
    """
    Read the hero timestamps log file and return a pandas dataframe.
    """
    data = {
        "operation": [],
        "function": [],
        "time": []
    }
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line.strip() == "":
                continue
            parts = line.split(" ")
            if len(parts) < 4:
                continue
            operation = parts[0].split("-")[0]
            function = parts[1]
            time = float(parts[3])
            data["operation"].append(operation)
            data["function"].append(function)
            data["time"].append(time)
    return pd.DataFrame(data)

def read_snitch_device_cycles(log_file:Path, dev_freq=30e6)-> pd.DataFrame:
    """
    Read the snitch device cycles log file and return a pandas dataframe.
    """
    data = pd.read_csv(log_file, sep=' ')
    # Convert to seconds
    data["tot"] = data["tot"] / dev_freq
    data["dma"] = data["dma"] / dev_freq
    data["issue"] = data["issue"] / dev_freq
    data["compute"] = data["compute"] / dev_freq
    # Add a new column with the x axis values
    return data


def read_gemm_log(log_file:Path)-> pd.DataFrame:
    """
    Read the gemm log file and return a pandas dataframe.
    The dataframe will have columns for each gemm operation.
    """
    """
    Read the log file and return a pandas dataframe.
    """
    data = pd.read_csv(log_file, sep=',', header=0)
    # Add a new column with the x axis values
    # data["x"] = "m:" + data['m'].astype(str) + "-n:" + data['n'].astype(str) + "-k:" + data['k'].astype(str) + "-tA:" + data["transa"].astype(str) + "-tB:" + data["transb"].astype(str) 
    return data
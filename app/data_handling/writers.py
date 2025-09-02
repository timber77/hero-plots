from pathlib import Path

def split_benchmark_tool_full_output(log_file: Path, outfolder: Path = "./measurements/"):
    """
    Split the full output from the benchmark tool into multiple text files, one for each "information source". 
    The output is split into:
    - conv breakdown timings: Giving a high level breakdown of the convolution operations split into the gemm backend and Im2Col
    - hero timestamps (if Snitch is used)
    - device cycles (if Snitch is used): The device cycles spent on compute and dma operations in the snitch cluster
    - benchmark tool output (everything after the device cycles)

    The output files are stored in a folder structure like:
    ./{outfolder}/{HW}/{model}/
    where HW is either "CVA6" or "CVA6+Snitch" depending on whether Snitch is used, and model is the name of the model being benchmarked.
    HW and model are extracted from the log file.

    Args:
        log_file (Path): Path to the log file containing full the benchmark tool output.

    """
    model = None

    with_snitch = False
    with_iommu = False
    if "IOMMU" in str(log_file) or "iommu" in str(log_file):
        with_iommu = True
    conv_timer_start = -1
    conv_timer_length = 0

    hero_timestamps_start = -1
    hero_timestamps_end = -1 # exclusive

    device_cycles_start = -1
    device_cycles_end = -1 # exclusive

    benchmark_tool_output_start = -1 # goes till eof

    with open (log_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("INFO: Graph: "):
                model = line.split(" ")[2].strip("[]")
                model = Path(model)
                model = model.stem
            elif line.startswith("[CONV_TIMER] ===== SUMMARY ====="):
                # This intentionally overwrites if it was set before, as we are not interested in the warmup run
                conv_timer_start = i
                conv_timer_length = 0
            elif line.startswith("[CONV_TIMER]"):
                conv_timer_length += 1
            elif line.startswith("info function time diff"):
                hero_timestamps_start = i
                with_snitch = True
            elif line.startswith("perf cycles device: tot dma issue compute"):
                # the perf cycles always come directly after the hero timestamps
                hero_timestamps_end = i
                device_cycles_start = i
            elif line.startswith("INFO: Inference timings in us:"):
                device_cycles_end = i
                benchmark_tool_output_start = i

        if model is None:
            raise ValueError("Model name not found in log file")

        HW = "CVA6+Snitch" if with_snitch else "CVA6"
        if with_iommu:
            HW += "+IOMMU"

        out_folder = Path(f"{outfolder}/{HW}/{model}/")
        out_folder.mkdir(parents=True, exist_ok=True)

        # Write conv breakdown timings
        with open(out_folder / f"conv_breakdown.txt", 'w') as conv_breakdown_file:
            for line in lines[conv_timer_start:conv_timer_start + conv_timer_length+1]:
                conv_breakdown_file.write(line)
        print(f"Conv breakdown timings written to {out_folder / f'conv_breakdown.txt'}")
        if with_snitch:
            with open(out_folder / f"hero_timestamps.txt", 'w') as hero_timestamps_file:
                for line in lines[hero_timestamps_start:hero_timestamps_end]:
                    hero_timestamps_file.write(line)

            print(f"Hero timestamps written to {out_folder / f'hero_timestamps.txt'}")

            with open(out_folder / f"device_cycles.txt", 'w') as device_cycles_file:
                for line in lines[device_cycles_start:device_cycles_end]:
                    if line.startswith("perf cycles device:"):
                        line = line.split(":")[1].strip(" ")  # Remove the prefix
                    device_cycles_file.write(line)
            print(f"Device cycles written to {out_folder / f'device_cycles.txt'}")
        with open(out_folder / f"benchmark.txt", 'w') as benchmark_file:
            for line in lines[benchmark_tool_output_start:]:
                benchmark_file.write(line.replace("\t", " "))  # Replace tabs with spaces for better processing
        print(f"Benchmark tool output written to {out_folder / f'benchmark.txt'}")

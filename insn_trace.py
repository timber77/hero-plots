import pandas as pd
import pathlib


ILA_FILE = pathlib.Path("/home/msc25f15/iladata.csv")

OBJDUMP_FILE = pathlib.Path("/scratch/msc25f15/mnt/milkv-01/hero-tools/apps/omp/OpenBLAS/install/libopenblas_riscv64_generic-r0.3.27.dev.so.dev.dis")



def get_addr_insn_map(objdump_file: pathlib.Path) -> dict:
    """
    Read the objdump file and return a dictionary with the address as key and the instruction as value.
    """
    addr_insn_map = {}
    with open(objdump_file, "r") as f:
        for line in f:
            if not line[0].isdigit():
                continue
            if line[8] != ":":
                continue
            parts = line.split(":")
            addr = parts[0]
            insn = parts[1]
            addr_insn_map[addr] = insn
    return addr_insn_map


ila_data = pd.read_csv(ILA_FILE, header=0, dtype=str, skiprows=[1])

core_0_pc = ila_data["design_1_i/carfield_xilinx_ip_0/inst/i_carfield_xilinx/i_carfield/gen_spatz_cluster.i_fp_cluster_wrapper/i_snitch_cluster_wrapper/i_cluster/gen_core[0].i_snitch_cc/i_snitch/pc_q[31:0]"]
adr_insn_map = get_addr_insn_map(OBJDUMP_FILE)

# mapped = core_0_pc.map(adr_insn_map)
# print(core_0_pc)
with open("mapped.txt", "w") as f:
    last_pc = "SOMETHING"
    duplicate_count = 0
    for i in range(len(core_0_pc)):
        if (core_0_pc[i] == last_pc):
            duplicate_count += 1
            continue
        if duplicate_count > 0:
            f.write(f"-----{duplicate_count}-----\n")
        duplicate_count = 0
        f.write(f"{ila_data['Sample in Buffer'][i]} : {core_0_pc[i]}: {adr_insn_map[core_0_pc[i]]}")
        last_pc = core_0_pc[i]

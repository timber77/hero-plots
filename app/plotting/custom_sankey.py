import warnings
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.io as pio
from typing import List, Tuple
from pathlib import Path

class Node:
    def __init__(self, name: str, x: float, y: float, color: str = "black"):
        self.name = name
        self.x = x
        self.y = y
        self.color = color

class Link:
    def __init__(self, source: int, target: int, value: float, color: str = "lightgray"):
        self.source = source
        self.target = target
        self.value = value
        self.color = color

class Model_Sankey:
    """ 
    A class used to build and hold the data for plotting the timing breakdown of a model inference.
    """



    def __init__(self, model: str, hw: str, num_horizontal_layers: int, normalization = False):

        self.nodes: List[Node] = []
        self.links: List[Link] = []
        self.default_plotting_template = pio.templates[pio.templates.default]
        self.colors = self.default_plotting_template.layout.colorway
        self.model = model
        self.hw = hw
        self.num_horizontal_layers = num_horizontal_layers
        self.normalization = normalization
        self.normalization_val = 0

        # Add a inital node with name "Total"
        self._add_node("Total", 0, 0.5, color=self._get_color_for_horizontal_layer(0))

    @staticmethod
    def _calc_pos_aligned_to_top(i: float, total: float) -> float:
        # only works with node.pad = 0

        center_cord = i/(2*total)
        top_border_cord = center_cord- i/(2*total) # Could also be set to 0
        bottom_border_cord = center_cord + i/(2*total)

        return (center_cord, top_border_cord, bottom_border_cord)
    
    def _node_is_leaf(self, name: str) -> bool:
        """
        Checks if a node is a leaf node (i.e., it has no outgoing links).
        """
        for link in self.links:
            if self.nodes[link.source].name == name:
                return False
        return True


    def _get_node_value(self, name:str) -> float:
        """
        Returns the value of a node by its name.
        """
        sum_value_sources = 0
        sum_value_targets = 0
        for link in self.links:
            if self.nodes[link.target].name == name:
                sum_value_targets += link.value
            if self.nodes[link.source].name == name:
                sum_value_sources += link.value
        if sum_value_sources == 0 and sum_value_targets == 0:
            raise ValueError(f"Node '{name}' not found in links.")
        
        if name == "Total" and sum_value_targets != 0 and sum_value_sources != 0:
            return min(sum_value_sources, sum_value_targets)
        return max(sum_value_sources, sum_value_targets)


    def _get_color_for_horizontal_layer(self, horizontal_layer: int) -> str:
        # Return a color from the palette based on the horizontal layer
        return self.colors[horizontal_layer]

    def _get_node_index(self, name: str) -> int:
        for i, node in enumerate(self.nodes):
            if node.name == name:
                return i
        raise ValueError(f"Node '{name}' not found in nodes.")

    def _add_link(self, name: str, value: float, source_name: str, target_name: str, color: str = "lightgray"):
        """
        Adds a link between two nodes in the Sankey diagram.
        """
        source_index = self._get_node_index(source_name)
        target_index = self._get_node_index(target_name)
        self.links.append(Link(source_index, target_index, value, color=color))

    def _add_node(self, name: str, x: float, y: float, color: str = "black"):
        self.nodes.append(Node(name, x, y, color))

    def add_new(self, name: str, value: float, source_name: str, horizontal_layer: int):
        """
        Adds a new entry going from source_name to name.
        """
        if self.normalization:
            value /= self.normalization_val
        
        x = horizontal_layer / (self.num_horizontal_layers-1)
        color = self._get_color_for_horizontal_layer(horizontal_layer)

        y = 0

        self._add_node(name, x, y, color)
        self._add_link(name, value, source_name, name)


    def normalize_values(self, normalization: float):
        # Add an invisible node for normalization with the normalization as value in front of the "Total" node
        self._add_node("Normalization", 0.25, 2, color="rgba(0,0,0,0)")
        loc_tot = self._get_node_value("Total")
        # self._add_link("Normalization", loc_tot, "Normalization", "Total", color="rgba(0,0,0,0)")
        self._add_link("Normalization", normalization-loc_tot, "Total", "Normalization", color="rgba(0,0,0,0)")
        self.normalization_val = normalization-loc_tot

        # for link in self.links:
        #     link.value /= normalization
    

    def hide_horizontal_layer(self, layer: int):
        # Hide a specific horizontal layer and all links to that layer by setting its nodes' visibility to False
        for n in self.nodes:
            if n.x == layer / (self.num_horizontal_layers-1):
                n.color = "rgba(0,0,0,0)"
        for link in self.links:
            if self.nodes[link.source].x == layer / (self.num_horizontal_layers-1) or self.nodes[link.target].x == layer / (self.num_horizontal_layers-1):
                link.color = "rgba(0,0,0,0)"

    def realign_nodes(self):
        bottom_cords = [0] * self.num_horizontal_layers
        for i in range(self.num_horizontal_layers):
            for n in self.nodes:
                if n.x == i / (self.num_horizontal_layers-1):
                    if bottom_cords[i] == 0:
                        if n.name == "Total":
                            vc, vt, vb = self._calc_pos_aligned_to_top(self._get_node_value(n.name), self._get_node_value("Total"))
                        
                        else:
                            # For the other nodes we want it to be aligned to the center of total without normalization
                            vc, vt, vb = self._calc_pos_aligned_to_top(self._get_node_value(n.name), self._get_node_value("Total"))
                        # print(f"Node {n.name} Vertical Center:", vc, "Vertical Top:", vt, "Vertical Bottom:", vb)
                        bottom_cords[i] = vb
                        n.y = vc
                        if abs(vt) >= 10e-6:
                            warnings.warn(f"Node {n.name} has a non-zero top value {vt}. This might lead to misalignment in the Sankey diagram.")
                    else:
                        node_length = self._get_node_value(n.name) / (self._get_node_value("Total"))
                        padding = 0.05
                        n.y = bottom_cords[i] + node_length/2 + padding
                        bottom_cords[i] = n.y + node_length / 2


    def create_figure(self) -> go.Figure:

        labels = [node.name if node.color != "rgba(0,0,0,0)" else "" for node in self.nodes]
        for i, label in enumerate(labels):
            if label == "CPU Backend GEMM":
                labels[i] = "GEMM"
            elif label == "data_in_copy_map":
                labels[i] = "IOMMU page map"
            elif label == "offload_wait":
                labels[i] = "Device work"
        x = [node.x for node in self.nodes]
        y = [node.y for node in self.nodes]
        node_colors = [node.color for node in self.nodes]
        sources = [link.source for link in self.links]
        targets = [link.target for link in self.links]
        values = [link.value for link in self.links]
        link_colors = [link.color for link in self.links]
        # Create a Sankey diagram using the stored data
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=0,
                thickness=20,
                line=dict(color="black", width=0.0),
                label=labels,
                x=x,
                y=y,
                color=node_colors,
                align="left"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            ),
            arrangement="fixed",
        )])

        if self.normalization_val != 0:
            tot = self._get_node_value("Total")
            total_bottom = (tot - self.normalization_val)/tot
            
            # Add shape using the same coordinate system as the Sankey
            fig.add_shape(
                type="rect",
                x0=0, y0=1-total_bottom,           # Start from bottom-left of the lower portion
                x1=0.02, y1=0, # Cover full width, height proportional to missing data
                line=dict(color="white", width=1),
                fillcolor="rgba(255,255,255,1)",
                layer="above",
                xref="paper",  # Use paper coordinates for x (0-1 across full width)
                yref="paper"   # Use paper coordinates for y (0-1 across full height)
            )

        fig.update_layout(title=dict(text=None, automargin=True, yref="paper"), 
                          font_size=16,
                          margin=dict(l=10, r=20, t=10, b=150),
                        )
        return fig
    

    def create_treemap(self) -> go.Figure:
        """
        Creates a treemap figure from the nodes and links.
        """
        labels = [node.name for node in self.nodes]
        values = [self._get_node_value(node.name) for node in self.nodes]
        colors = [node.color for node in self.nodes]
        parents =[""] + [self.nodes[link.source].name if link.source != -1 else "" for link in self.links]
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            textinfo="label+value",
            branchvalues="total",
            texttemplate="%{label}<br>%{value:.2f} s"
        ))
        fig.update_layout(title_text=f"Treemap for {self.model} on {self.hw}", uniformtext=dict(minsize=16, mode="hide"))
        return fig
    

    def create_icicle(self) -> go.Figure:
        """
        Creates an icicle figure from the nodes and links.
        """
        labels = [node.name for node in self.nodes]
        values = [self._get_node_value(node.name) for node in self.nodes]
        colors = [node.color for node in self.nodes]
        parents = [""] + [self.nodes[link.source].name if link.source != -1 else "" for link in self.links]
        fig = go.Figure(go.Icicle(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors, line=dict(color="black", width=0.5)),
            textinfo="label+value+percent parent",
            branchvalues="total",
            tiling=dict(pad=0, orientation="h"),
            sort=False

            # texttemplate="%{label}<br>%{value:.2f} s"
        ))
        fig.update_layout(title_text=f"Icicle for {self.model} on {self.hw}",
                           uniformtext=dict(minsize=16, mode="hide")
                           )
        return fig
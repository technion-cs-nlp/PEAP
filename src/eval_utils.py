import sys

from transformer_lens import HookedTransformer, ActivationCache, utils

from tqdm import tqdm
import torch
import einops
import pandas as pd
from functools import partial
from fancy_einsum import einsum
from typing import  Literal, Dict, List, Set
from typing import NamedTuple
from dataclasses import dataclass, replace
from collections import defaultdict ,OrderedDict
from abc import abstractmethod
import networkx as nx
import heapdict

from pos_aware_edge_attribution_patching import Edge, EAPResults, Experiament

import os



def split_layers_and_heads_for_heads_results(act: torch.Tensor, model: HookedTransformer, num_layers) -> torch.Tensor:
    return einops.rearrange(act, '(layer head) batch seq d_model -> layer batch seq head d_model',
                            layer=num_layers,
                            head=model.cfg.n_heads)

def add_space(x: str):
    if x[0] != " ":
        return " " + x
    return x



@dataclass(frozen=True)
class ModelCfg:
    n_layers: int
    n_heads: int
    parallel_attn_mlp: bool

     
        
@dataclass(frozen=True)
class Node:
    """
    A class representing a node in the abstractcomputational graph of a transformer model.
    """
    layer_idx: int
    head_idx: int
    node_type: str
    span_idx: int
    span_name: str
    model_cfg: ModelCfg
    exp: Experiament

    @abstractmethod
    def get_children(self) -> List['Node']:
        pass

    @abstractmethod
    def get_parents(self) -> List['Node']:
        pass

   
    def __hash__(self):
        return hash((
            self.layer_idx,
            self.head_idx,
            self.node_type,
            self.span_idx,
            self.span_name,
        ))


@dataclass(frozen=True)
class MLPNode(Node):
    """
    A class representing a MLP node in the abstract computational graph of a transformer model.
    """
    def get_input_name(self) -> str:
        return utils.get_act_name(f"mlp_in", self.layer_idx)
   
    def get_output_name(self) -> str:
        return utils.get_act_name(f"mlp_out", self.layer_idx)

    def get_children(self) -> List['Node']:
        children = []
        
        for l in range(self.layer_idx):
            children += [MLPNode(layer_idx=l, #mlp
                           head_idx=None,
                           node_type="m",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]
                         
            children += [AttnOutNode(layer_idx=l, #attn
                           head_idx=h,
                           node_type="h",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp) for h in range(self.model_cfg.n_heads)]
        if not self.model_cfg.parallel_attn_mlp:
            children += [AttnOutNode(layer_idx=self.layer_idx, #attn
                           head_idx=h,
                           node_type="h",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp) for h in range(self.model_cfg.n_heads)]
        
        children += [PreResNode(layer_idx=0, # resid_pre
                           head_idx=None,
                           node_type="r",
                           span_idx=self.span_idx,
                           span_name=self.span_name,
                           model_cfg=self.model_cfg,
                           exp=self.exp)]
        return children

    def get_parents(self) -> List['Node']:
        parents = []
            
        for l in range(self.layer_idx+1, self.model_cfg.n_layers):
            parents += [MLPNode(layer_idx=l,
                           head_idx=None,
                           node_type="m",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]

            parents += [AttnInNode(layer_idx=l,
                           head_idx=h,
                           node_type="h",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp) for h in range(self.model_cfg.n_heads)]
        
        parents += [PostResNode(layer_idx=self.model_cfg.n_layers-1,
                           head_idx=None,
                           node_type="r",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]

            
        return parents

@dataclass(frozen=True)
class PreResNode(Node):
    """
    A class representing a PreRes node in the abstractcomputational graph of a transformer model.
    """
    def get_input_name(self) -> str:
        return utils.get_act_name(f"resid_pre", self.layer_idx)
   
    def get_output_name(self) -> str:
        return utils.get_act_name(f"resid_pre", self.layer_idx)

    def get_children(self) -> List['Node']:
        return []

    def get_parents(self) -> List['Node']:
        parents = []
            
        for l in range(self.model_cfg.n_layers):
            parents += [MLPNode(layer_idx=l,
                           head_idx=None,
                           node_type="m",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]

            parents += [AttnInNode(layer_idx=l,
                           head_idx=h,
                           node_type="h",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp) for h in range(self.model_cfg.n_heads)]
        
        parents += [PostResNode(layer_idx=self.model_cfg.n_layers-1,
                           head_idx=None,
                           node_type="r",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]
        
            
        return parents


@dataclass(frozen=True)
class PostResNode(Node):
    """
    A class representing a PostRes node in the abstract computational graph of a transformer model.
    """
    def get_input_name(self) -> str:
        return utils.get_act_name(f"resid_post", self.layer_idx)
   
    def get_output_name(self) -> str:
        return utils.get_act_name(f"resid_post", self.layer_idx)
    
    def get_children(self) -> List['Node']:
        
        children = []
        
        for l in range(self.model_cfg.n_layers):
            children += [MLPNode(layer_idx=l,
                           head_idx=None,
                           node_type="m",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]
            children += [AttnOutNode(layer_idx=l,
                           head_idx=h,
                           node_type="h",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp) for h in range(self.model_cfg.n_heads)]
        
        children += [PreResNode(layer_idx=0,
                           head_idx=None,
                           node_type="r",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]
        return children

    def get_parents(self) -> List['Node']:
        return []


@dataclass(frozen=True)
class AttnInNode(Node):
    """
    A class representing a AttnIn node in the abstract computational graph of a transformer model.
    """
    def get_input_name(self) -> str:
        return utils.get_act_name(f"attn_in", self.layer_idx)
   
    def get_output_name(self) -> str:
        return utils.get_act_name(f"attn_in", self.layer_idx)
        
    def get_children(self) -> List['Node']:
        children = []
        for l in range(self.layer_idx):
            children += [MLPNode(layer_idx=l,
                           head_idx=None,
                           node_type="m",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]

            children += [AttnOutNode(layer_idx=l,
                           head_idx=h,
                           node_type="h",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp) for h in range(self.model_cfg.n_heads)]
    
        children += [PreResNode(layer_idx=0,
                           head_idx=None,
                           node_type="r",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]
        
        return children

    def get_parents(self) -> List['Node']:
        parents = []

        end_pos = len(self.exp.spans)-1  if self.span_idx < len(self.exp.spans)-1 else len(self.exp.spans) #whether to includ positoins 
        for s_idx in range(self.span_idx, end_pos):
            parents += [AttnOutNode(layer_idx=self.layer_idx,
                head_idx=self.head_idx,
                node_type="h",
                    span_idx=s_idx,
                    span_name=self.exp.spans[s_idx],
                    model_cfg=self.model_cfg,
                    exp=self.exp)]
        
        return parents



@dataclass(frozen=True)
class AttnOutNode(Node):
    """
    A class representing a AttnOut node in the abstractcomputational graph of a transformer model.
    """
    def get_input_name(self) -> str:
        return utils.get_act_name(f"result", self.layer_idx)
   
    def get_output_name(self) -> str:
        return utils.get_act_name(f"result", self.layer_idx)

    def get_children(self) -> List['Node']:
        children = []
        
        start_pos = 0 if self.span_idx < len(self.exp.spans)-1 else self.span_idx #whether to includ positoins 
        for s_idx in range(start_pos,self.span_idx + 1):
            children += [AttnInNode(layer_idx=self.layer_idx,
                head_idx=self.head_idx,
                node_type="h",
                    span_idx=s_idx,
                    span_name=self.exp.spans[s_idx],
                    model_cfg=self.model_cfg,
                    exp=self.exp)]
        
        return children

    def get_parents(self) -> List['Node']:
        parents = []

        for l in range(self.layer_idx+1, self.model_cfg.n_layers):
            parents += [MLPNode(layer_idx=l,
                           head_idx=None,
                           node_type="m",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]

            parents += [AttnInNode(layer_idx=l,
                           head_idx=h,
                           node_type="h",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp) for h in range(self.model_cfg.n_heads)]
        
        if not self.model_cfg.parallel_attn_mlp:
            parents += [MLPNode(layer_idx=self.layer_idx,
                           head_idx=None,
                           node_type="m",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]
            
        parents += [PostResNode(layer_idx=self.model_cfg.n_layers-1,
                           head_idx=None,
                           node_type="r",
                           span_idx=self.span_idx,
                          span_name=self.span_name,
                          model_cfg=self.model_cfg,
                          exp=self.exp)]
        
        return parents

   
def get_edge_score(exp: Experiament, child: Node, parent: Node, span_type: str,full_results: EAPResults, search_type: Literal["min", "max", "abs"]) -> float:
    """
    Return the edge score between two nodes in the computation graph.
    
    This function is used to higher the abstraction level of the computation graph. 
    Instead of looking on k/q/v or k_input/v_input/q_input seperatly we higher the abstraction level, and return a single score for each conection between two nodes.
    Args:
        exp (Experiment): The experiment object containing spans and configuration
        child (Node): The child/upstream node in the computation graph
        parent (Node): The parent/downstream node in the computation graph 
        span_type (str): The type of span to analyze
        full_results (EAPResults): Results object containing edge activation patterns
        search_type (Literal["min", "max", "abs"]): Is the search type min/max/abs

    Returns:
        float: The computed edge score based on the specified search criteria
    """
    is_crossing = child.span_idx != parent.span_idx or (child.node_type == "h" and parent.node_type == "h" and child.layer_idx == parent.layer_idx)
    if parent.node_type == "h":
        if not is_crossing:
            if search_type == "min":
                results = float('inf')
            else:
                results = -float('inf')
            for node_type in ["q","k","v"]:
                edge = Edge(upstream_layer_idx=child.layer_idx,
                upstream_head_idx=child.head_idx,
                upstream_type=child.node_type,
                upstream_full_name=child.get_output_name(),
                downstream_layer_idx=parent.layer_idx,
                downstream_head_idx=parent.head_idx,
                downstream_type=node_type,
                downstream_full_name=utils.get_act_name(f"{node_type}_input",parent.layer_idx),
                span_upstream=None,
                span_downstream=None,
                is_crossing=is_crossing)
                if search_type == "abs":
                    results = max(results, abs(full_results.results[edge][span_type][parent.span_idx]))
                elif search_type == "max":
                    results = max(results, full_results.results[edge][span_type][parent.span_idx])
                else: #min
                    results = min(results, full_results.results[edge][span_type][parent.span_idx])
            return results
        else:
            if child.span_idx == len(exp.spans) -1 : # a circuit without positions -> no crossign edges 
                if search_type == "min":
                    return -float('inf')
                else:
                    return float('inf')
            if search_type == "min":
                results = float('inf')
            else:
                results = -float('inf')
            for node_type in ["q","k","v"]:
                edge = Edge(upstream_layer_idx=child.layer_idx,
                upstream_head_idx=child.head_idx,
                upstream_type=node_type,
                upstream_full_name=utils.get_act_name(f"{node_type}",parent.layer_idx),
                downstream_layer_idx=parent.layer_idx,
                downstream_head_idx=parent.head_idx,
                downstream_type="h",
                downstream_full_name=utils.get_act_name("result",parent.layer_idx),
                span_upstream=child.span_name,
                span_downstream=parent.span_name,
                is_crossing=is_crossing)
                if search_type == "abs":
                    results = max(results, abs(full_results.results[edge][span_type]))
                elif search_type == "max":
                    results = max(results, full_results.results[edge][span_type])
                else: #min
                    results = min(results, full_results.results[edge][span_type])
            return results

    else:
        edge = Edge(upstream_layer_idx=child.layer_idx,
        upstream_head_idx=child.head_idx,
        upstream_type=child.node_type,
        upstream_full_name=child.get_output_name(),
        downstream_layer_idx=parent.layer_idx,
        downstream_head_idx=parent.head_idx,
        downstream_type=parent.node_type,
        downstream_full_name=parent.get_input_name(),
        span_upstream=None,
        span_downstream=None,
        is_crossing=False)
            
        if search_type == "abs":
            results = abs(full_results.results[edge][span_type][child.span_idx])
        else:
            results = full_results.results[edge][span_type][child.span_idx]

        return results






def find_abstract_circuit_by_thresh(model: HookedTransformer, exp: Experiament, span_type: str, full_results: EAPResults, thresh: float, use_abs: bool=True) -> nx.DiGraph:
    """
    Find an abstract circuit with edges above a certain threshold.
    
    This function builds a directed graph where nodes represent model components (attention heads,
    MLPs, etc.) and edges represent connections between them that exceed a threshold score. 

    Args:
        model (HookedTransformer): The transformer model being analyzed
        exp (Experiment): The experiment configuration containing spans and other settings
        span_type (str): The type of span to analyze (e.g. "clean", "corrupted")
        full_results (EAPResults): PEAP scores
        thresh (float): Threshold value for including edges in the circuit
        use_abs (bool, optional): Whether to use absolute values of scores. Defaults to True.

    Returns:
        nx.DiGraph: A circuit.
    """
    
    model_cfg = ModelCfg(n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads, parallel_attn_mlp=model.cfg.parallel_attn_mlp)
    open_list = defaultdict(lambda: heapdict.heapdict())
    abstract_circuit = nx.DiGraph()

    # from res -> input
    res_node = PostResNode(layer_idx=model.cfg.n_layers-1,
                           head_idx=None,
                           node_type="r",
                           span_idx=len(exp.spans) -2,
                           span_name=exp.spans[-2],
                           model_cfg=model_cfg,
                           exp=exp)

    open_list[exp.spans[-2]][res_node] = 1
    abstract_circuit.add_node(res_node)
    for span_name in exp.spans[:-1][::-1]:
        while len(open_list[span_name]) > 0:
            parent, _ = open_list[span_name].popitem()
            for child in parent.get_children():
                if get_edge_score(exp, child, parent, span_type, full_results, use_abs) > thresh:
                    if child not in abstract_circuit.nodes:
                        abstract_circuit.add_node(child)
                    abstract_circuit.add_edge(parent, child)
                    if child not in open_list[child.span_name]:
                        open_list[child.span_name][child] = model.cfg.n_layers - child.layer_idx

    # from input -> res
    open_list = defaultdict(lambda: heapdict.heapdict())
    embed_graph = nx.DiGraph()
    for span_name_idx in range(len(exp.spans[:-1])):
        res_node = PreResNode(layer_idx=0,
                               head_idx=None,
                               node_type="r",
                               span_idx=span_name_idx,
                               span_name=exp.spans[span_name_idx],
                               model_cfg=model_cfg,
                               exp=exp)

        open_list[exp.spans[span_name_idx]][res_node] = 0
        embed_graph.add_node(res_node)
    for span_name in exp.spans[:-1]:
        while len(open_list[span_name]) > 0:
            child, _ = open_list[span_name].popitem()
            for parent in child.get_parents():
                if get_edge_score(exp, child, parent, span_type, full_results, use_abs) > thresh:
                    if parent not in embed_graph.nodes:
                        embed_graph.add_node(parent)
                    embed_graph.add_edge(parent, child)
                    if parent not in open_list[parent.span_name]:
                        open_list[parent.span_name][parent] = parent.layer_idx


    inter_graph = nx.intersection(abstract_circuit, embed_graph)
   
    return inter_graph





def find_abstract_circuit_size_k(model: HookedTransformer , exp: Experiament, span_type: str, full_results: EAPResults, top_k: int, search_type: Literal["abs","max","min"] , with_pos: bool) -> nx.DiGraph:
    """
    Finds an abstract circuit at size k. Starts from the logits and goes to the embeddings.
    The circuit won't necesarly be at size k, but it will be the largest circuit that can be found up to size k.
    Args:
        model (HookedTransformer): The model used for evaluation.
        exp (Experiament): The experiment object.
        span_type (str): The type of span.
        full_results (EAPResults): PEAP scores
        top_k (int): The number of top results to consider.
        search_type (bool): Whether to use absolute values.
        with_pos (bool): Whether to include position information.

    Returns:
        nx.DiGraph: The abstract circuit.
    """

    model_cfg = ModelCfg(n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads, parallel_attn_mlp=model.cfg.parallel_attn_mlp)

    open_list = heapdict.heapdict()
    abstract_circuit = nx.DiGraph()

    # from res -> input
    res_node = PostResNode(layer_idx=model.cfg.n_layers-1,
                           head_idx=None,
                           node_type="r",
                           span_idx=len(exp.spans) -2 if with_pos else len(exp.spans) -1,
                           span_name=exp.spans[-2] if with_pos else exp.spans[-1],
                           model_cfg=model_cfg,
                           exp=exp)
    abstract_circuit.add_node(res_node)
    for child in res_node.get_children():
        if search_type == "min":
            open_list[(child,res_node)] = get_edge_score(exp, child, res_node, span_type, full_results, search_type)
        else:
            open_list[(child,res_node)] = -1 * get_edge_score(exp, child, res_node, span_type, full_results, search_type)
        
    for _ in range(top_k):
        edge, score = open_list.popitem()
        child, parent = edge
        if child not in abstract_circuit.nodes:
            abstract_circuit.add_node(child)
            for new_child in child.get_children():
                if search_type == "min":
                    open_list[(new_child,child)] = get_edge_score(exp, new_child, child, span_type, full_results, search_type)
                else:
                    open_list[(new_child,child)] = -1 * get_edge_score(exp, new_child, child, span_type, full_results, search_type)
        abstract_circuit.add_edge(parent,child, weight=score if search_type == "min" else -1 * score)
        
    
    # from input -> res

    embed_nodes = []
    for span_name_idx in range(len(exp.spans)):
        res_node = PreResNode(layer_idx=0,
                               head_idx=None,
                               node_type="r",
                               span_idx=span_name_idx,
                               span_name=exp.spans[span_name_idx],
                               model_cfg=model_cfg,
                               exp=exp)
        if res_node in abstract_circuit.nodes:
            embed_nodes.append(res_node)
     
    if len(embed_nodes) == 0:
        return nx.DiGraph()
    
    circuit_copy = abstract_circuit.copy()
    for node in circuit_copy.nodes:
        has_path = False
        for res_node in embed_nodes:
            if nx.has_path(circuit_copy, node, res_node):
                has_path = True
                break
        if not has_path:
            abstract_circuit.remove_node(node)
        
    
    return abstract_circuit


def find_abstract_circuit_size_k_reversed(model: HookedTransformer , exp: Experiament, span_type: str, full_results: EAPResults, top_k: int, search_type: Literal["abs","max","min"] , with_pos: bool) -> nx.DiGraph:
    """
    Finds an abstract circuit at size k. Starts from the Embeddings and goes to the logits.
    The circuit won't necesarly be at size k, but it will be the largest circuit that can be found up to size k.

    Args:
        model (HookedTransformer): The model used for evaluation.
        exp (Experiament): The experiment object.
        span_type (str): The type of span.
        full_results (EAPResults): PEAP scores
        top_k (int): The number of top results to consider.
        search_type (bool): Whether to use absolute values.
        with_pos (bool): Whether to include position information.

    Returns:
        nx.DiGraph: The abstract circuit.
    """

    model_cfg = ModelCfg(n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads, parallel_attn_mlp=model.cfg.parallel_attn_mlp)

    open_list = heapdict.heapdict()
    abstract_circuit = nx.DiGraph()

    # from input -> res

    embed_nodes = []
    span_range = [i for i in range(len(exp.spans)-1)] if with_pos else [len(exp.spans) -1]
    for span_name_idx in span_range:
        res_node = PreResNode(layer_idx=0,
                               head_idx=None,
                               node_type="r",
                               span_idx=span_name_idx,
                               span_name=exp.spans[span_name_idx],
                               model_cfg=model_cfg,
                               exp=exp)
        embed_nodes.append(res_node)
    for res_node in embed_nodes:
        for parent in res_node.get_parents():
            if search_type == "min":
                open_list[(res_node, parent)] = get_edge_score(exp, res_node, parent, span_type, full_results, search_type)
            else:
                open_list[(res_node, parent)] = -1 * get_edge_score(exp, res_node, parent, span_type, full_results, search_type)
        
    for _ in range(top_k):
        edge, score = open_list.popitem()
        child, parent = edge
        if parent not in abstract_circuit.nodes:
            abstract_circuit.add_node(parent)
            for new_parent in parent.get_parents():
                if search_type == "min":
                    open_list[(parent, new_parent)] = get_edge_score(exp, parent, new_parent, span_type, full_results, search_type)
                else:
                    open_list[(parent, new_parent)] = -1 * get_edge_score(exp, parent, new_parent, span_type, full_results, search_type)
        abstract_circuit.add_edge(parent,child, weight=score if search_type == "min" else -1 * score)
        
    
    # from input -> res

    res_node = PostResNode(layer_idx=model.cfg.n_layers-1,
                           head_idx=None,
                           node_type="r",
                           span_idx=len(exp.spans) -2 if with_pos else len(exp.spans) -1,
                           span_name=exp.spans[-2] if with_pos else exp.spans[-1],
                           model_cfg=model_cfg,
                           exp=exp)
    
    if res_node not in abstract_circuit.nodes:
        return nx.DiGraph()
    
    circuit_copy = abstract_circuit.copy()
    for node in circuit_copy.nodes:
        if not nx.has_path(circuit_copy, res_node, node):
            abstract_circuit.remove_node(node)
        
    
    return abstract_circuit



@torch.no_grad()
def run_edges_with_mean_ablation(model: HookedTransformer,
                                 test_df: pd.DataFrame,
                                 abstract_circuit: nx.DiGraph,
                                 exp: Experiament,
                                 with_pos: bool,
                                 sample_ablation_size: int,
                                 patch_q=True
                                 ):

    """Run edge ablation by patching activations with mean values.
    This function is used to calculte the faithfulness of a citrcuit.

    Args:
        model (HookedTransformer): The transformer model to run experiments on
        test_df (pd.DataFrame): DataFrame containing test examples
        abstract_circuit (nx.DiGraph): The abstract circuit to run experiments on
        exp (Experiament): Experiment configuration object
        with_pos (bool): Whether to evalute faithfulness with spans (PEAP), or as a non-positional circuit
        sample_ablation_size (int): Number of samples to use for ablation
        patch_q (bool, optional): Whether to patch query vectors. Defaults to True.

    Returns:
        tuple: Contains:
            - pd.DataFrame: Results of ablation experiments
            - dict: Cache of model activations
            - dict: Cache of mean activations
            - list: List of hook functions used
    """
    def ablate_with_mean_cache(value, hook, row):
        layer = int(hook.name.split(".")[1])
        nodes_from_all_spans = [node for node in abstract_circuit.nodes if node.get_input_name() == hook.name] ## all nodes in all spans.

        curr_cache = ActivationCache(clean_cache, model)

        resid_pre = curr_cache['blocks.0.hook_resid_pre'].clone().unsqueeze(0) #layer batch seq d_model
        mean_resid_pre = mean_cache['blocks.0.hook_resid_pre'].clone().unsqueeze(0)#layer batch seq d_model

        if not model.cfg.parallel_attn_mlp:
            if layer == 0:
                if "attn" in hook.name:
                    mlp_out = torch.zeros_like(resid_pre)#layer batch seq d_model
                    mean_mlp_out = mlp_out #layer batch seq d_model
                    mean_heads_output = torch.zeros_like(split_layers_and_heads_for_heads_results(mean_cache.stack_head_results(layer=1), model=model, num_layers=1))  # layer batch seq heads d_model
                    heads_output = mean_heads_output  # layer batch seq heads d_model
                else:
                    mlp_out = torch.zeros_like(resid_pre)#layer batch seq d_model
                    mean_mlp_out = mlp_out #layer batch seq d_model
                    heads_output = split_layers_and_heads_for_heads_results(curr_cache.stack_head_results(layer=1), model=model, num_layers=1).clone()
                    mean_heads_output = split_layers_and_heads_for_heads_results(mean_cache.stack_head_results(layer=1), model=model, num_layers=1).clone()
                    
            else:
                num_layers_attn = layer if "attn" in hook.name else layer + 1
                num_layers_mlp = layer + 1 if "resid" in hook.name else layer
                mlp_out = curr_cache.stack_activation(activation_name="mlp_out", layer=num_layers_mlp).clone()
                mean_mlp_out = mean_cache.stack_activation(activation_name="mlp_out", layer=num_layers_mlp).clone()
                heads_output = split_layers_and_heads_for_heads_results(curr_cache.stack_head_results(layer=num_layers_attn), model=model, num_layers=num_layers_attn).clone()
                mean_heads_output = split_layers_and_heads_for_heads_results(mean_cache.stack_head_results(layer=num_layers_attn), model=model, num_layers=num_layers_attn).clone()

        else:
            if layer == 0:
                mlp_out = torch.zeros_like(resid_pre)#layer batch seq d_model
                mean_mlp_out = mlp_out #layer batch seq d_model
                mean_heads_output = torch.zeros_like(split_layers_and_heads_for_heads_results(mean_cache.stack_head_results(layer=1), model=model, num_layers=1))  # layer batch seq heads d_model
                heads_output = mean_heads_output  # layer batch seq heads d_model
            else:
                num_layers = layer + 1 if "resid" in hook.name else layer
                mlp_out = curr_cache.stack_activation(activation_name="mlp_out", layer=num_layers).clone()
                mean_mlp_out = mean_cache.stack_activation(activation_name="mlp_out", layer=num_layers).clone()
                heads_output = split_layers_and_heads_for_heads_results(curr_cache.stack_head_results(layer=num_layers), model=model, num_layers=num_layers).clone()
                mean_heads_output = split_layers_and_heads_for_heads_results(mean_cache.stack_head_results(layer=num_layers), model=model, num_layers=num_layers).clone()
                
            
        for parent in nodes_from_all_spans:
            span_start = row[exp.spans[parent.span_idx]] if with_pos else 1
            span_end = row[exp.spans[parent.span_idx + 1]] if with_pos else row["length"]
            neighbors = [child for child in abstract_circuit.neighbors(parent)]
            mlps_idx = [node.layer_idx for node in neighbors if node.node_type == "m"]
            mlps_mask = torch.ones_like(mlp_out)
            mlps_mask[mlps_idx, :, span_start:span_end] = 0
            mlp_masked = mlp_out * mlps_mask
            mean_mlp_masked = mean_mlp_out * mlps_mask

            head_layers = [child.layer_idx for child in neighbors if "result" in child.get_output_name()]
            head_heads = [child.head_idx for child in neighbors if "result" in child.get_output_name()]
            heads_mask = torch.ones_like(heads_output)
            heads_mask[head_layers, :,span_start:span_end, head_heads, :] = 0
            heads_masked = heads_output * heads_mask
            mean_heads_masked = mean_heads_output * heads_mask

            resid_idx = [child.layer_idx for child in neighbors if child.node_type == "r"]
            assert len(resid_idx) <= 1
            assert (resid_idx == [0] or resid_idx == [])
            resid_pre_mask = torch.ones_like(resid_pre)
            resid_pre_mask[resid_idx, :, span_start:span_end] = 0
            resid_pre_masked = resid_pre * resid_pre_mask
            mean_resid_pre_masked = mean_resid_pre * resid_pre_mask

            if parent.node_type == "h":
                #vale - (batch seq head d_model)
                value[:, span_start:span_end,parent.head_idx] -= mlp_masked[:,:,span_start:span_end].sum(0)
                value[:, span_start:span_end, parent.head_idx] -= heads_masked[:,:,span_start:span_end].sum((0, 3))
                value[:, span_start:span_end, parent.head_idx] -= resid_pre_masked[:,:,span_start:span_end].sum(0)

                value[:, span_start:span_end,parent.head_idx] += mean_mlp_masked[:,:,span_start:span_end].sum(0)
                value[:, span_start:span_end, parent.head_idx] += mean_heads_masked[:,:,span_start:span_end].sum((0, 3))
                value[:, span_start:span_end, parent.head_idx] += mean_resid_pre_masked[:,:,span_start:span_end].sum(0)
            else:
                # vale - (batch seq d_model)
                value[:, span_start:span_end] -= mlp_masked[:,:,span_start:span_end].sum(0)
                value[:, span_start:span_end] -= heads_masked[:,:,span_start:span_end].sum((0, 3))
                value[:,span_start:span_end] -= resid_pre_masked[:,:,span_start:span_end].sum(0)

                value[:,span_start:span_end] += mean_mlp_masked[:,:,span_start:span_end].sum(0)
                value[:, span_start:span_end] += mean_heads_masked[:,:,span_start:span_end].sum((0, 3))
                value[:, span_start:span_end] += mean_resid_pre_masked[:,:,span_start:span_end].sum(0)

        return value
    
    def hook_crossing_edges(value, hook, row, patch_q=True):
        layer = int(hook.name.split(".")[1])
        nodes_from_all_spans = [node for node in abstract_circuit.nodes if node.get_input_name() == utils.get_act_name('result', layer=layer)]
        for parent in nodes_from_all_spans:
            span_start = row[exp.spans[parent.span_idx]]
            span_end = row[exp.spans[parent.span_idx + 1]]
            neighbors = [child.span_idx for child in abstract_circuit.neighbors(parent)]
            for child_span_idx in [idx for idx in range(parent.span_idx + 1) if idx not in neighbors]:
                child_span_start = row[exp.spans[child_span_idx]]
                child_span_end = row[exp.spans[child_span_idx + 1]]
                v = clean_cache[utils.get_act_name('v', layer=layer)][:, child_span_start:child_span_end, parent.head_idx].clone()
                pattern = clean_cache[utils.get_act_name('pattern', layer=layer)][:,parent.head_idx, span_start: span_end,  child_span_start: child_span_end].clone().to(model.cfg.dtype)
                z = einsum(
                "batch key_pos  d_head, \
                batch  query_pos key_pos -> \
                batch query_pos  d_head",
                v,
                pattern,
            )
                v_mean = mean_cache[utils.get_act_name('v', layer=layer)][:, child_span_start:child_span_end, parent.head_idx].clone()
                mean_pattern = mean_cache[utils.get_act_name('pattern', layer=layer)][:,parent.head_idx, span_start: span_end,  child_span_start: child_span_end].clone().to(model.cfg.dtype)
                if patch_q:
                    z_mean = einsum(
                    "batch key_pos  d_head, \
                    batch  query_pos key_pos -> \
                    batch query_pos  d_head",
                    v_mean,
                    mean_pattern,
                        )
                else:
                    z_mean = einsum(
                    "batch key_pos  d_head, \
                    batch  query_pos key_pos -> \
                    batch query_pos  d_head",
                    v_mean,
                    pattern,
                        )
                
                value[:, span_start: span_end, parent.head_idx] -= z
                value[:, span_start: span_end, parent.head_idx] += z_mean

        return value
    
    model.reset_hooks()
    faithfulness_list, acc_correct_list, avg_circuit_size_list, circuit_prediction_list, model_prediction_list = [],[],[],[],[]

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.clone().detach()

    hooks_out = [(utils.get_act_name('resid_pre', layer=0), forward_cache_hook)]
    hooks_out += [(utils.get_act_name('result', layer=l), forward_cache_hook) for l in range(model.cfg.n_layers)]
    hooks_out += [(utils.get_act_name('mlp_out', layer=l), forward_cache_hook) for l in range(model.cfg.n_layers)]
    hooks_out += [(utils.get_act_name('v', layer=l), forward_cache_hook) for l in range(model.cfg.n_layers)]
    hooks_out += [(utils.get_act_name('pattern', layer=l), forward_cache_hook) for l in range(model.cfg.n_layers)]
    hooks_in_names = list(set([node.get_input_name() for node in abstract_circuit.nodes if "resid_pre" not in node.get_input_name() and "result" not in node.get_input_name()]))
    if with_pos:
        hooks_crossing_names = list(set([utils.get_act_name('z', layer=node.layer_idx) for node in abstract_circuit.nodes if "result" in node.get_input_name()]))


    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        avg_circuit_size_list.append(calculate_circuit_size(row, abstract_circuit, exp, with_pos))
        mean_cache,df_c = exp.get_mean_activations(model=model, row=row, index=index, seed=exp.seed, sample_ablation_size=sample_ablation_size)
        hooks_in = [(hook_name, partial(ablate_with_mean_cache, row=row))
                    for hook_name in hooks_in_names]
        hooks = hooks_in + hooks_out
        if with_pos:
            hooks_crossing = [(hook_name, partial(hook_crossing_edges, row=row,patch_q=patch_q)) for hook_name in hooks_crossing_names]
            hooks += hooks_crossing
        clean_cache = {}
        model.reset_hooks()
        clean_logits = model(row["prompt"])
        circuit_logits = model.run_with_hooks(row["prompt"], fwd_hooks=hooks)
        del clean_cache, mean_cache, hooks_in, hooks
        if with_pos:
            del hooks_crossing
        torch.cuda.empty_cache()
        faithfulness_list.append(exp.metric(circuit_logits, row, model).item())
        acc_correct_list.append(1 if model.tokenizer.decode(clean_logits[0, row[exp.spans[-2]]].argmax()) == model.tokenizer.decode(circuit_logits[0, row[exp.spans[-2]]].argmax()) else 0)
        circuit_prediction_list.append(model.tokenizer.decode(circuit_logits[0, row[exp.spans[-2]]].argmax()))
        model_prediction_list.append(model.tokenizer.decode(clean_logits[0, row[exp.spans[-2]]].argmax()))
    
    return faithfulness_list, acc_correct_list, avg_circuit_size_list, circuit_prediction_list, model_prediction_list


def calculate_circuit_size(exmaple: pd.DataFrame,  abstract_circuit: nx.DiGraph, exp: Experiament, with_pos: bool) -> int:
    """
    Calculates the number of edges in a circuit. This is used to calculate the exact size of the circuit (not the abstract size).

    Args:
        exmaple (pd.DataFrame): A exmaple from the test dataset containing span information
        abstract_circuit (nx.DiGraph): The abstract circuit 
        exp (Experiament): The experiment object containing span definitions
        with_pos (bool): Whether to use positional information when calculating spans

    Returns:
        int: The total number of edges in the circuit, weighted by node type and span sizes
    """
    num_edges = 0
    for node in abstract_circuit.nodes:
        span_start_downstream = exmaple[exp.spans[node.span_idx]] if with_pos else 1
        span_end_downstream = exmaple[exp.spans[node.span_idx + 1]] if with_pos else exmaple["length"]
        num_nodes_per_span_downstream = span_end_downstream - span_start_downstream
        if node.node_type == "h" and node.get_input_name() == utils.get_act_name('result', layer=node.layer_idx):
            for child in abstract_circuit.neighbors(node):
                if child.span_idx != node.span_idx: 
                    span_start_upstream = exmaple[exp.spans[child.span_idx]] 
                    span_end_upstream = exmaple[exp.spans[child.span_idx + 1]]
                    num_nodes_per_span_upstream = span_end_upstream - span_start_upstream
                    num_edges += 3 * num_nodes_per_span_upstream * num_nodes_per_span_downstream
                else:
                    num_edges += 3 * (num_nodes_per_span_downstream * (num_nodes_per_span_downstream + 1)) / 2
        
        else:
            for child in abstract_circuit.neighbors(node):
                num_edges += num_nodes_per_span_downstream if node.node_type != "h" else 3 * num_nodes_per_span_downstream
                
                
    return num_edges

    
    

@torch.no_grad()
def calculate_prior_performance(model: HookedTransformer,
                                 test_df: pd.DataFrame,
                                 exp: Experiament,
                                 sample_ablation_size: int
                                 ) -> float:
    """
    Evalute the performance of a model on a given test dataset.

    Args:
        model (HookedTransformer): The model to evaluate.
        test_df (pd.DataFrame): The test dataset.
        exp (Experiament): The experiment object.
        sample_ablation_size: The size of the ablation samples.

    Returns:
        float: The prior performance of the model on the test dataset.
    """

    total_metric = []
    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        _,df_mean = exp.get_mean_activations(model=model, row=row, index=index, seed=exp.seed,sample_ablation_size=sample_ablation_size)
        for index_2, row_2 in df_mean.iterrows():
            logits = model(row_2["prompt"])
            total_metric.append(exp.metric(logits=logits, row=row, model=model))
    return torch.stack(total_metric).mean().item()


def full_computation_graph_size(model: HookedTransformer, exp: Experiament, df : pd.DataFrame, use_point_of_diff: bool) -> int:
    
    """
    Calculates the mean size of the full computation graph (not the abstract size) for a model and a given dataset.

    Args:
        model (HookedTransformer): The transformer model to analyze
        exp (Experiament): The experiment configuration object
        df (pd.DataFrame): DataFrame containing the input examples
        use_point_of_diff (bool): Whether to only consider computation up to the point of difference
            between examples

    Returns:
        int: Total number of nodes in the computation graph averaged across examples
    """
    model_cfg = ModelCfg(n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads, parallel_attn_mlp=model.cfg.parallel_attn_mlp)
    
    mean_nodes = 0 # the embedding layer
    mean_edges = 0
    for index, row in df.iterrows():
        if not use_point_of_diff:
            prompt_length = row["length"] - 1
        else:
            prompt_list = model.to_str_tokens(row["prompt"])
            if "ioi" in exp.exp_name:
                diff_index = min(prompt_list.index(row["IO_token"]), prompt_list.index(row["S1_token"]))
         
            if "wino_bias" in exp.exp_name:
                diff_index = 2
            
            if "greater_than" in exp.exp_name:
                diff_index = 8
            for span_idx in range(len(exp.spans)):
                if diff_index < row[exp.spans[span_idx]]:
                    prompt_length = row["length"] - row[exp.spans[span_idx - 1]]
                    break
        mean_nodes += prompt_length #embedding layer
        node = PostResNode(layer_idx=model.cfg.n_layers-1,  # residual node at the last layer
                            head_idx=None,
                            node_type="r",
                            span_idx=len(exp.spans) - 1,
                            span_name=exp.spans[-1],
                            model_cfg=model_cfg,
                            exp=exp)
        mean_nodes += prompt_length
        mean_edges += len(node.get_children()) * prompt_length
        
            
        for layer in range(model.cfg.n_layers):
            for head_idx in range(model.cfg.n_heads):
                node = AttnInNode(layer_idx=layer,
                                        head_idx=head_idx,
                                        node_type="h",
                                        span_idx=len(exp.spans) - 1,
                                        span_name=exp.spans[-1],
                                        model_cfg=model_cfg,
                                        exp=exp)
                mean_nodes += prompt_length
                mean_edges += 3 * len(node.get_children()) * prompt_length
                mean_edges += 3 * (prompt_length * (prompt_length + 1)/2)

                
            node = MLPNode(layer_idx=layer,
                                    head_idx=None,
                                    node_type="m",
                                    span_idx=len(exp.spans) - 1,
                                    span_name=exp.spans[-1],
                                    model_cfg=model_cfg,
                                    exp=exp)
            mean_nodes += prompt_length
            mean_edges += len(node.get_children()) * prompt_length



    return mean_edges/df.shape[0], mean_nodes/df.shape[0]



@torch.no_grad()
def calculate_model_prior_performance(model: HookedTransformer,
                                 test_df: pd.DataFrame,
                                 exp: Experiament,
                                 sample_ablation_size
                                 ) -> float:
    """
    Calculates the prior performance of a model on a given test dataset.

    Args:
        model (HookedTransformer): The model to evaluate.
        test_df (pd.DataFrame): The test dataset.
        exp (Experiament): The experiment object.
        sample_ablation_size: The size of the ablation samples.

    Returns:
        float: The prior performance of the model on the test dataset.
    """

    total_metric = []
    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        _,df_mean = exp.get_mean_activations(model=model, row=row, index=index, seed=exp.seed,sample_ablation_size=sample_ablation_size)
        for index_2, row_2 in df_mean.iterrows():
            logits = model(row_2["prompt"])
            total_metric.append(exp.metric(logits=logits, row=row, model=model))
    return torch.stack(total_metric).mean().item()


def circuit_size_without_pos(model: HookedTransformer, exp: Experiament, circuit: nx.DiGraph) -> tuple[int, int]:
    """
    Calculates the number of edges and nodes in a circuit without positions.
    Args:
        model (HookedTransformer): The model to evaluate.
        exp (Experiament): The experiment object.
        circuit (nx.DiGraph): The graph representing the circuit.

    Returns:
        tuple: Contains:
            - int: The number of edges in the circuit
            - int: The number of nodes in the circuit
    """
    model_cfg = ModelCfg(n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads, parallel_attn_mlp=model.cfg.parallel_attn_mlp)
    edges = 0
    nodes = []
    for span_idx in range(len(exp.spans[:-1])):
        nodes.append(PostResNode(layer_idx=model.cfg.n_layers-1,
                          head_idx=None,
                          node_type="r",
                          span_idx=span_idx,
                          span_name=exp.spans[span_idx],
                          model_cfg=model_cfg,
                          exp=exp))
        
        for layer in range(model.cfg.n_layers):
            for head_idx in range(model.cfg.n_heads):
                if AttnInNode(layer_idx=layer,
                                      head_idx=head_idx,
                                      node_type="h",
                                      span_idx=len(exp.spans) - 1,
                                      span_name=exp.spans[-1],
                                      model_cfg=model_cfg,
                                      exp=exp) in circuit.nodes:
                    nodes.append(AttnInNode(layer_idx=layer,
                                      head_idx=head_idx,
                                      node_type="h",
                                      span_idx=span_idx,
                                      span_name=exp.spans[span_idx],
                                      model_cfg=model_cfg,
                                      exp=exp))
                if AttnOutNode(layer_idx=layer,
                                      head_idx=head_idx,
                                      node_type="h",
                                      span_idx=len(exp.spans) - 1,
                                      span_name=exp.spans[-1],
                                      model_cfg=model_cfg,
                                      exp=exp) in circuit.nodes:
                    nodes.append(AttnOutNode(layer_idx=layer,
                                      head_idx=head_idx,
                                      node_type="h",
                                      span_idx=span_idx,
                                      span_name=exp.spans[span_idx],
                                      model_cfg=model_cfg,
                                      exp=exp))
            if MLPNode(layer_idx=layer,
                                  head_idx=None,
                                  node_type="m",
                                  span_idx=len(exp.spans) - 1,
                                  span_name=exp.spans[-1],
                                  model_cfg=model_cfg,
                                  exp=exp) in circuit.nodes:
                nodes.append(MLPNode(layer_idx=layer,
                                    head_idx=None,
                                    node_type="m",
                                    span_idx=span_idx,
                                    span_name=exp.spans[span_idx],
                                    model_cfg=model_cfg,
                                    exp=exp))
        
    for node in nodes:
        for child in node.get_children():
            temp_child = replace(child, span_idx=len(exp.spans) - 1, span_name=exp.spans[-1])
            temp_node = replace(node, span_idx=len(exp.spans) - 1, span_name=exp.spans[-1])
            if (temp_node, temp_child) in circuit.edges:
                edges += 1

    return edges, len(nodes)



@torch.no_grad()
def calculate_model_performance(model: HookedTransformer, exp: Experiament, test_df: pd.DataFrame) -> float:
    """
    Calculates the model performance by evaluating it on a test dataset.

    Args:
        model (HookedTransformer): The trained model to evaluate.
        exp (Experiament): The experiment object containing the metric function.
        test_df (pd.DataFrame): The test dataset to evaluate the model on.

    Returns:
        float: The average model performance score.
    """
    diff_list = [] 
    model.reset_hooks()
    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        logits = model(row["prompt"])
        diff_list.append(exp.metric(logits, row, model))

    diff = torch.stack(diff_list).mean().item()
    print("model performance:", diff)
    return diff
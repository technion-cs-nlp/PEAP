from transformer_lens import HookedTransformer, ActivationCache, utils
from tqdm import tqdm
import torch
import einops
import pandas as pd
import pickle
from typing import Dict, Optional, List, Union
import argparse
from dataclasses import dataclass
import numpy as np
from typing import  Dict, Optional, List, Union
from jaxtyping import Float
from exp import Experiament, IOI, GreaterThan, WinoBias, prob_diff, logit_diff


def calculate_z_scores(
        v: Float[torch.Tensor, "batch key_pos head_index d_head"],
        pattern: Float[torch.Tensor, "batch head_index query_pos key_pos"],
    ) -> Float[torch.Tensor, "batch query_pos head_index d_head"]:
    v_ = einops.rearrange(
        v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
    )
    pattern_ = einops.rearrange(
        pattern,
        "batch head_index query_pos key_pos -> batch head_index query_pos key_pos",
    )
    return einops.rearrange(
            pattern_ @ v_,
            "batch head_index query_pos d_head -> batch query_pos head_index d_head",
        )

 
def calculate_attention_scores(
        q: Float[torch.Tensor, "batch query_pos head_index d_head"],
        k: Float[torch.Tensor, "batch key_pos head_index d_head"],
        attn_scale
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
    q_ = einops.rearrange(
        q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
    )
    k_ = einops.rearrange(
        k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
    )
    attn_scores = q_ @ k_ / attn_scale
    return attn_scores   


def calculate_heads_output(model, layer_idx, z):
    w = einops.rearrange(
                    model.W_O[layer_idx],
                    "head_index d_head d_model -> d_model head_index d_head",
                )
    return einops.einsum(
            z,
            w,
            "... head_index d_head, d_model head_index d_head -> ... head_index d_model",
        ) # [batch, pos, head_index, d_model]
   


def get_hooked_activations(model: HookedTransformer, clean: pd.DataFrame, counterfactual: pd.DataFrame, exp: Experiament):
    model.reset_hooks()
    clean_cache = {}
    clean_grad_cache = {}
    counterfactual_cache = {}

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()
    
    kq_pref = "rot_" if model.cfg.positional_embedding_type == "rotary" else ""
    
    def edge_outgoing_filter(name): return name.endswith(
        ("hook_result", "hook_mlp_out", "hook_resid_pre",
         "hook_v", "hook_pattern", "hook_z", "attn_scores", f"hook_{kq_pref}q", f"hook_{kq_pref}k"))

    model.add_hook(edge_outgoing_filter, forward_cache_hook, "fwd")

    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()

    def edge_back_filter(name): return name.endswith(
        ("hook_q_input", "hook_k_input", "hook_v_input", "hook_resid_post", "hook_mlp_in", "hook_result","attn_out"))

    model.add_hook(edge_back_filter, backward_cache_hook, "bwd")

    logits = model(clean["prompt"], return_type="logits")  # batch seq d

    diff = exp.metric(logits, clean, model)
    diff.backward()

    # cache the activations of the corrupted inputs
    model.reset_hooks()

    def forward_corrupted_cache_hook(act, hook):
        counterfactual_cache[hook.name] = act.detach()

    model.add_hook(edge_outgoing_filter, forward_corrupted_cache_hook, "fwd")
    with torch.no_grad():
        model(counterfactual["prompt"])
    model.reset_hooks()

    clean_cache = ActivationCache(clean_cache, model)
    counterfactual_cache = ActivationCache(counterfactual_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, counterfactual_cache, clean_grad_cache


@dataclass(frozen=True)
class Edge:

    upstream_layer_idx: int
    upstream_head_idx: Optional[int]  # None for mlp and residual
    upstream_type: str
    upstream_full_name: str
    downstream_layer_idx: int
    downstream_head_idx: Optional[int]  # None for mlp and residual
    downstream_type: str
    downstream_full_name: str
    span_upstream: Optional[str] = None # Only for crossing edges
    span_downstream: Optional[str] = None # Only for crossing edges
    is_crossing: bool = False

    def __hash__(self):
        return hash((
            self.upstream_layer_idx,
            self.upstream_head_idx,
            self.upstream_type,
            self.upstream_full_name,
            self.downstream_layer_idx,
            self.downstream_head_idx,
            self.downstream_type,
            self.downstream_full_name,
            self.span_upstream,
            self.span_downstream,
            self.is_crossing
        ))


@dataclass
class EdgeScore:
    """
    EdgeScore is a dataclass that contains the scores for an edge in position-aware edge attribution patching (PEAP).
    
    PEAP computes attribution scores between different spans of tokens in the input sequence.
    There are two types of edges:

    1. Non-crossing edges (edges within the same residual stream position):
        - Scores are stored as numpy arrays, input per span, the i'th score is the attribution score for the i'th span
        - The last score corresponds to the full prompt (span="length") without any span segmentation
        
    2. Crossing edges (attention edges between different positions):
        - Each score represents an attribution between two different spans

        
    The scores capture different aggregation methods:
    - avg_score: Mean attribution across positions at the same span
    - sum_score: Total attribution across positions at the same span  
    - sum_abs_pos_score: Sum of absolute values at the same span
    - sum_abs_exp_score: Sum of absolute values of expected attributions at the same span
    - max_abs_score: Maximum absolute attribution value at the same span
    """
    avg_score: Union[List[np.ndarray], np.ndarray]
    sum_score: Union[List[np.ndarray], np.ndarray]
    sum_abs_pos_score: Union[List[np.ndarray], np.ndarray]
    sum_abs_exp_score:Union[List[np.ndarray], np.ndarray]
    max_abs_score: Union[List[np.ndarray], np.ndarray]

    def __init__(self, avg_score: np.ndarray, sum_score: np.ndarray, sum_abs_pos_score: np.ndarray, sum_abs_exp_score: np.ndarray, max_abs_score: np.ndarray):
        self.avg_score = [avg_score]
        self.sum_score = [sum_score]
        self.sum_abs_pos_score = [sum_abs_pos_score]
        self.sum_abs_exp_score = [sum_abs_exp_score]
        self.max_abs_score = [max_abs_score]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


@dataclass
class EAPResults:
    dataset_name: str
    dataset_size: int
    dataset: List[str]
    counterfactual_name: str
    counterfactual: List[str]
    span_names: List[str]
    results: Union[Dict[Edge, EdgeScore], List[EdgeScore]]

    def __init__(self, dataset_name: str, dataset_size: int, dataset: List[str], counterfactual_name: str, counterfactual: List[str], span_names: List[str],
                 results=None):
        if results is None:
            results = {}
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.dataset = dataset
        self.counterfactual_name = counterfactual_name
        self.counterfactual = counterfactual
        self.span_names = span_names
        self.results = results

    def __getitem__(self, key):
        return getattr(self, key)

    def update_score(self, edge: Edge, scores: Dict[str, torch.Tensor]):
        if edge not in self.results:
            self.results[edge] = EdgeScore(avg_score=scores["avg"], sum_score=scores["sum"], sum_abs_pos_score=scores["sum_abs_pos"], sum_abs_exp_score=scores["sum_abs_exp"], max_abs_score=scores["max_abs"])
        else:
            self.results[edge].avg_score.append(scores["avg"])  
            self.results[edge].sum_score.append(scores["sum"])
            self.results[edge].sum_abs_pos_score.append(scores["sum_abs_pos"])
            self.results[edge].sum_abs_exp_score.append(scores["sum_abs_exp"])
            self.results[edge].max_abs_score.append(scores["max_abs"])
    
    def get_average_scores(self):
        for edge in self.results.keys():
            self.results[edge].avg_score = np.mean(self.results[edge].avg_score, axis=0)
            self.results[edge].sum_score = np.mean(self.results[edge].sum_score, axis=0)
            self.results[edge].sum_abs_pos_score = np.mean(self.results[edge].sum_abs_pos_score, axis=0)
            self.results[edge].sum_abs_exp_score = np.mean(self.results[edge].sum_abs_exp_score, axis=0)
            self.results[edge].max_abs_score = np.mean(self.results[edge].max_abs_score, axis=0)


@dataclass(frozen=True)
class Node:
    layer_idx: int
    head_idx: Union[int, None]
    node_type: str
    full_name: str


def create_nodes(model: HookedTransformer) -> List[Node]:
    nodes = [Node(layer_idx=0,
                  head_idx=None,
                  node_type="r",
                  full_name=utils.get_act_name("resid_pre", 0))]
    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            nodes.append(Node(layer_idx=layer,
                              head_idx=head_idx,
                              node_type="h",
                              full_name=utils.get_act_name("result", layer)))
        nodes.append(Node(layer_idx=layer,
                          head_idx=None,
                          node_type="m",
                          full_name=utils.get_act_name("mlp_out", layer)))
    return nodes


def split_layers_and_heads(act: torch.Tensor, model: HookedTransformer) -> torch.Tensor:
    return einops.rearrange(act, '(layer head) batch seq d_model -> layer head batch seq d_model',
                            layer=model.cfg.n_layers,
                            head=model.cfg.n_heads)


def split_layers_and_heads_for_heads_results(act: torch.Tensor, model: HookedTransformer) -> torch.Tensor:
    return einops.rearrange(act, '(layer head) batch seq d_model -> layer batch seq head d_model',
                            layer=model.cfg.n_layers,
                            head=model.cfg.n_heads)




def position_aware_edge_attribution_patching(
                              exp: Experiament,
                              dataset_size: int,
                              save_path: str,
                              ) -> None:

    """
    This function is used to compute the position-aware edge attribution patching (PEAP) for a given experiment.
    It starts by computing the in position edges, and then the crossing edges (attention edges between different positions).
    """

    print(f"Computing PEAP for {exp.exp_name} with {dataset_size} examples")

    dtype = "bf16" if "Llama" in exp.model_name else "float32"
    model = HookedTransformer.from_pretrained(
        exp.model_path,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda",
        dtype=dtype
    )
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.set_ungroup_grouped_query_attention(True)
    
    kq_pref = "rot_" if model.cfg.positional_embedding_type == "rotary" else ""
    
    
    ds_clean, ds_counterfactual = exp.create_datasets()
    
    random_indices = np.random.choice(range(len(ds_clean)), size=dataset_size, replace=False)

# Select the same rows from both datasets
    ds_clean = ds_clean.iloc[random_indices]
    ds_counterfactual = ds_counterfactual.iloc[random_indices]

    nodes = create_nodes(model)

    results = EAPResults(dataset_name=exp.clean_dataset_path,
                         dataset_size=dataset_size,
                         dataset=ds_clean["prompt"].to_list(),
                         counterfactual_name=exp.counter_dataset_path,
                         counterfactual=ds_counterfactual["prompt"].to_list() if exp.ablation_type == "counterfactual" else [],
                         span_names=exp.spans
                         )
    for index, row in tqdm(ds_clean.iterrows(), total=ds_clean.shape[0]):
        print("clean promp", row["prompt"])
        if exp.ablation_type == "counterfactual":
            print("counter prompt", ds_counterfactual.loc[index]["prompt"])
            
        if exp.ablation_type == "mean":
            mean_activations, mean_df  = exp.get_mean_activations(model=model, row=row, index=index, seed=exp.seed)
            del mean_df
        

        masks, lengths, indices = exp.get_span_masks(row, model, dtype=model.cfg.dtype)

        clean_cache, counter_cache, clean_grad_cache = get_hooked_activations(model, clean=row, counterfactual=ds_counterfactual.loc[index], exp=exp)

        # activations
        # heads output
        clean_head_act = split_layers_and_heads(clean_cache.stack_head_results(), model=model).clone().to("cuda")  # layer head batch seq d_model
        clean_head_results_act = split_layers_and_heads_for_heads_results(clean_cache.stack_head_results(), model=model).clone().to("cuda")  # layer batch seq head d_model
        # mlp output
        clean_mlp_act = clean_cache.stack_activation(activation_name="mlp_out").clone().to("cuda")  # layer batch seq d_model
        # residual 0 output
        clean_resid_pre_act = clean_cache['blocks.0.hook_resid_pre'].clone().to("cuda")  # batch seq d_model

        if exp.ablation_type == "mean":
            counter_head_act = split_layers_and_heads(mean_activations.stack_head_results(), model=model).clone().to("cuda")
            counter_mlp_act = mean_activations.stack_activation(activation_name="mlp_out").to("cuda").clone().clone() # layer batch seq d_model
            counter_resid_pre_act = mean_activations['blocks.0.hook_resid_pre'].to("cuda").clone().clone()  # batch seq d_model
        elif exp.ablation_type == "zero":
            counter_head_act = torch.zeros_like(clean_head_act,dtype=model.cfg.dtype)
            counter_mlp_act = torch.zeros_like(clean_mlp_act,dtype=model.cfg.dtype)
            counter_resid_pre_act = torch.zeros_like(clean_resid_pre_act,dtype=model.cfg.dtype)
        elif exp.ablation_type == "counterfactual":
            counter_head_act = split_layers_and_heads(counter_cache.stack_head_results(), model=model).clone().to("cuda")
            counter_mlp_act = counter_cache.stack_activation(activation_name="mlp_out").clone().to("cuda")  # layer batch seq d_model
            counter_resid_pre_act = counter_cache['blocks.0.hook_resid_pre'].clone().to("cuda")  # batch seq d_model

        clean_v_act = clean_cache.stack_activation(activation_name="v").clone().to("cuda")  # layer batch seq head d_head

        clean_k_act = clean_cache.stack_activation(activation_name=kq_pref + "k").clone().to("cuda")  # layer batch seq head d_head
        clean_q_act = clean_cache.stack_activation(activation_name=kq_pref + "q").clone().to("cuda")  # layer batch seq head d_head

        # attn pattern
        clean_attn_pattern = clean_cache.stack_activation(activation_name="pattern").clone().to("cuda").to(model.cfg.dtype)  # layer batch head seq seq #For numerical reaosn the pattern is float32

        clean_attn_scores = clean_cache.stack_activation(activation_name="attn_scores").clone().to("cuda").to(model.cfg.dtype) # layer batch head seq seq #For numerical reaosn the pattern is float32
        # gradients
        clean_z_scores = clean_cache.stack_activation(activation_name="z").clone().to("cuda")  # layer batch seq head d_head
        grad_q = clean_grad_cache.stack_activation(activation_name="q_input").clone().to("cuda")  # layer batch seq head dim
        grad_q = einops.rearrange(grad_q, "layer batch seq n_heads d -> layer n_heads batch seq d")
        
        grad_k = clean_grad_cache.stack_activation(activation_name="k_input").clone().to("cuda")  # layer batch seq head dim
        grad_k = einops.rearrange( grad_k, "layer batch seq n_heads d -> layer n_heads batch seq d")
        
        grad_v = clean_grad_cache.stack_activation(activation_name="v_input").clone().to("cuda")  # layer batch seq head dim
        grad_v = einops.rearrange(grad_v, "layer batch seq n_heads d -> layer n_heads batch seq d")
        
        grad_mlp = clean_grad_cache.stack_activation(activation_name="mlp_in").clone().to("cuda")  # layer batch seq dim
        grad_resid = clean_grad_cache.stack_activation(activation_name="resid_post").clone().to("cuda")  # layer batch seq dim
        grad_heads_output = clean_grad_cache.stack_activation(activation_name="result").clone().to("cuda")  # layer batch seq head d_head

        for i in tqdm(range(len(nodes))):
            node = nodes[i]
            if node.node_type == "m":
                clean = clean_mlp_act[node.layer_idx].clone()  #  batch seq d
                counter = counter_mlp_act[node.layer_idx].clone()   #  batch seq d
            elif node.node_type == "h":
                clean = clean_head_act[node.layer_idx, node.head_idx].clone()  #  batch seq d
                counter = counter_head_act[node.layer_idx, node.head_idx].clone()
            else:
                clean = clean_resid_pre_act.clone()  # batch seq d
                counter = counter_resid_pre_act.clone()  # batch seq d

            attribute_score_mlp = (grad_mlp[node.layer_idx:] * (counter - clean)).unsqueeze(1).repeat(1, len(exp.spans), 1, 1, 1)  # layer span batch seq d model
            attribute_score_mlp_masked = torch.where(masks.unsqueeze(0).repeat(attribute_score_mlp.shape[0], 1, 1, 1, 1) == 1, attribute_score_mlp, 0)
            attribute_score_mlp_avg = (attribute_score_mlp_masked.sum(dim=( 2, 3, 4)) / lengths.unsqueeze(0).repeat(attribute_score_mlp.shape[0], 1)).detach().to(torch.float32).cpu().numpy()
            attribute_score_mlp_sum = (attribute_score_mlp_masked.sum(dim=(2, 3, 4))).detach().to(torch.float32).cpu().numpy()
            attribute_score_mlp_sum_abs_pos = (attribute_score_mlp_masked.sum(4).abs().sum(dim=(2, 3))).detach().to(torch.float32).cpu().numpy()
            attribute_score_mlp_sum_abs_exp = (attribute_score_mlp_masked.sum((3,4)).abs().sum(dim=2)).detach().to(torch.float32).cpu().numpy()
            attribute_score_mlp_max_abs = (attribute_score_mlp_masked.sum(4).abs().max(3).values.sum(2)).detach().to(torch.float32).cpu().numpy()

            del attribute_score_mlp_masked, attribute_score_mlp, 

            attribute_score_resid = (grad_resid[-1] * (counter - clean)).unsqueeze(0).repeat(len(exp.spans), 1, 1, 1)  # span batch seq d model
            attribute_score_resid_masked = torch.where( masks == 1, attribute_score_resid, 0)
            attribute_score_resid_avg = (attribute_score_resid_masked.sum(dim=(1, 2, 3)) / lengths).detach().to(torch.float32).cpu().numpy()
            attribute_score_resid_sum = (attribute_score_resid_masked.sum(dim=(1, 2, 3))).detach().to(torch.float32).cpu().numpy()
            attribute_score_resid_sum_abs_pos = (attribute_score_resid_masked.sum(3).abs().sum(dim=(1, 2))).detach().to(torch.float32).cpu().numpy()
            attribute_score_resid_sum_abs_exp = (attribute_score_resid_masked.sum((2,3)).abs().sum(dim=(1))).detach().to(torch.float32).cpu().numpy()
            attribute_score_resid_max_abs = (attribute_score_resid_masked.sum(3).abs().max(2).values.sum(1)).detach().to(torch.float32).cpu().numpy()

            del attribute_score_resid, attribute_score_resid_masked

            attribute_score_q = (grad_q[node.layer_idx:] * (counter - clean)).unsqueeze(3).repeat(1, 1, len(exp.spans), 1, 1, 1)  # layer haed span batch seq d_model
            attribute_score_q_masked = torch.where(masks.unsqueeze(0).unsqueeze(1).repeat(attribute_score_q.shape[0], model.cfg.n_heads, 1, 1, 1, 1) == 1, attribute_score_q, 0)
            attribute_score_q_avg = (attribute_score_q_masked.sum(dim=(3, 4, 5)) / lengths.unsqueeze( 0).unsqueeze(1).repeat(attribute_score_q.shape[0], model.cfg.n_heads, 1)).detach().to(torch.float32).cpu().numpy()
            attribute_score_q_sum = (attribute_score_q_masked.sum(dim=(3, 4, 5))).detach().to(torch.float32).cpu().numpy()
            attribute_score_q_sum_abs_pos = (attribute_score_q_masked.sum(5).abs().sum(dim=(3, 4))).to(torch.float32).detach().cpu().numpy()
            attribute_score_q_sum_abs_exp = (attribute_score_q_masked.sum((4,5)).abs().sum(dim=(3))).detach().to(torch.float32).cpu().numpy()
            attribute_score_q_max_abs = (attribute_score_q_masked.sum(5).abs().max(4).values.sum(3)).detach().to(torch.float32).cpu().numpy()

            del attribute_score_q, attribute_score_q_masked

            attribute_score_k = (grad_k[node.layer_idx:] * (counter - clean)).unsqueeze(3).repeat(1, 1, len(exp.spans), 1, 1, 1)  # layer haed span batch seq d_model
            attribute_score_k_masked = torch.where(masks.unsqueeze(0).unsqueeze(1).repeat(attribute_score_k.shape[0], model.cfg.n_heads, 1, 1, 1, 1) == 1, attribute_score_k, 0)
            attribute_score_k_avg = (attribute_score_k_masked.sum(dim=(3, 4, 5)) / lengths.unsqueeze(0).unsqueeze(1).repeat(attribute_score_k.shape[0], model.cfg.n_heads, 1)).detach().to(torch.float32).cpu().numpy()
            attribute_score_k_sum = (attribute_score_k_masked.sum(dim=(3, 4, 5))).detach().to(torch.float32).cpu().numpy()
            attribute_score_k_sum_abs_pos = (attribute_score_k_masked.sum(5).abs().sum(dim=(3, 4))).detach().to(torch.float32).cpu().numpy()
            attribute_score_k_sum_abs_exp = (attribute_score_k_masked.sum((4,5)).abs().sum(dim=(3))).detach().to(torch.float32).cpu().numpy()
            attribute_score_k_max_abs = (attribute_score_k_masked.sum(5).abs().max(4).values.sum(3)).detach().to(torch.float32).cpu().numpy()

            del attribute_score_k, attribute_score_k_masked

            attribute_score_v = (grad_v[node.layer_idx:] * (counter - clean)).unsqueeze(3).repeat(1, 1, len(exp.spans), 1, 1, 1)  # layer haed span batch seq d_model
            attribute_score_v_masked = torch.where(masks.unsqueeze(0).unsqueeze(1).repeat(attribute_score_v.shape[0], model.cfg.n_heads, 1, 1, 1, 1) == 1, attribute_score_v, 0)
            attribute_score_v_avg = (attribute_score_v_masked.sum(dim=(3, 4, 5)) / lengths.unsqueeze(0).unsqueeze(1).repeat(attribute_score_v.shape[0], model.cfg.n_heads, 1)).detach().to(torch.float32).cpu().numpy()
            attribute_score_v_sum = (attribute_score_v_masked.sum(dim=(3, 4, 5))).detach().to(torch.float32).cpu().numpy()
            attribute_score_v_sum_abs_pos = (attribute_score_v_masked.sum(5).abs().sum(dim=(3,4))).detach().to(torch.float32).cpu().numpy()
            attribute_score_v_sum_abs_exp = (attribute_score_v_masked.sum((4,5)).abs().sum(dim=(3))).detach().to(torch.float32).cpu().numpy()
            attribute_score_v_max_abs = (attribute_score_v_masked.sum(5).abs().max(4).values.sum(3)).detach().to(torch.float32).cpu().numpy()

            del attribute_score_v, attribute_score_v_masked

            edge = Edge(upstream_layer_idx=node.layer_idx,
                        upstream_head_idx=node.head_idx,
                        upstream_type=node.node_type,
                        upstream_full_name=node.full_name,
                        downstream_layer_idx=model.cfg.n_layers-1,
                        downstream_head_idx=None,
                        downstream_type="r",
                        downstream_full_name=utils.get_act_name("resid_post", model.cfg.n_layers-1),
                        span_upstream=None,
                        span_downstream=None,
                        is_crossing=False
                        )
            results.update_score(
                edge, {"avg": attribute_score_resid_avg,
                       "sum": attribute_score_resid_sum,
                       "sum_abs_pos": attribute_score_resid_sum_abs_pos,
                       "sum_abs_exp": attribute_score_resid_sum_abs_exp,
                       "max_abs": attribute_score_resid_max_abs})

             
            
            start_layer = 0 if node.node_type == "r" else  1
                
            for l in range(start_layer, model.cfg.n_layers - node.layer_idx):
                abs_layer_idx = l + node.layer_idx
                edge = Edge(upstream_layer_idx=node.layer_idx,
                            upstream_head_idx=node.head_idx,
                            upstream_type=node.node_type,
                            upstream_full_name=node.full_name,
                            downstream_layer_idx=abs_layer_idx,
                            downstream_head_idx=None,
                            downstream_type="m",
                            downstream_full_name=utils.get_act_name(
                                "mlp_in", abs_layer_idx),
                            span_upstream=None,
                            span_downstream=None,
                            is_crossing=False
                            )
                results.update_score(
                    edge, {"avg": attribute_score_mlp_avg[l],
                           "sum": attribute_score_mlp_sum[l],
                           "sum_abs_pos": attribute_score_mlp_sum_abs_pos[l],
                           "sum_abs_exp": attribute_score_mlp_sum_abs_exp[l],
                           "max_abs": attribute_score_mlp_max_abs[l]})

                for head_idx in range(model.cfg.n_heads):
                    edge = Edge(upstream_layer_idx=node.layer_idx,
                                upstream_head_idx=node.head_idx,
                                upstream_type=node.node_type,
                                upstream_full_name=node.full_name,
                                downstream_layer_idx=abs_layer_idx,
                                downstream_head_idx=head_idx,
                                downstream_type="k",
                                downstream_full_name=utils.get_act_name(
                                    "k_input", abs_layer_idx),
                                span_upstream=None,
                                span_downstream=None,
                                is_crossing=False
                                )
                    results.update_score(edge, {"avg": attribute_score_k_avg[l, head_idx],
                                                "sum": attribute_score_k_sum[l, head_idx],
                                                "sum_abs_pos": attribute_score_k_sum_abs_pos[l, head_idx],
                                                "sum_abs_exp": attribute_score_k_sum_abs_exp[l, head_idx],
                                                "max_abs": attribute_score_k_max_abs[l, head_idx]})
                    edge = Edge(upstream_layer_idx=node.layer_idx,
                                upstream_head_idx=node.head_idx,
                                upstream_type=node.node_type,
                                upstream_full_name=node.full_name,
                                downstream_layer_idx=abs_layer_idx,
                                downstream_head_idx=head_idx,
                                downstream_type="q",
                                downstream_full_name=utils.get_act_name(
                                    "q_input",abs_layer_idx),
                                span_upstream=None,
                                span_downstream=None,
                                is_crossing=False
                                )
                    results.update_score(edge, {"avg": attribute_score_q_avg[l, head_idx],
                                                "sum": attribute_score_q_sum[l, head_idx],
                                                "sum_abs_pos": attribute_score_q_sum_abs_pos[l, head_idx],
                                                "sum_abs_exp": attribute_score_q_sum_abs_exp[l, head_idx],
                                                "max_abs": attribute_score_q_max_abs[l, head_idx]})
                    edge = Edge(upstream_layer_idx=node.layer_idx,
                                upstream_head_idx=node.head_idx,
                                upstream_type=node.node_type,
                                upstream_full_name=node.full_name,
                                downstream_layer_idx=abs_layer_idx,
                                downstream_head_idx=head_idx,
                                downstream_type="v",
                                downstream_full_name=utils.get_act_name(
                                    "v_input", abs_layer_idx),
                                span_upstream=None,
                                span_downstream=None,
                                is_crossing=False
                                )
                    results.update_score(edge, {"avg": attribute_score_v_avg[l, head_idx],
                                                "sum": attribute_score_v_sum[l, head_idx],
                                                "sum_abs_pos": attribute_score_v_sum_abs_pos[l, head_idx],
                                                "sum_abs_exp": attribute_score_v_sum_abs_exp[l, head_idx],
                                                "max_abs": attribute_score_v_max_abs[l, head_idx]})
                    
                    
            if not model.cfg.parallel_attn_mlp and node.node_type == "h":
                edge = Edge(upstream_layer_idx=node.layer_idx,
                            upstream_head_idx=node.head_idx,
                            upstream_type=node.node_type,
                            upstream_full_name=node.full_name,
                            downstream_layer_idx=node.layer_idx,
                            downstream_head_idx=None,
                            downstream_type="m",
                            downstream_full_name=utils.get_act_name("mlp_in", node.layer_idx),
                            span_upstream=None,
                            span_downstream=None,
                            is_crossing=False
                            )
                results.update_score(
                edge, {"avg": attribute_score_mlp_avg[0],
                       "sum": attribute_score_mlp_sum[0],
                       "sum_abs_pos": attribute_score_mlp_sum_abs_pos[0],
                       "sum_abs_exp": attribute_score_mlp_sum_abs_exp[0],
                       "max_abs": attribute_score_mlp_max_abs[0]})
                        
            del attribute_score_mlp_avg, attribute_score_mlp_sum, attribute_score_mlp_sum_abs_pos, attribute_score_mlp_sum_abs_exp, attribute_score_mlp_max_abs\
                , attribute_score_resid_avg, attribute_score_resid_sum, attribute_score_resid_sum_abs_pos, attribute_score_resid_sum_abs_exp, attribute_score_resid_max_abs\
                , attribute_score_q_avg, attribute_score_q_sum, attribute_score_q_sum_abs_pos, attribute_score_q_sum_abs_exp, attribute_score_q_max_abs\
                , attribute_score_k_avg, attribute_score_k_sum, attribute_score_k_sum_abs_pos, attribute_score_k_sum_abs_exp, attribute_score_k_max_abs\
                , attribute_score_v_avg, attribute_score_v_sum, attribute_score_v_sum_abs_pos, attribute_score_v_sum_abs_exp, attribute_score_v_max_abs\
                , clean, counter
            
            
        # crossing edges
        softmax = torch.nn.Softmax(dim=-1)
        attn_scale = np.sqrt(model.cfg.d_head)

        for upstream_span_idx in tqdm(range(len(exp.spans)-1)):
            len_upstream_span = indices[upstream_span_idx][1] - indices[upstream_span_idx][0]
            pos_indices = torch.arange(len_upstream_span)
            seq_start = indices[upstream_span_idx][0]
            seq_indices = seq_start + pos_indices
            for layer_idx in range(model.cfg.n_layers):
                pattern = clean_attn_pattern[layer_idx].clone()# head_index query_pos key_pos
                patched_v = clean_v_act[layer_idx].clone()# key_pos head_index d_head
                if exp.ablation_type == "mean":
                    counter = mean_activations[utils.get_act_name("v", layer_idx)].clone()
                elif exp.ablation_type == "zero":
                    counter = torch.zeros_like(clean_v_act[layer_idx],dtype=model.cfg.dtype)
                elif exp.ablation_type == "counterfactual":
                    counter = counter_cache[utils.get_act_name("v", layer_idx)].clone()
                patched_v = patched_v.repeat(len_upstream_span,1,1,1)
                for i in range(len_upstream_span):
                    patched_v[i , seq_start + i] = counter[0, seq_start + i]
                
                z_patched_v = calculate_z_scores(patched_v, pattern)

                q = clean_q_act[layer_idx].clone()  # query_pos head_index d_head
                # key_pos head_index d_head
                patched_k = clean_k_act[layer_idx].clone()
                if exp.ablation_type == "mean":
                    counter = mean_activations[utils.get_act_name(kq_pref + "k", layer_idx)].clone()
                elif exp.ablation_type == "zero":
                    counter = torch.zeros_like(clean_k_act[layer_idx],dtype=model.cfg.dtype)
                elif exp.ablation_type == "counterfactual":
                    counter = counter_cache[utils.get_act_name(kq_pref + "k", layer_idx)].clone()
                    
                patched_k = patched_k.repeat(len_upstream_span,1,1,1)
                for i in range(len_upstream_span):
                    patched_k[i, seq_start + i] = counter[0, seq_start + i]
                patched_attn_scores = calculate_attention_scores(q, patched_k, attn_scale)
                
                orig_attn_scores = clean_attn_scores[layer_idx].clone()
                patched_attn_scores = torch.where(orig_attn_scores != torch.tensor(-torch.inf), patched_attn_scores,  torch.tensor(-torch.inf))
                patched_pattern = softmax(patched_attn_scores)
                patched_pattern = torch.where(torch.isnan(patched_pattern), torch.zeros_like(patched_pattern), patched_pattern)

                z_patched_k = calculate_z_scores(clean_v_act[layer_idx],patched_pattern)
                

                
                grad = grad_heads_output[layer_idx].clone()  # seq n_heads d
                results_clean = clean_head_results_act[layer_idx].clone() # seq n_heads d


                result_patched_v = []
                for i in range(z_patched_v.shape[0]):
                    result_patched_v.append(calculate_heads_output(model, layer_idx, z_patched_v[i]))
                if len(result_patched_v) > 0:
                    result_patched_v = torch.stack(result_patched_v, dim=0)
                else:
                    result_patched_v = results_clean
                    
                
                result_patched_k = []
                for i in range(z_patched_k.shape[0]):
                    result_patched_k.append(calculate_heads_output(model, layer_idx, z_patched_k[i]))
                if len(result_patched_k) > 0:
                    result_patched_k = torch.stack(result_patched_k, dim=0)
                else:
                    result_patched_k = results_clean

                
                attribute_score_v = grad * (result_patched_v - results_clean) #batch seq n_heads d
                attribute_score_k = grad * (result_patched_k - results_clean) #batch n_heads d

                for head_idx in range(model.cfg.n_heads):
                    for downstream_span_idx in range(upstream_span_idx + 1, len(exp.spans)-1):
                        edge = Edge(upstream_layer_idx=layer_idx,
                                    upstream_head_idx=head_idx,
                                    upstream_type="v",
                                    upstream_full_name=utils.get_act_name(
                                        "v", layer_idx),
                                    downstream_layer_idx=layer_idx,
                                    downstream_head_idx=head_idx,
                                    downstream_type='h',
                                    downstream_full_name=utils.get_act_name(
                                        "result", layer_idx),
                                    span_upstream=exp.spans[upstream_span_idx],
                                    span_downstream=exp.spans[downstream_span_idx],
                                    is_crossing=True)
                        if indices[downstream_span_idx][1] - indices[downstream_span_idx][0] > 0:
                            attribute_score = attribute_score_v[:,indices[downstream_span_idx][0]:indices[downstream_span_idx][1], head_idx]
                            scores_v = {
                                "avg": attribute_score.sum(-1).mean().detach().to(torch.float32).cpu().numpy(),
                                "sum": attribute_score.sum().detach().to(torch.float32).cpu().numpy(),
                                "sum_abs_pos": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                                "sum_abs_exp": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                                "max_abs": 0
                            }
                        else:
                            scores_v = {
                                "avg": 0,
                                "sum": 0,
                                "sum_abs_pos": 0,
                                "sum_abs_exp": 0,
                                "max_abs": 0
                            }
                        results.update_score(edge, scores_v)
                        edge = Edge(upstream_layer_idx=layer_idx,
                                    upstream_head_idx=head_idx,
                                    upstream_type="k",
                                    upstream_full_name=utils.get_act_name(
                                        "k", layer_idx),
                                    downstream_layer_idx=layer_idx,
                                    downstream_head_idx=head_idx,
                                    downstream_type='h',
                                    downstream_full_name=utils.get_act_name(
                                        "result", layer_idx),
                                    span_upstream=exp.spans[upstream_span_idx],
                                    span_downstream=exp.spans[downstream_span_idx],
                                    is_crossing=True)
                        if indices[downstream_span_idx][1] - indices[downstream_span_idx][0] > 0:
                            attribute_score = attribute_score_k[:,indices[downstream_span_idx][0]:indices[downstream_span_idx][1], head_idx]
                            scores_k = {
                                "avg": attribute_score.sum(-1).mean().detach().to(torch.float32).cpu().numpy(),
                                "sum": attribute_score.sum().detach().to(torch.float32).cpu().numpy(),
                                "sum_abs_pos": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                                "sum_abs_exp": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                                "max_abs": 0
                            }
                        else:
                            scores_k = {
                                "avg": 0,
                                "sum": 0,
                                "sum_abs_pos": 0,
                                "sum_abs_exp": 0,
                                "max_abs": 0
                            }
                        results.update_score(edge, scores_k)
                    
                    downstream_span_idx = upstream_span_idx
                    edge = Edge(upstream_layer_idx=layer_idx,
                                    upstream_head_idx=head_idx,
                                    upstream_type="v",
                                    upstream_full_name=utils.get_act_name(
                                        "v", layer_idx),
                                    downstream_layer_idx=layer_idx,
                                    downstream_head_idx=head_idx,
                                    downstream_type='h',
                                    downstream_full_name=utils.get_act_name(
                                        "result", layer_idx),
                                    span_upstream=exp.spans[upstream_span_idx],
                                    span_downstream=exp.spans[downstream_span_idx],
                                    is_crossing=True)
                    attribute_score = attribute_score_v[:,indices[downstream_span_idx][0]:indices[downstream_span_idx][1], head_idx]
                    attribute_score = [attribute_score[i,i:] for i in range(attribute_score.shape[1])]
                    if len(attribute_score) > 0:
                        attribute_score = torch.cat(attribute_score, dim=0)
                        scores_v = {
                            "avg": attribute_score.sum(-1).mean().detach().to(torch.float32).cpu().numpy(),
                            "sum": attribute_score.sum().detach().to(torch.float32).cpu().numpy(),
                            "sum_abs_pos": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                            "sum_abs_exp": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                            "max_abs": 0
                        }
                    else:
                        scores_v = {
                        "avg": 0,
                        "sum": 0,
                        "sum_abs_pos": 0,
                        "sum_abs_exp": 0,
                        "max_abs": 0
                    }
                    results.update_score(edge, scores_v)
                    edge = Edge(upstream_layer_idx=layer_idx,
                                upstream_head_idx=head_idx,
                                upstream_type="k",
                                upstream_full_name=utils.get_act_name(
                                    "k", layer_idx),
                                downstream_layer_idx=layer_idx,
                                downstream_head_idx=head_idx,
                                downstream_type='h',
                                downstream_full_name=utils.get_act_name(
                                    "result", layer_idx),
                                span_upstream=exp.spans[upstream_span_idx],
                                span_downstream=exp.spans[downstream_span_idx],
                                is_crossing=True)
                    attribute_score = attribute_score_k[:,indices[downstream_span_idx][0]:indices[downstream_span_idx][1], head_idx]
                    attribute_score = [attribute_score[i,i:] for i in range(attribute_score.shape[1])]
                    if len(attribute_score) > 0:
                        attribute_score = torch.cat(attribute_score, dim=0)
                        scores_k = {
                            "avg": attribute_score.sum(-1).mean().detach().to(torch.float32).cpu().numpy(),
                            "sum": attribute_score.sum().detach().to(torch.float32).cpu().numpy(),
                            "sum_abs_pos": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                            "sum_abs_exp": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                            "max_abs": 0
                        }
                    else:
                        scores_k = {
                            "avg": np.array([0]),
                            "sum": np.array([0]),
                            "sum_abs_pos": np.array([0]),
                            "sum_abs_exp": np.array([0]),
                            "max_abs": np.array([0])
                        }
                    results.update_score(edge, scores_k)

        for downstream_span_idx in tqdm(range(len(exp.spans)-1)):
            for layer_idx in range(model.cfg.n_layers):

                k = clean_k_act[layer_idx].clone()  # key_pos head_index d_head
                if exp.ablation_type == "mean":
                    counter_q = mean_activations[utils.get_act_name(kq_pref + "q", layer_idx)].clone()
                elif exp.ablation_type == "zero":
                    counter_q = torch.zeros_like(clean_q_act[layer_idx])
                elif exp.ablation_type == "counterfactual":
                    counter_q = counter_cache[utils.get_act_name(kq_pref + "q", layer_idx)].clone()

                counter_attn_scores = calculate_attention_scores(counter_q, k, attn_scale)
                
                orig_attn_scores = clean_attn_scores[layer_idx].clone()
                counter_attn_scores = torch.where(
                    orig_attn_scores != torch.tensor(-torch.inf), counter_attn_scores,  torch.tensor(-torch.inf))

                for upstream_span_idx in range(downstream_span_idx):
                    len_upstream_span = indices[upstream_span_idx][1] - indices[upstream_span_idx][0]
                    pos_indices = torch.arange(len_upstream_span)
                    seq_start = indices[upstream_span_idx][0]
                    seq_indices = seq_start + pos_indices
                    patched_attn_scores = orig_attn_scores.detach().clone() # n_heads seq seq
                    patched_attn_scores = patched_attn_scores.repeat(len_upstream_span, 1, 1, 1)
                    for i in range(len_upstream_span):
                        patched_attn_scores[i,:,indices[downstream_span_idx][0]: indices[downstream_span_idx][1], seq_start + i] = counter_attn_scores[0, :,indices[downstream_span_idx][0]: indices[downstream_span_idx][1], seq_start + i]
                
                    patched_pattern = softmax(patched_attn_scores)
                    patched_pattern = torch.where(torch.isnan(
                        patched_pattern), torch.zeros_like(patched_pattern), patched_pattern)
                    z_patched_q = calculate_z_scores(clean_v_act[layer_idx], patched_pattern)
                    
                    grad = grad_heads_output[layer_idx].clone()  # seq n_heads d
                    results_clean = clean_head_results_act[layer_idx].clone()
                    
                    result_patched_q = []
                    for i in range(z_patched_q.shape[0]):
                        result_patched_q.append(calculate_heads_output(model, layer_idx, z_patched_q[i]))
                    if len(result_patched_q) > 0:
                        result_patched_q = torch.stack(result_patched_q, dim=0)
                    else:
                        result_patched_q = results_clean
                    
                    attribute_score_q = grad * (result_patched_q - results_clean)

                    for head_idx in range(model.cfg.n_heads):

                        edge = Edge(upstream_layer_idx=layer_idx,
                                    upstream_head_idx=head_idx,
                                    upstream_type="q",
                                    upstream_full_name=utils.get_act_name(
                                        "q", layer_idx),
                                    downstream_layer_idx=layer_idx,
                                    downstream_head_idx=head_idx,
                                    downstream_type='h',
                                    downstream_full_name=utils.get_act_name(
                                        "result", layer_idx),
                                    span_upstream=exp.spans[upstream_span_idx],
                                    span_downstream=exp.spans[downstream_span_idx],
                                    is_crossing=True)
                        if indices[downstream_span_idx][1] - indices[downstream_span_idx][0] > 0:
                            attribute_score = attribute_score_q[:,indices[downstream_span_idx][0]:indices[downstream_span_idx][1], head_idx]
                            scores_q = {
                                "avg": attribute_score.sum(-1).mean().detach().to(torch.float32).cpu().numpy(),
                                "sum": attribute_score.sum().detach().to(torch.float32).cpu().numpy(),
                                "sum_abs_pos": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                                "sum_abs_exp": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                                "max_abs": 0
                            }
                        else:
                            scores_q = {
                                "avg": np.array([0]),
                                "sum": np.array([0]),
                                "sum_abs_pos": np.array([0]),
                                "sum_abs_exp": np.array([0]),
                                "max_abs": np.array([0])
                            }
                        results.update_score(edge, scores_q)

                upstream_span_idx = downstream_span_idx
                len_upstream_span = indices[upstream_span_idx][1] - indices[upstream_span_idx][0]
                pos_indices = torch.arange(len_upstream_span)
                seq_start = indices[upstream_span_idx][0]
                seq_indices = seq_start + pos_indices
                patched_attn_scores = orig_attn_scores.detach().clone() # n_heads seq seq
                patched_attn_scores = patched_attn_scores.repeat(len_upstream_span, 1, 1, 1)
                for i in range(len_upstream_span):
                    patched_attn_scores[i,:,indices[downstream_span_idx][0]: indices[downstream_span_idx][1], seq_start + i] = counter_attn_scores[0,:,indices[downstream_span_idx][0]: indices[downstream_span_idx][1], seq_start + i]
                
                patched_pattern = softmax(patched_attn_scores)
                patched_pattern = torch.where(torch.isnan(
                    patched_pattern), torch.zeros_like(patched_pattern), patched_pattern)
                z_patched_q = calculate_z_scores(clean_v_act[layer_idx], patched_pattern)

                grad = grad_heads_output[layer_idx].clone()  # seq n_heads d
                results_clean = clean_head_results_act[layer_idx].clone()
                
                result_patched_q = []
                for i in range(z_patched_q.shape[0]):
                    result_patched_q.append(calculate_heads_output(model, layer_idx, z_patched_q[i]))
                if len(result_patched_q) > 0:
                    result_patched_q = torch.stack(result_patched_q, dim=0)
                else:
                    result_patched_q = results_clean

                attribute_score_q = grad * (result_patched_q - results_clean)

                for head_idx in range(model.cfg.n_heads):

                    edge = Edge(upstream_layer_idx=layer_idx,
                                upstream_head_idx=head_idx,
                                upstream_type="q",
                                upstream_full_name=utils.get_act_name(
                                    "q", layer_idx),
                                downstream_layer_idx=layer_idx,
                                downstream_head_idx=head_idx,
                                downstream_type='h',
                                downstream_full_name=utils.get_act_name(
                                    "result", layer_idx),
                                span_upstream=exp.spans[upstream_span_idx],
                                span_downstream=exp.spans[downstream_span_idx],
                                is_crossing=True)
                    attribute_score = attribute_score_q[:,indices[downstream_span_idx][0]:indices[downstream_span_idx][1], head_idx]
                    attribute_score = [attribute_score[i,i:] for i in range(attribute_score.shape[1])]
                    if len(attribute_score) > 0:
                        attribute_score = torch.cat(attribute_score, dim=0)
                        scores_q = {
                            "avg": attribute_score.sum(-1).mean().detach().to(torch.float32).cpu().numpy(),
                            "sum": attribute_score.sum().detach().to(torch.float32).cpu().numpy(),
                            "sum_abs_pos": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                            "sum_abs_exp": attribute_score.sum(-1).abs().sum().detach().to(torch.float32).cpu().numpy(),
                            "max_abs": 0
                        }
                    else:
                        scores_q = {
                            "avg": np.array([0]),
                            "sum": np.array([0]),
                            "sum_abs_pos": np.array([0]),
                            "sum_abs_exp": np.array([0]),
                            "max_abs": np.array([0])
                        }
                    results.update_score(edge, scores_q)
        del clean_cache, counter_cache,  clean_grad_cache, clean_head_act, grad, clean_head_results_act,\
            masks, lengths, indices,\
            grad_mlp, grad_resid, grad_heads_output,  clean_mlp_act, clean_resid_pre_act, attribute_score_q, attribute_score_v, attribute_score_k,\
                clean_v_act, clean_k_act, clean_q_act, clean_attn_pattern, clean_attn_scores, grad_q, grad_k, grad_v, counter_head_act, counter_mlp_act, counter_resid_pre_act
        if exp.ablation_type == "mean":
            del mean_activations
            
        torch.cuda.empty_cache()

    # Get the average scores for each edge
    results.get_average_scores()
    
    with open(save_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    

    ap = argparse.ArgumentParser()
    ap.add_argument("-at", "--ablation_type", required=True, type=str, choices=["counterfactual", "mean", "zero"],
                    help="ablation type")
    
    ap.add_argument("-e", "--exp", required=True, type=str,choices=["wino_bias", "greater_than", "ioi"],
                    help="exp name")
    
    ap.add_argument("-cl", "--clean", required=True, type=str,
                    help="clean ds file path")
    
    ap.add_argument("-co", "--counter", required=False, type=str,
                    help="counterfactual ds file path")
    
    ap.add_argument("-sp", "--spans", nargs='+', required=True, type=str,
                    help="spans names. should be in the order of the spans")
    
    ap.add_argument("-m", "--model", required=True, type=str,choices=["gpt2", "meta-llama/Meta-Llama-3-8B"],
                    help="model name as in TrnasformerLens") 
    
    ap.add_argument("-p", "--save_path", required=True, type=str,
                    help="save path")

    ap.add_argument("-s", "--seed", default=42, type=int, help="seed")
    
    ap.add_argument("-ds", "--dataset_size", required=True, type=int, help="dataset size")
    
    args = vars(ap.parse_args())

    model_name = args['model'].split("/")[-1]
    model_path = args['model']

    if "length" != args['spans'][-1]:
        raise ValueError("last span should be 'length'")
        

    print("ablation type:", args['ablation_type'])
    if "wino_bias" in args["exp"]:
        exp = WinoBias(exp_name=args["exp"],
                       model_name=model_name,
                       model_path=model_path,
                       ablation_type=args['ablation_type'],
                       clean_dataset_path=args['clean'],
                       counter_dataset_path=args['counter'],
                       spans=args['spans'],
                       metric=logit_diff,
                       seed=args['seed'])
    elif "greater_than" in args["exp"]:
        exp = GreaterThan(exp_name=args["exp"],
                        model_name=model_name,
                        model_path=model_path,
                         ablation_type=args['ablation_type'],
                         clean_dataset_path=args['clean'],
                         counter_dataset_path=args['counter'],
                        spans=args['spans'],
                         metric=prob_diff,
                         seed=args['seed'])
    elif "ioi" in args["exp"]:
         exp = IOI(exp_name=args["exp"],
                        model_name=model_name,
                        model_path=model_path,
                         ablation_type=args['ablation_type'],
                         clean_dataset_path=args['clean'],
                         counter_dataset_path=args['counter'],
                        spans=args['spans'],
                         metric=logit_diff,
                         seed=args['seed'])
    
    
    position_aware_edge_attribution_patching(exp=exp,
                              dataset_size=args['dataset_size'],
                              save_path=args["save_path"])

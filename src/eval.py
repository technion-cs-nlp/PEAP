import sys

from transformer_lens import HookedTransformer

import torch

import pandas as pd

from typing import  Literal, List

from dataclasses import dataclass

import numpy as np

import pickle

import random
from pos_aware_edge_attribution_patching import  Experiament, WinoBias, GreaterThan, IOI, logit_diff, prob_diff, EAPResults, Edge, EdgeScore, Node
from eval_utils import  find_abstract_circuit_size_k , full_computation_graph_size, find_abstract_circuit_size_k_reversed,run_edges_with_mean_ablation, calculate_prior_performance, calculate_model_performance
import argparse
import os
from dataclasses import dataclass



@dataclass()
class EvalResults:
    exp: Experiament
    n_examples: int
    n_ablation_size: int
    top_k: List[int]
    top_k_used: List[int]
    circuit_size: List[float]
    diff_list = List[float]
    acc_list = List[float]
    circuit_prediction_list = List[str]
    model_prediction_list = List[str]
    test_df: pd.DataFrame
    model_logit_diff: float
    prior_logit_diff: float
    model_num_edges: int
    model_diff_point_num_edges: int

    """
    A class to store and manage evaluation for the full circuit discovery pipline.

    Attributes:
        exp (Experiament): The experiment configuration object
        n_examples (int): Number of examples evaluated
        n_ablation_size (int): Size of ablation samples used
        top_k (List[int]): List of k values used for circuit disocvry
        top_k_used (List[int]): Actual k values achieved for each circuit discovery (because not all k values are possible)
        circuit_size (List[float]): Size of circuit graph at each k value
        diff_list (List[float]): logit/prob difference of the circuit at each k value
        acc_list (List[float]): accuracy of the circuit at each k value
        circuit_prediction_list (List[str]): circuit top prediction at each k value
        model_prediction_list (List[str]): Original model predictions at each k value
        test_df (pd.DataFrame): Test dataset used for evaluation
        model_logit_diff (float): Baseline model logit difference
        prior_logit_diff (float): Prior performance baseline
        model_num_edges (int): Total number of edges in full model
        model_diff_point_num_edges (int): Number of edges for the first token to be different from the counterfactual prompt
    """
    
    def __init__(self, exp, n_examples, n_ablation_size, top_k, top_k_used, circuit_size, diff_list, acc_list, circuit_prediction_list, model_prediction_list ,test_df, model_logit_diff, prior_logit_diff, model_num_edges, model_diff_point_num_edges):
        self.exp = exp
        self.n_examples = n_examples
        self.n_ablation_size = n_ablation_size
        self.top_k = top_k
        self.top_k_used = top_k_used
        self.mean_circuit_size = circuit_size
        self.diff_list = diff_list
        self.acc_list = acc_list
        self.circuit_prediction_list = circuit_prediction_list
        self.model_prediction_list = model_prediction_list
        self.test_df = test_df
        self.model_logit_diff = model_logit_diff
        self.prior_logit_diff = prior_logit_diff
        self.model_num_edges = model_num_edges
        self.model_diff_point_num_edges = model_diff_point_num_edges
    
    
def run_faithfulness(eval_size: int, peap_results_path: str, save_path: str, sum_span_scores:bool, exp: Experiament, top_k: List[int], graph_with_pos: bool, ablation_size: int, search_type: Literal["abs","max","min"], is_reversed: bool, patch_q: bool) -> None:
    """
    Run the full circuit discovery pipline.
    1. Find the circuit at size k
    2. Calculate it faithfulness

    Args:
        eval_size (int): Number of examples to evaluate
        peap_results_path (str): Path to saved circuit results
        sum_span_scores (bool): Whether to sum scores across spans
        exp (Experiament): Experiment configuration object
        top_k (List[int]): What circuit size to evaluate
        graph_with_pos (bool): Whether to discover positioanl circuit or non-positional circuit
        ablation_size (int): Number of samples for ablation
        search_type (Literal["abs","max","min"]): Type of circuit discovery strategy to use
        is_reversed (bool): Whether to reverse the circuit direction
        patch_q (bool): Whether to patch query vectors

    Returns:
        None: Results are printed and saved rather than returned
    """
    print("eval_size:",eval_size)
    print("peap_results_path:",peap_results_path)
    print("exp:",exp)
    print("top_k:",top_k)
    print("graph_with_pos:",graph_with_pos)
    print("ablation_size:",ablation_size)
    print("search_type:",search_type)
    print("is_reversed:",is_reversed)
    print("sum_span_scores",sum_span_scores)
    print("patch_q",patch_q)
    
    dtype = "bf16" if "Llama" in exp.model_path else "fp32"
    model = HookedTransformer.from_pretrained(
        exp.model_path,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=dtype
    )
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.set_ungroup_grouped_query_attention(True)


    torch.manual_seed(exp.seed)
    torch.cuda.manual_seed(exp.seed)
    np.random.seed(exp.seed)
    random.seed(exp.seed)
    
    
    with open(peap_results_path, 'rb') as f:
        full_results = pickle.load(f)
    
    test_df = exp.get_df_for_eval(n_examples=eval_size)
    print("test_df shape:",test_df.shape)
    
    model_edges, model_nodes = full_computation_graph_size(model=model,exp=exp,df=test_df,use_point_of_diff=False)
    model_edges_diff_point, model_nodes_diff_point = full_computation_graph_size(model=model,exp=exp,df=test_df, use_point_of_diff=True)
    model_performance = calculate_model_performance(model=model,exp=exp,test_df=test_df)
    prior_metric = calculate_prior_performance(model=model,test_df=test_df,exp=exp, sample_ablation_size=ablation_size)
    
    all_diff_list = []
    all_acc_list = []
    all_mean_circuit_size_list = []
    all_circuit_prediction_list = []
    all_model_prediction_list = []
    top_k_used = []
    print("top_k:", top_k)
    span_type = "sum_score" if sum_span_scores else "avg_score"
    for k in top_k:
        print(k)
        if is_reversed:
            circuit = find_abstract_circuit_size_k_reversed(model=model, exp=exp, span_type=span_type, full_results=full_results, top_k=k, search_type=search_type, with_pos=graph_with_pos)
        else:
            circuit = find_abstract_circuit_size_k(model=model, exp=exp, span_type=span_type, full_results=full_results, top_k=k, search_type=search_type, with_pos=graph_with_pos)
        print(len(circuit.nodes))
        if len(circuit.edges) == 0:
            continue
        top_k_used.append(k)

        diff_list, acc_list, mean_circuit_size_list, circuit_prediction_list, model_prediction_list = run_edges_with_mean_ablation(model=model,test_df=test_df,abstract_circuit=circuit,exp=exp, with_pos=graph_with_pos ,sample_ablation_size=ablation_size, patch_q=patch_q)
        
        all_mean_circuit_size_list.append(mean_circuit_size_list)
        all_diff_list.append(diff_list)
        all_acc_list.append(acc_list)
        all_circuit_prediction_list.append(circuit_prediction_list)
        all_model_prediction_list.append(model_prediction_list)

    
    
        eval_results = EvalResults(exp=exp,
                                n_examples=eval_size,
                                n_ablation_size=ablation_size,
                                top_k=top_k,
                                top_k_used=top_k_used,
                                circuit_size=all_mean_circuit_size_list,
                                diff_list=all_diff_list,
                                acc_list=all_acc_list,
                                circuit_prediction_list=all_circuit_prediction_list,
                                model_prediction_list=all_model_prediction_list,
                                test_df=test_df,
                                model_logit_diff=model_performance,
                                prior_logit_diff=prior_metric,
                                model_num_edges=model_edges,
                                model_diff_point_num_edges=model_edges_diff_point)
        
        
                

        with open(save_path, 'wb') as handle:
            pickle.dump(eval_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-at", "--ablation_type", required=True, type=str, choices=["counterfactual", "mean", "zero"],
                    help="ablation type")
    ap.add_argument("-e", "--exp", required=True, type=str,choices=["wino_bias", "greater_than", "ioi"],
                    help="exp name")
    ap.add_argument("-cl", "--clean", required=True, type=str,
                    help="name of the clean df file")
    ap.add_argument("-co", "--counter", required=False, type=str,
                    help="name of the counterfactual df file")
    ap.add_argument("-sp", "--save_path", required=True, type=str,
                    help="path to save results, should be a pickle file")
    ap.add_argument("-s", "--seed", required=True, type=int,
                    help="dataset seed")
    ap.add_argument("-m", "--model", required=True, type=str,
                    help="model name")
    ap.add_argument("-tk", "--topk", nargs='+', required=True, type=int,
                    help="top_k")
    ap.add_argument("-spn", "--spans", nargs='+', required=True, type=str,
                    help="the spans names")
    ap.add_argument("-n", "--n_examples", required=True, type=int,
                    help="number of examples to evaluate")
    ap.add_argument("-as", "--ablation_size", required=True, type=int,
                    help="number of examples to use for mean ablation")
    ap.add_argument("-p", "--peap_results_path", required=True, type=str,
                    help="path to the peap results")
    ap.add_argument("-gpos", "--graph_with_pos", action='store_true')
    ap.add_argument("-r", "--reversed", action='store_true')
    ap.add_argument("-st", "--search_type", required=True, type=str, choices=["abs","max","min"],
                help="search type")
    ap.add_argument("-ss", "--sum_scores", action='store_true', help="if to sum scores at the span. If not, use the avg score")
    ap.add_argument("-dpq", "--d_patch_q", action='store_false', help="if to patch the query vectors")

    args = vars(ap.parse_args())
    slurm_job_id = os.getenv('SLURM_JOB_ID')

    model_name = args['model'].split("/")[-1]
    model_path = args['model']
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
    
    run_faithfulness(
            eval_size=args['n_examples'],
            peap_results_path=args["peap_results_path"],
            sum_span_scores=args['sum_scores'],
            exp=exp,
            top_k=args['topk'],
            graph_with_pos=args['graph_with_pos'],
            ablation_size=args['ablation_size'],
            search_type=args['search_type'],
            is_reversed=args['reversed'],
            patch_q=args["d_patch_q"],
            save_path=args["save_path"])

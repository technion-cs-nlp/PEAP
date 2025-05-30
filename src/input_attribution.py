import sys

from transformer_lens import HookedTransformer, ActivationCache, utils

from tqdm import tqdm
import torch

import einops
import pandas as pd
from functools import partial

from typing import Callable, Tuple, Literal, Dict, Optional, List, Union, Set

from tqdm import tqdm

from dataclasses import dataclass
from collections import defaultdict ,OrderedDict
from copy import copy
import numpy as np

from transformer_lens.evals import IOIDataset, ioi_eval
from src.pos_aware_edge_attribution_patching import  Experiament, WinoBias, GreaterThan, IOI

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from statistics import mean, stdev

from anthropic import Anthropic
from typing import Optional

from datasets import load_dataset
from exp import Experiment



def logit_diff(logits: torch.Tensor, correct_answer: str, wrong_answer: str, model: HookedTransformer) -> torch.Tensor:
    """
        Return average logit difference between correct and incorrect answers
    """
    # Get logits diff
    correct_token = model.to_tokens(correct_answer, prepend_bos=False)[0,0]
    wrong_token = model.to_tokens(wrong_answer, prepend_bos=False)[0,0]

    logit_correct = logits[0, -1, correct_token]
    logit_wrong = logits[0, -1, wrong_token]
    diff = logit_correct - logit_wrong
    return diff



def prob_diff(logits: torch.Tensor, correct_answer, wrong_answer, model: HookedTransformer) -> torch.Tensor:
    year_indices = torch.tensor([model.tokenizer(f'{year:02d}').input_ids[0] for year in range(100)])
    probs = torch.softmax(logits[:,-1], dim=-1)[:, year_indices]

    prob_diff =  probs[:,correct_answer + 1 :].sum() - probs[:,: correct_answer + 1].sum()

    return prob_diff


@torch.no_grad()
def get_mean_activations(model: HookedTransformer):
    
    df = load_dataset("monology/pile-uncopyrighted", data_files="test.jsonl.zst")
    df = df["train"]["text"][:64]
    tokens =  torch.tensor([model.tokenizer(x)["input_ids"][:30] for x in df])
    
    model.reset_hooks()
    mean_activations = {}
    def forward_cache_hook(act, hook):
        mean_activations[hook.name] = act[:,:30].detach().mean((0)).unsqueeze(0) #seq d
    
    nodes_names = ("mlp_out", "result","resid_pre")
    def nodes(name): return name.endswith(nodes_names)
    model.add_hook(nodes, forward_cache_hook, "fwd")
    model(tokens)
    model.reset_hooks()
    print("mean activation")
    mean_activations = ActivationCache(mean_activations, model)
    
    return mean_activations


def get_important_positions(scores: List[List[float]],
                            prompt: List[List[str]],
                            save_path: str,
                            num_std_dev: float = 0,
                            use_softmax: bool = False,
                            temperature: float = 1):
    """
    Get important token positions based on attribution scores.

    Args:
        scores (List[List[float]]): List of attribution scores for each position in each prompt
        prompt (List[List[str]]): List of tokenized prompts
        num_std_dev (float, optional): Number of standard deviations above mean to consider important. 
            Only used when use_softmax=False. Defaults to 0.
        use_softmax (bool, optional): Whether to use softmax to normalize scores. Defaults to False.
        temperature (float, optional): Temperature parameter for softmax. Higher values make distribution
            more uniform. Only used when use_softmax=True. Defaults to 1.

    Returns:
        Tuple[List[List[str]], List[List[Tuple[str, int]]]]: 
            - List of tokenized prompts
            - List of (token, importance) tuples for each prompt where importance is 0 or 1
    """
    tokens, masks = [], []
    for i in range(len(scores)):
        num_tokens = len(scores[i])
        if use_softmax:
            score = torch.nn.functional.softmax(torch.tensor(scores[i])/temperature, dim=0).tolist()
            mask = [1 if p > 1/num_tokens else 0 for p in score]
        else:
            score = scores[i]
            mean_s = mean(score)
            std_s = stdev(score)
            mask = [1 if p > mean_s + num_std_dev * std_s else 0 for p in score ]
        #mask[-1] = 1
        temp = [mask[j] for j in range(len(prompt[i]))]
        mask = [(prompt[i][j],mask[j]) for j in range(len(prompt[i]))]
        tokens.append(prompt[i])
        masks.append(mask)


        plt.grid(False)
        
        data = np.array([temp])
        for j in range(data.shape[1]):
            plt.text(j, 0, f'{data[0,j]}', ha='center', va='center', color='black',fontsize=8)
        plt.imshow(data, cmap='Blues',vmin=0, vmax=1)
        plt.xticks(ticks=np.arange(len(prompt[i])), labels=prompt[i], rotation=90)
        plt.savefig(save_path)
        plt.show()
    return tokens, masks


def get_activations(model: HookedTransformer, clean_prompt,  counter_prompt, correct_answer, wrong_answer, exp: Experiament):
    """
    Get model activations for clean and counterfactual inputs, along with gradients.

    Args:
        model (HookedTransformer): The transformer model to get activations from
        clean_prompt (str): The original input prompt
        counter_prompt (str): The counterfactual/corrupted input prompt
        correct_answer (str): The correct answer token
        wrong_answer (str): The incorrect answer token 
        exp (Experiament): Experiment object containing metric function

    Returns:
        Tuple[ActivationCache, ActivationCache, ActivationCache]:
            - Cache of activations from clean input
            - Cache of activations from counterfactual input  
            - Cache of gradients from clean input
    """
    model.reset_hooks()
    clean_cache = {}
    clean_grad_cache = {}
    counterfactual_cache = {}

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()
        
    def node_outgoing_filter(name): return name.endswith(("resid_pre", "hook_result", "hook_mlp_out","q_input","v_input","hook_v","hook_q"))

    model.add_hook(node_outgoing_filter, forward_cache_hook, "fwd")

    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()

    def edge_back_filter(name): return name.endswith(("resid_pre", "hook_result", "hook_mlp_out"))

    model.add_hook(edge_back_filter, backward_cache_hook, "bwd")

    logits = model(clean_prompt, return_type="logits")  # batch seq d

    diff = exp.metric(logits=logits, correct_answer=correct_answer ,wrong_answer=wrong_answer, model=model)
    diff.backward()

    # cache the activations of the corrupted inputs
    model.reset_hooks()

    def forward_corrupted_cache_hook(act, hook):
        counterfactual_cache[hook.name] = act.detach()

    model.add_hook(node_outgoing_filter, forward_corrupted_cache_hook, "fwd")
    with torch.no_grad():
        model(counter_prompt)
    model.reset_hooks()

    clean_cache = ActivationCache(clean_cache, model)
    counterfactual_cache = ActivationCache(counterfactual_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, counterfactual_cache, clean_grad_cache



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



def grad_embedding_attribution(model: HookedTransformer,
                          exp: Experiament,
                          clean_prompts: List[str],
                          correct_tokens: List[str],
                          wrong_tokens: List[str]
                          ) -> None:

    """
    Calculates gradient-based attribution scores for token embeddings.

    This function computes attribution scores by:
    1. Running forward pass through the model
    2. Computing gradients 
    3. Calculating attribution scores as norm of (activation * gradient) for each position

    Args:
        model (str): Name/path of the model to load
        exp (Experiment): Experiment configuration object
        clean_prompts (List[str]): List of input prompts to analyze
        correct_tokens (List[str]): List of correct answer tokens
        wrong_tokens (List[str]): List of incorrect answer tokens

    Returns:
        Tuple[List[List[float]], List[List[str]]]: Returns two lists:
            - List of attribution scores for each position in each prompt
            - List of tokenized prompts
    """
    model = HookedTransformer.from_pretrained(
       model,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="bf16" if "Llama" in model else "float32")
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
        
    attribute_scores = []
    prompts = []
    for index, example in enumerate(clean_prompts):
        clean_cache = {}
        clean_grad_cache = {}
        
        def forward_cache_hook(act, hook):
            clean_cache[hook.name] = act.detach()
        def backward_cache_hook(act, hook):
            clean_grad_cache[hook.name] = act.detach()
            
        tokens = model.to_str_tokens(example)
        prompts.append(tokens)
        model.reset_hooks()
        model.add_hook(utils.get_act_name("resid_pre",0), forward_cache_hook, "fwd")  
        model.add_hook(utils.get_act_name("resid_pre",0), backward_cache_hook, "bwd") 
        logits = model(example, return_type="logits")  # batch seq d
        diff = exp.metric(logits=logits, correct_answer=correct_tokens[index] ,wrong_answer=wrong_tokens[index], model=model)
        diff.backward()
        model.reset_hooks()
        
        clean_resid_pre_act = clean_cache['blocks.0.hook_resid_pre'].clone()
        grad_resid = clean_grad_cache['blocks.0.hook_resid_pre'].clone()

        score = torch.norm(clean_resid_pre_act * grad_resid, dim=-1)[0].tolist()
        
        attribute_scores.append(score)
        del clean_resid_pre_act, grad_resid, score
    return attribute_scores, prompts


def node_attribution_patching(model: HookedTransformer,
                              exp: Experiament,
                              clean_prompts: List[str],
                              correct_tokens: List[str],
                              wrong_tokens: List[str]
                              ) -> None:
    
    """
    Performs node attribution patching to analyze the contribution of different model components.


    Args:
        model (HookedTransformer): The transformer model to analyze
        exp (Experiment): Experiment configuration object containing task-specific logic
        clean_prompts (List[str]): List of unmodified input prompts
        correct_tokens (List[str]): List of correct completion tokens for each prompt
        wrong_tokens (List[str]): List of incorrect completion tokens for each prompt

    Returns:
        None: Results are computed and stored but not returned directly

    """
    model = HookedTransformer.from_pretrained(
       model,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="bf16" if "Llama" in model else "float32"
    )
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.set_ungroup_grouped_query_attention(True)

    nodes = create_nodes(model)
    mean_act = get_mean_activations(model)

    attribute_scores = []
    prompts = []
    for index, example in enumerate(clean_prompts):
        print
        prompt_tok = model.to_str_tokens(example)
        len_prompt = len(prompt_tok)
        prompts.append(prompt_tok)
        
        clean_cache, counter_cache, clean_grad_cache = get_activations(model, clean_prompt=example, counter_prompt=example, correct_answer=correct_tokens[index], wrong_answer=wrong_tokens[index],exp=exp)
    
        # activations
        # heads output
        clean_head_act = split_layers_and_heads(clean_cache.stack_head_results(), model=model).clone()  # layer head batch seq d_model
        clean_head_results_act = split_layers_and_heads_for_heads_results(clean_cache.stack_head_results(), model=model).clone()  # layer batch seq head d_model
        # mlp output
        clean_mlp_act = clean_cache.stack_activation(activation_name="mlp_out").clone()  # layer batch seq d_model
        # residual 0 output
        clean_resid_pre_act = clean_cache['blocks.0.hook_resid_pre'].clone()  # batch seq d_model
    

        counter_head_act = split_layers_and_heads(mean_act.stack_head_results(), model=model).clone()[:,:,:,:len_prompt]   # layer head batch seq d_model
        # mlp output
        counter_mlp_act = mean_act.stack_activation(activation_name="mlp_out").clone()[:,:,:len_prompt]   # layer batch seq d_model
        # residual 0 output
        counter_resid_pre_act = mean_act['blocks.0.hook_resid_pre'].clone()[:,:len_prompt]   # batch seq d_model
    
        grad_mlp = clean_grad_cache.stack_activation(activation_name="mlp_out").clone()  # layer batch seq dim
        grad_resid = clean_grad_cache.stack_activation(activation_name="resid_pre").clone()  # layer batch seq dim
        grad_heads_output = clean_grad_cache.stack_activation(activation_name="result").clone()  # layer batch seq head d_head
    
        
        #attribute_score = torch.zeros(clean_mlp_act.shape[-2])
        attribute_score = []
        for i in tqdm(range(len(nodes))):
            node = nodes[i]
            if node.node_type == "m":
                clean = clean_mlp_act[node.layer_idx,0].clone().cpu()
                counter = counter_mlp_act[node.layer_idx,0].clone().cpu()
                grad = grad_mlp[node.layer_idx,0].cpu()  
            elif node.node_type == "h":
                clean = clean_head_act[node.layer_idx, node.head_idx,0].clone().cpu()    #  batch seq d
                counter = counter_head_act[node.layer_idx, node.head_idx,0].clone().cpu()
                grad = grad_heads_output[node.layer_idx,0,:,node.head_idx].clone().cpu()   
            else:
                clean = clean_resid_pre_act[0].clone().cpu()    # batch seq d
                counter = counter_resid_pre_act[0].clone().cpu()    # batch seq d
                grad = grad_resid[0,0].clone().cpu()  
            attribute_score.append((grad * (counter - clean)).sum(-1).abs())
        attribute_score = torch.stack(attribute_score)
        attribute_score = torch.sum(attribute_score,dim=0)
        attribute_scores.append(attribute_score.tolist())
        del clean_cache, counter_cache, clean_grad_cache, clean_head_act, clean_head_results_act, clean_mlp_act, clean_resid_pre_act, counter_head_act, counter_mlp_act, grad_mlp, grad_resid,
    return attribute_scores, prompts

def input_attribution(model: HookedTransformer,
                              exp: Experiament,
                              clean_prompts: List[str],
                              correct_tokens: List[str],
                              wrong_tokens: List[str]
                              ) -> None:
    
    """
    Performs input attribution analysis by measuring how patching each input token with mean activations affects model behavior.

    Args:
        model (HookedTransformer): The transformer model to analyze
        exp (Experiment): Experiment class containing metric for measuring model behavior
        clean_prompts (List[str]): List of input prompts to analyze
        correct_tokens (List[str]): List of correct completion tokens for each prompt
        wrong_tokens (List[str]): List of incorrect completion tokens for each prompt

    Returns:
        Tuple[List[List[float]], List[List[str]]]: 
            - List of attribution scores for each token in each prompt
            - List of tokenized prompts
    """
    model = HookedTransformer.from_pretrained(
       model,
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
        fold_ln=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    dtype="bf16" if "Llama" in model else "float32"
    )
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    nodes = create_nodes(model)
    mean_act = get_mean_activations(model)
    
    def forward_cache_hook(act, hook, index):
        act[:,index] = mean_act[hook.name][:,index]
        return act
        
    attribute_scores = []
    prompts = []
    for index, example in enumerate(clean_prompts):
        tokens = model.to_str_tokens(example)
        prompts.append(tokens)
        scores = []
        for token_index in range(len(tokens)):
            model.reset_hooks()
            model.add_hook(utils.get_act_name("resid_pre", 0), partial(forward_cache_hook, index=token_index), "fwd")
            logits = model(example, return_type="logits")  # batch seq d
            diff = exp.metric(logits=logits, correct_answer=correct_tokens[index] ,wrong_answer=wrong_tokens[index], model=model)
            model.reset_hooks()
            logits = model(example, return_type="logits")  # batch seq d
            orig_diff = exp.metric(logits=logits, correct_answer=correct_tokens[index] ,wrong_answer=wrong_tokens[index], model=model)
            scores.append(abs(diff-orig_diff).item())
        attribute_scores.append(scores)
    return attribute_scores, prompts


def create_inputs(model_name: str, clean_data_path:str, counter_data_path:str, exp_name: str, num_exp: int, seed: int = 5) -> Tuple[List[str], List[str], List[str], Experiment]:
    if exp_name == "ioi_BABA":
        exp = IOI(exp_name="ioi_BABA",
                    model_name=model_name,
                    model_path=model_name,
                    ablation_type="counterfactual",
                    clean_dataset_path=clean_data_path,
                    counter_dataset_path=counter_data_path,
                    spans=["prefix","S1","S1+1","IO","action1","S2","action2","to","length"],
                    metric=logit_diff,
                    seed=seed)



        df = pd.read_csv(clean_data_path)
        df = df[df["top_answer"] == df["IO_token"]].sample(num_exp,random_state=seed)
        clean_prompts = df["prompt"].to_list()
        correct_tokens =  df["IO_token"].to_list()
        wrong_tokens =  df["S1_token"].to_list()
    elif exp_name == "ioi_ABBA":
        exp = IOI(exp_name=exp_name,
                    model_name=model_name,
                    model_path=model_name,
                    ablation_type="counterfactual",
                    clean_dataset_path=clean_data_path,
                    counter_dataset_path=counter_data_path,
                    spans=["prefix","IO","and","S1", "S1+1", "action1","S2","action2","to","length"],
                    metric=logit_diff,
                    seed=seed)



        df = pd.read_csv(clean_data_path)
        df = df[df["top_answer"] == df["IO_token"]].sample(num_exp,random_state=seed)
        clean_prompts = df["prompt"].to_list()
        correct_tokens =  df["IO_token"].to_list()
        wrong_tokens =  df["S1_token"].to_list()
    elif exp_name == "gt" or exp_name == "greater_than":
        exp = GreaterThan(exp_name="greater_than",
                        model_name=model_name,
                        model_path=model_name,
                        ablation_type="counterfactual",
                        clean_dataset_path=clean_data_path,
                        counter_dataset_path=counter_data_path,
                        spans=["The", "NOUN", "lasted", "from", "the_1", "year_1", "XX1", "YY", "to", "the_2", "year_2", "XX2", "length"],
                        metric=prob_diff,
                        seed=seed)
        
        df = pd.read_csv(clean_data_path)
        clean_prompts = df["clean"].to_list()
        correct_tokens =  df["label"].to_list()
        wrong_tokens =  df["label"].to_list()
    else:
        if "first" in exp_name:
            exp= WinoBias(exp_name="wino_bias",
                        model_name="Meta-Llama-3-8B",
                        model_path="meta-llama/Meta-Llama-3-8B",
                        ablation_type=logit_diff,
                        clean_dataset_path=clean_data_path,
                        counter_dataset_path=counter_data_path,
                        spans=["correct_profession","interaction","wrong_profession","conjunction","first_pronoun","circumstances","dot","The","pronoun","second_pronoun","refers","to","the","length"],
                        metric=logit_diff,
                        seed=seed)
        else:
            exp= WinoBias(exp_name="wino_bias",
                        model_name="Meta-Llama-3-8B",
                        model_path="meta-llama/Meta-Llama-3-8B",
                        ablation_type=logit_diff,
                        clean_dataset_path=clean_data_path,
                        counter_dataset_path=counter_data_path,
                        spans=["wrong_profession","interaction","correct_profession","conjunction","first_pronoun","circumstances","dot","The","pronoun","second_pronoun","refers","to","the","length"],
                        metric=logit_diff,
                        seed=seed)
        df = pd.read_csv(clean_data_path)
        df = df[df['top_answer'] == df['wrong_token']].sample(num_exp,random_state=seed)
        clean_prompts = df["prompt"].to_list()
        correct_tokens =  df["correct_token"].to_list()
        wrong_tokens =  df["wrong_token"].to_list()

    return clean_prompts, correct_tokens, wrong_tokens, exp
            



def run_attribution_experiments(attribution_method: Literal["input_atr", "node_atr", "embed_grad"], clean_prompts: List[str], correct_tokens: List[str], wrong_tokens: List[str], exp: Experiament, model: HookedTransformer, temperature: float):
    
    """
    Runs attribution analysis on a model using specified method.

    Args:
        attribution_method (str): The method to use for attribution analysis
        clean_prompts (List[str]): List of input prompts to analyze
        correct_tokens (List[str]): List of correct completion tokens for each prompt
        wrong_tokens (List[str]): List of incorrect completion tokens for each prompt
        exp (Experiment): Experiment class containing metric for measuring model behavior
        model (HookedTransformer): The transformer model to analyze
        temperature (float): Temperature for softmax
    """
    
    print("*********grad attribution*********")
    use_softmax=True
    if attribution_method == "input_atr":
        attr, prompt = input_attribution(model=model, exp=exp, clean_prompts=clean_prompts, correct_tokens=correct_tokens, wrong_tokens=wrong_tokens)
        use_softmax=False
    elif attribution_method == "node_atr":
        attr, prompt = node_attribution_patching(model=model, exp=exp, clean_prompts=clean_prompts, correct_tokens=correct_tokens, wrong_tokens=wrong_tokens)
        use_softmax=False
    elif attribution_method == "embed_grad":
        attr, prompt = grad_embedding_attribution(model=model, exp=exp, clean_prompts=clean_prompts, correct_tokens=correct_tokens, wrong_tokens=wrong_tokens)
    for i in range(len(attr)):
            attr[i] = attr[i][1:]
            prompt[i] = prompt[i][1:]    
            plt.grid(False)
            
            data = np.array([[round(x,2) for x in attr[i]]])
            for j in range(data.shape[1]):
                plt.text(j, 0, f'{data[0,j]}', ha='center', va='center', color='black',fontsize=8)
            plt.imshow(data, cmap='Blues',vmin=0, vmax=max(attr[i]))
            plt.xticks(ticks=np.arange(len(prompt[i])), labels=prompt[i], rotation=90)
            plt.show()
    return get_important_positions(attr,prompt,use_softmax=use_softmax,temperature=temperature)




import sys

from transformer_lens import HookedTransformer, ActivationCache
import torch
import pandas as pd
import random
from typing import Dict, List, Union
import numpy as np
from abc import abstractmethod
from typing import Callable, Tuple, Dict, Optional, List, Union
from functools import partial

import os


class Experiament:
    exp_name: str
    model_name: str
    model_path: str
    ablation_type: str
    clean_dataset_path: str
    counter_dataset_path: Union[str, List[str]]
    spans: List[str]
    metric: Callable[[torch.Tensor, pd.DataFrame, HookedTransformer], torch.Tensor]
    seed: int
    
    def __init__(self, exp_name, model_name, model_path, ablation_type, clean_dataset_path, counter_dataset_path, spans,  metric, seed):
        self.exp_name = exp_name
        self.model_name = model_name
        self.model_path = model_path
        self.ablation_type = ablation_type
        self.clean_dataset_path = clean_dataset_path
        self.counter_dataset_path = counter_dataset_path
        self.spans = spans
        self.metric = metric
        self.seed=seed
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    
    def get_span_masks(self, row: pd.DataFrame, model: HookedTransformer, dtype: str):
        
        masks = torch.zeros((len(self.spans), 1,row['length'], model.cfg.d_model), dtype=dtype).to("cuda")
        lengths = torch.zeros(len(self.spans),dtype=dtype).to("cuda")
        indices = []
        

        for span_idx in range(len(self.spans[:-1])):
            masks[span_idx, 0, row[self.spans[span_idx]]:row[self.spans[span_idx + 1]]] = 1
            lengths[span_idx] = max(row[self.spans[span_idx + 1]] - row[self.spans[span_idx]],1)
            indices.append((row[self.spans[span_idx]], row[self.spans[span_idx + 1]]))
       
        masks[-1, 0, row[self.spans[0]]:row[self.spans[-1]]] = 1
        lengths[-1] = row[self.spans[-1]]
        indices.append((row[self.spans[0]], row[self.spans[-1]]))
        return masks, lengths, indices
    
    
    @abstractmethod
    def create_datasets(self)->Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        pass
    
    @abstractmethod
    def get_mean_activations(self, model: HookedTransformer, row: pd.DataFrame, index: int, sample_ablation_size: int)->Tuple[ActivationCache, pd.DataFrame]:
        pass
    
    @abstractmethod
    def get_df_for_eval(self,n_examples)->pd.DataFrame:
        pass
        
class GreaterThan(Experiament):
    
    def __init__(self, exp_name,  model_name, model_path,  ablation_type, clean_dataset_path, counter_dataset_path, spans,  metric, seed):
        super().__init__(exp_name=exp_name, model_name=model_name, model_path=model_path, ablation_type=ablation_type, clean_dataset_path=clean_dataset_path, counter_dataset_path=counter_dataset_path , spans=spans, metric=metric, seed=seed)
        
    
    def create_datasets(self)->Union[Tuple[pd.DataFrame, pd.DataFrame],Tuple[pd.DataFrame, None]]:      

        df_clean = pd.read_csv(self.clean_dataset_path) 
        df_clean = df_clean[(df_clean["split"] == "circuit")]
        df_counter = None
        df_counter = pd.read_csv(self.counter_dataset_path).loc[df_clean.index]
        
        return df_clean, df_counter
    
    
    def get_mean_activations(self, model: HookedTransformer, row: pd.DataFrame, index:int,  seed:int, sample_ablation_size:int=16) -> ActivationCache:
        label = row["label"]
        df_counter = pd.read_csv(self.counter_dataset_path)
        if sample_ablation_size > 1: # mean ablation    
            df_counter = df_counter[(df_counter["split"] == "ablation") & (~(df_counter["label"]==label))]
            df_counter = df_counter.sample(n=sample_ablation_size, random_state=seed)
        else: # sample ablation
            df_counter = pd.DataFrame([df_counter.loc[index]])
            
       
        model.reset_hooks()
        mean_activations = {}
        def forward_cache_hook(act, hook):
            mean_activations[hook.name] = torch.unsqueeze(act.detach().clone().mean(0), 0) # batch seq d

        kq_pref = "rot_" if model.cfg.positional_embedding_type == "rotary" else ""
        nodes_names = ("mlp_out", "result", "hook_v", f"hook_{kq_pref}q", f"hook_{kq_pref}k", "resid_pre", "attn_out","pattern")
        def nodes(name): return name.endswith(nodes_names)
        model.add_hook(nodes, forward_cache_hook, "fwd")
        model(df_counter["prompt"].to_list())
        model.reset_hooks()
        mean_activations = ActivationCache(mean_activations, model)

        return mean_activations, df_counter

           
    
    def get_df_for_eval(self,n_examples)->pd.DataFrame:
        df = pd.read_csv(self.clean_dataset_path)
        df = df[df["split"] == "eval"]
        return  df.sample(n=min(n_examples, df.shape[0]), random_state=self.seed)
    
class IOI(Experiament):
    
    def __init__(self, exp_name, model_name, model_path, ablation_type, clean_dataset_path, counter_dataset_path, spans,  metric, seed):
        super().__init__(exp_name=exp_name, model_name=model_name, model_path=model_path, ablation_type=ablation_type, clean_dataset_path=clean_dataset_path, counter_dataset_path=counter_dataset_path,spans=spans,  metric=metric, seed=seed)
       
        
    def create_datasets(self)->Union[Tuple[pd.DataFrame, pd.DataFrame],Tuple[pd.DataFrame, None]]:
        
        df_clean = pd.read_csv(self.clean_dataset_path) 
        df_clean = df_clean[(df_clean["split"] == "circuit") & (df_clean["top_answer"] == df_clean["correct_token"])]
        df_counter = None
        df_counter = pd.read_csv(self.counter_dataset_path).loc[df_clean.index]
        
        return df_clean, df_counter
    
    
    def get_mean_activations(self, model: HookedTransformer, row, index, seed, sample_ablation_size=16):
        label = row["label"]
        df = pd.read_csv(self.counter_dataset_path)
        if sample_ablation_size > 1: # mean ablation
            df = df[(df["split"] == "ablation") & (~(df["label"] == label)) & (df["prompt_id"] == row["prompt_id"])]
            df = df[~(df['S1_token'].str.contains(row["S1_token"], case=False, na=False) |
                    df['S2_token'].str.contains(row["S1_token"], case=False, na=False) |
                    df['IO_token'].str.contains(row["S1_token"], case=False, na=False) |
                    df['S1_token'].str.contains(row["IO_token"], case=False, na=False) |
                    df['S2_token'].str.contains(row["IO_token"], case=False, na=False) |
                    df['IO_token'].str.contains(row["IO_token"], case=False, na=False))]
            df = df.sample(n=sample_ablation_size, random_state=seed)
        else: # sample ablation
            df = pd.DataFrame([df.loc[index]])

        model.reset_hooks()
        mean_activations = {}
        
        def forward_cache_hook(act, hook):
            if hook.name not in mean_activations:
                mean_activations[hook.name] = act.detach() / sample_ablation_size # batch seq d
            else:
                mean_activations[hook.name] += act.detach() / sample_ablation_size # batch seq d
            
        kq_pref = "rot_" if model.cfg.positional_embedding_type == "rotary" else ""
        nodes_names = ("mlp_out", "result", "hook_v", f"hook_{kq_pref}q", f"hook_{kq_pref}k", "resid_pre", "attn_out","pattern")
        def nodes(name): return name.endswith(nodes_names)
        model.add_hook(nodes, forward_cache_hook, "fwd")

        for index, row in df.iterrows():
            model(row["prompt"])
        model.reset_hooks()
        
        mean_activations = ActivationCache(
            mean_activations, model)
        torch.cuda.empty_cache()

        return mean_activations, df
        
    
            
    def get_df_for_eval(self,n_examples)->pd.DataFrame:
        df =pd.read_csv(self.clean_dataset_path)
        df = df[(df["split"] == "eval") & (df["top_answer"] == df["correct_token"])]
        return  df.sample(n=min(n_examples,df.shape[0]), random_state=self.seed)
        
class WinoBias(Experiament):
    
    def __init__(self, exp_name, model_name, model_path, ablation_type, clean_dataset_path, counter_dataset_path, spans,  metric, seed):
        super().__init__(exp_name=exp_name,model_name=model_name, model_path=model_path, ablation_type=ablation_type, clean_dataset_path=clean_dataset_path, counter_dataset_path=counter_dataset_path, spans=spans, metric=metric, seed=seed)
        

    
    def create_datasets(self)->Union[Tuple[pd.DataFrame, pd.DataFrame],Tuple[pd.DataFrame, None]]:
        
        df_clean = pd.read_csv(self.clean_dataset_path)
        
        df_clean = df_clean[(df_clean["split"] == "circuit") &
                                     (df_clean['top_answer'] == df_clean['wrong_token'])]

        df_counter = pd.read_csv(self.counter_dataset_path)
        df_counter = df_counter.loc[df_clean.index]
        
        return df_clean, df_counter
    
    @torch.no_grad()
    def get_mean_activations(self, model: HookedTransformer, row, index,  seed, sample_ablation_size=16):
        wrong_token = row["wrong_token"]
        correct_token = row["correct_token"]
        prompt_id = row["id"]
        
        
        
        df_list = []
        for file in self.counter_dataset_path:
            df = pd.read_csv(file)
            df = df[~(df['correct_token'].str.contains(wrong_token, case=False, na=False) |
                df['wrong_token'].str.contains(wrong_token, case=False, na=False) |
                    df['correct_token'].str.contains(correct_token, case=False, na=False) | 
                        df['wrong_token'].str.contains(correct_token, case=False, na=False))]
            df = df[(df["id"] == prompt_id) & (df["split"] == "ablation")].sample(n=sample_ablation_size//4, random_state=seed)
            df_list.append(df)
        
        df = pd.concat(df_list, ignore_index=True)
        mean_activations = {}

        def forward_cache_hook(act, hook):
            mean_activations[hook.name] = torch.unsqueeze(act.detach().mean(0), 0) # batch seq d
        
        kq_pref = "rot_" if model.cfg.positional_embedding_type == "rotary" else ""
        nodes_names = ("mlp_out", "result", "hook_v", f"hook_{kq_pref}q", f"hook_{kq_pref}k", "resid_pre", "attn_out","pattern")
        def nodes(name): return name.endswith(nodes_names)
        model.add_hook(nodes, forward_cache_hook, "fwd")
        model(df["prompt"].to_list())
        model.reset_hooks()
        mean_activations = ActivationCache(
            mean_activations, model)
        torch.cuda.empty_cache()

        return mean_activations, df
    
    
    def get_df_for_eval(self,n_examples)->pd.DataFrame:
        
        df = pd.read_csv(self.clean_dataset_path)
        df = df[(df["split"] == "evel") & 
                (df['top_answer'] == df['wrong_token'])]
        return df.sample(n=min(n_examples, df.shape[0]), random_state=self.seed)


def logit_diff(logits: torch.Tensor, row: pd.DataFrame, model: HookedTransformer) -> torch.Tensor:
    """
        Return average logit difference between correct and incorrect answers
    """
    # Get logits diff

    correct_token = model.to_tokens(row["correct_token"], prepend_bos=False)[0,0].item()
    wrong_token = model.to_tokens(row["wrong_token"], prepend_bos=False)[0,0].item()
    
    assert  row["correct_token"] == model.tokenizer.decode([correct_token])
    assert  row["wrong_token"] == model.tokenizer.decode([wrong_token])

    if len(logits.shape) == 3:
        logit_correct = logits[0, -1, correct_token]
        logit_wrong = logits[0, -1, wrong_token]
    else:
        logit_correct = logits[-1, correct_token]
        logit_wrong = logits[-1, wrong_token]
    diff = logit_correct - logit_wrong
    return diff


def prob_diff(logits: torch.Tensor, row: pd.DataFrame, model: HookedTransformer) -> torch.Tensor:
    year_indices = torch.tensor([model.tokenizer(f'{year:02d}').input_ids[0] for year in range(100)])
    correct_year = row["label"]
    probs = torch.softmax(logits[:,-1], dim=-1)[:, year_indices]

    prob_diff = probs[:,correct_year + 1 :].sum() - probs[:,: correct_year + 1].sum()

    return prob_diff

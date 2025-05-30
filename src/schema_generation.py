import argparse
import sys

from transformer_lens import HookedTransformer, ActivationCache

from tqdm import tqdm
import torch

import pandas as pd

from typing import Callable, Tuple, Literal, Dict, Optional, List, Union, Set, Any
from typing import NamedTuple
from tqdm import tqdm
import json
import os
import copy
from dataclasses import dataclass
from collections import defaultdict ,OrderedDict
from copy import copy
import copy
import ast
from openai import OpenAI


from pos_aware_edge_attribution_patching import  Experiament, WinoBias, GreaterThan, IOI, logit_diff, prob_diff

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from collections import defaultdict
from datetime import datetime

from anthropic import Anthropic
import anthropic
from typing import Optional
import backoff 
import re
from input_attribution import run_attribution_experiments



SYSTEM_TEXT  = f""""

You are a precise AI researcher, and your goal is to understand how a language model processes a dataset by analyzing its behavior across different segments of prompts. 
To do this, you need to divide all prompts in the dataset into spans, where each span represents a meaningful part of the sentence.
The aim is to split the prompts in the dataset systematically, allowing you to analyze the relationships between various parts of the sentence and support different types of model analysis.
"""

INSTRUCTIONS = f"""

Task:###
Your task is to define a schema - a structure that defines how to split all the examples in the dataset into meaningful spans.
The schema defines how to divide all examples into the same set of spans! Even though the examples do not have the exact same tokens, they share a similar structure. 
All parts of each prompt should be assigned to a span, meaning the schema must provide a complete division of every prompt.
###

Input Format:###
1.Tokens: A list of tokens representing the example. Your task is to find a schema that defines how to divide this list into meaningful spans.
2.Mask: A list of pairs in the format [(token, value)], where a value of 1 indicates that the token is important and should be placed in its own span, separated from other tokens.
###

instructions:###
1.Use syntactic and semantic rules to create a schema that defines how to divide all the examples in the dataset into meaningful spans.
2.Use the Masks to create additional spans for any token marked as significant (value = 1). Each of these tokens should be placed in its own span. Note: Apply this rule only if a specific token or token role is marked as important across many examples.
3.If you think certain parts or tokens are crucial for the model's processing of the prompt, assign them to a separate span to highlight their importance."
4.The spans should provide a complete division of the prompt, ensuring that every token is assigned to a span, and the spans should reflect the chronological structure of the prompt.
5.The examples may vary, so you must define a schema that is not tailored to any specific example but can be applied consistently across all examples.
###

Goal:###
Given a set of examples, your goal is to define a schema - a structure that divides all examples into the same set of sub-spans.
####


Return a JSON object describing the schema.
Each key in the dictionary should represent a span title (1-3 words), and the corresponding value should describe the tokens or segments assigned to that span.
Provide a brief description of each span’s role based on syntax, semantics, or another relevant aspect, but do not reference the Mask in the description.
Provide a variety of examples in the descriptions to clarify the scope of each span.
Assign a descriptive and unique span title (1-3 words) to each span. Avoid mentioning the Mask in the title (e.g., 'Significant Token').

```json
    {{
        "title":"description and examples",
    }}
```
###

"""

INSTRUCTIONS_NO_MASK = f"""

Task:###
Your task is to define a schema - a structure that defines how to split all the examples in the dataset into meaningful spans.
The schema defines how to divide all examples into the same set of spans! Even though the examples do not have the exact same tokens, they share a similar structure. 
All parts of each prompt should be assigned to a span, meaning the schema must provide a complete division of every prompt.
###

Input Format:###
Tokens: A list of tokens representing the example. Your task is to find a schema that defines how to divide this list into meaningful spans.
###

instructions:###
1.Use syntactic and semantic rules to create a schema that defines how to divide all the examples in the dataset into meaningful spans.
2.If you think certain parts or tokens are crucial for the model's processing of the prompt, assign them to a separate span to highlight their importance."
3.The spans should provide a complete division of the prompt, ensuring that every token is assigned to a span, and the spans should reflect the chronological structure of the prompt.
4.The examples may vary, so you must define a schema that is not tailored to any specific example but can be applied consistently across all examples.
###

Goal:###
Given a set of examples, your goal is to define a schema - a structure that divides all examples into the same set of sub-spans.
####


Return a JSON object describing the schema.
Each key in the dictionary should represent a span title (1-3 words), and the corresponding value should describe the tokens or segments assigned to that span.
Provide a brief description of each span’s role based on syntax, semantics, or another relevant aspect, but do not reference the Mask in the description.
Provide a variety of examples in the descriptions to clarify the scope of each span.
Assign a descriptive and unique span title (1-3 words) to each span.

```json
    {{
        "title":"description and examples",
    }}
```
###

"""



GUIDELINES = """
Guidelines: ###
1.Ensure the schema assigns each part in the prompt to a span.

2.The final token in every prompt must always be placed in its own span. The final token is alway placed in the last span.

3.Any token marked with a value of 1 is considered important and should be placed in its own span in the schema.

4.Create spans that can be applied to all the examples. Ensure spans are relevant and consistent across the entire dataset.
###
"""

GUIDELINES_NO_MASK = """
Guidelines: ###
1.Ensure the schema assigns each part in the prompt to a span.

2.The final token in every prompt must always be placed in its own span. The final token is alway placed in the last span.

3.Create spans that can be applied to all the examples. Ensure spans are relevant and consistent across the entire dataset.
###
"""


FINAL_INSTRUCTIONS = f"""

You previously received the following instructions:###

Task:###
Your task is to define a schema - a structure that defines how to split all the examples in the dataset into meaningful spans.
The schema defines how to divide all examples into the same set of spans! Even though the examples do not have the exact same tokens, they share a similar structure. 
All parts of each prompt should be assigned to a span, meaning the schema must provide a complete division of every prompt.
###

Input Format:###
1.Tokens: A list of tokens representing the example. Your task is to find a schema that defines how to divide this list into meaningful spans.
2.Mask: A list of pairs in the format [(token, value)], where a value of 1 indicates that the token is important and should be placed in its own span, separated from other tokens.
###

instructions:###
1.Use syntactic and semantic rules to create a schema that defines how to divide all the examples in the dataset into meaningful spans.
2.Use the Masks to create additional spans for any token marked as significant (value = 1). Each of these tokens should be placed in its own span. Note: Apply this rule only if a specific token or token role is marked as important across many examples.
3.If you think certain parts or tokens are crucial for the model's processing of the prompt, assign them to a separate span to highlight their importance."
4.The spans should provide a complete division of the prompt, ensuring that every token is assigned to a span, and the spans should reflect the chronological structure of the prompt.
5.The examples may vary, so you must define a schema that is not tailored to any specific example but can be applied consistently across all examples.
###

Goal:###
Given a set of examples, your goal is to define a schema - a structure that divides all examples into the same set of sub-spans.
####


#### End of previous instructions####

New Task: Creating a Unified schema
You have provided three distinct schemas. Based on your earlier responses, your task is to develop a new schema that applies universally to all examples.

Step-by-step instructions:###
Step 1: Review your previous answers and identify the common spans and elements they share.
Step 2: Establish a unified new schema, ensuring every part of the prompt is included in a span and the segmentation remains uniform across all cases.
Step 3: Return a JSON object describing the Scehma. Each key in the dictionary should represent a span title (1-3 words), and the corresponding value should describe the tokens or segments assigned to that span.Provide a brief description of each span’s role based on syntax, semantics, or another relevant aspect, but do not reference the Mask in the description. Provide a variety of examples in the descriptions to clarify the scope of each span. Assign a descriptive and unique span title (1-3 words) to each span. Avoid mentioning the Mask in the title (e.g., 'Significant Token').

```json
    {{
        "title":"description and examples",
    }}
```
###

"""

FINAL_INSTRUCTIONS_NO_MASK = f"""

You previously received the following instructions:###

Task:###
Your task is to define a schema - a structure that defines how to split all the examples in the dataset into meaningful spans.
The schema defines how to divide all examples into the same set of spans! Even though the examples do not have the exact same tokens, they share a similar structure. 
All parts of each prompt should be assigned to a span, meaning the schema must provide a complete division of every prompt.
###

Input Format:###
Tokens: A list of tokens representing the example. Your task is to find a schema that defines how to divide this list into meaningful spans.
###

instructions:###
1.Use syntactic and semantic rules to create a schema that defines how to divide all the examples in the dataset into meaningful spans.
2.If you think certain parts or tokens are crucial for the model's processing of the prompt, assign them to a separate span to highlight their importance."
3.The spans should provide a complete division of the prompt, ensuring that every token is assigned to a span, and the spans should reflect the chronological structure of the prompt.
4.The examples may vary, so you must define a schema that is not tailored to any specific example but can be applied consistently across all examples.
###

Goal:###
Given a set of examples, your goal is to define a schema - a structure that divides all examples into the same set of sub-spans.
####


#### End of previous instructions####

New Task: Creating a Unified schema
You have provided three distinct schemas. Based on your earlier responses, your task is to develop a new schema that applies universally to all examples.

Step-by-step instructions:###
Step 1: Review your previous answers and identify the common spans and elements they share.
Step 2: Establish a unified new schema, ensuring every part of the prompt is included in a span and the segmentation remains uniform across all cases.
Step 3: Return a JSON object describing the Scehma. Each key in the dictionary should represent a span title (1-3 words), and the corresponding value should describe the tokens or segments assigned to that span.Provide a brief description of each span’s role based on syntax, semantics, or another relevant aspect, but do not reference the Mask in the description. Provide a variety of examples in the descriptions to clarify the scope of each span. Assign a descriptive and unique span title (1-3 words) to each span.

```json
    {{
        "title":"description and examples",
    }}
```
###

"""

FINAL_GUIDELINES =  """
1.Utilize the previous schemas to create a unified format for dividing examples into spans.

2.The last token in each prompt must always be placed in its own span, which should always be the final span.

3.Define spans that are applicable to all examples, ensuring they are relevant and consistent across the entire dataset.

4.Avoid just copying your final response. Generate a completely new schema based on all prior responses.
###
"""




def call_chatgpt(client_oai: OpenAI, messege,temperature):
    """
    Calls the ChatGPT API with the given prompt.
    """
    response = client_oai.chat.completions.create(
        model="gpt-4",
        messages=messege,
        temperature=temperature
    )
    return response.choices[0].message.content

def handle_error(e):
    """Helper function to process error messages"""
    if hasattr(e, 'status_code'):
        if e.status_code == 529:
            print(f"Received 529 error - Service is at capacity. Retrying...")
            return True  # Indicates should retry
    return False  # Don't retry other errors

@backoff.on_exception(
    backoff.expo,
    anthropic.APIError,
    max_tries=5,  # Maximum number of retries
    giveup=lambda e: not handle_error(e),  # Only retry on 529 errors
    base=2,  # Start with 2 seconds delay
    factor=2  # Double the delay after each retry
)
def call_claude(client_anthropic: Anthropic, user_message: str, temperature) -> Optional[str]:
    """
    Send a message to Claude with automatic retries for 529 errors
    """    
    try:
        message = client_anthropic.messages.create(
            model="claude-3-5-sonnet-20240620",
            #model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": user_message
            }]
        )
        return message.content[0].text
    
    except anthropic.APIError as e:
        if hasattr(e, 'status_code'):
            if e.status_code == 429:
                print("Rate limit exceeded. Please wait before making more requests.")
            elif e.status_code == 401:
                print("Authentication error. Please check your API key.")
            elif e.status_code == 400:
                print("Bad request. Please check your input parameters.")
            else:
                print(f"API Error: {str(e)} (Status code: {e.status_code})")
        else:
            print(f"API Error without status code: {str(e)}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return ""


def call_hf_model(model, tokenizer, prompt, temperature):
    
    inputs = tokenizer(prompt,return_tensors="pt")
    with torch.no_grad():
        response = model.generate(
        input_ids=inputs["input_ids"],
        #max_length=50,  # You can adjust this value based on how long you want the response
        num_return_sequences=1,  # Number of sequences to generate
        do_sample=True,  # Sampling (True) vs greedy decoding (False)
        top_k=50,  # Adjust for more controlled sampling
        top_p=0.95,  # Adjust for more controlled sampling
        temperature=temperature,  # Adjust the creativity of the output (higher is more random)
        )
        response = tokenizer.decode(response[0], skip_special_tokens=True)
    return response

class SchemaGenerator:
    def __init__(self, api_key: str, llm_name: Literal["claude", "chatGPT", "hf"], hf_model=None, tokenizer=None):
        self.api_key = api_key
        if llm_name == "claude":
            self.client_anthropic = Anthropic(api_key=self.api_key)
        elif llm_name == "chatGPT":
            self.client_oai = OpenAI(api_key=self.api_key)
        elif llm_name == "hf":
            self.model = hf_model
            self.tokenizer = tokenizer




    def extract_json_from_text(self, text:str):
        # Use a regex pattern to find all JSON objects enclosed in triple backticks
        pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'

        # Find all matches
        matches = re.findall(pattern, text)

        if matches:
            # Get the last match
            last_match = matches[-1].strip()

            # Debug: Print the extracted JSON string
            print("Extracted JSON string:")
            print(last_match)
            print("--- End of Extracted JSON string ---\n")
            
            if not last_match:
                print("The extracted JSON string is empty.")
                return None

            # Custom function to handle duplicate keys
            def rename_duplicate_keys(pairs):
                new_pairs = []
                keys_count = {}
                for key, value in pairs:
                    if key in keys_count:
                        keys_count[key] += 1
                        new_key = f"{key} {keys_count[key]}"
                    else:
                        keys_count[key] = 0
                        new_key = key
                    new_pairs.append((new_key, value))
                return dict(new_pairs)

            # Try parsing as JSON with custom object_pairs_hook
            try:
                return json.loads(last_match, object_pairs_hook=rename_duplicate_keys)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON directly: {e}")

                # Attempt to fix common issues
                # Replace single quotes with double quotes
                json_string_fixed = last_match.replace("'", '"')
                
                # Remove trailing commas
                json_string_fixed = re.sub(r',\s*([\]}])', r'\1', json_string_fixed)
                
                try:
                    return json.loads(json_string_fixed, object_pairs_hook=rename_duplicate_keys)
                except json.JSONDecodeError as e2:
                    print(f"Failed to decode JSON after fixes: {e2}")
                    # As a last resort, try parsing as a Python literal
                    try:
                        return ast.literal_eval(last_match)
                    except Exception as e3:
                        print(f"Failed to parse using ast.literal_eval: {e3}")
                        return None
        else:
            print("No JSON object found in the text.")
            return None
                


    def create_schema(self, model_name: str, model: HookedTransformer,tokenizer: AutoTokenizer,tokens: List[str], masks: List, instructions: str, guidelines: str, system_text: str, temperature: float, use_mask: bool):
        """
        Creates a schema based on token sequences and optional masks using language model assistance.
        
        Args:
            model_name (str): Name of the language model to use ('chatGPT' or other)
            model (HookedTransformer): The transformer model
            tokenizer (AutoTokenizer): Tokenizer for the model
            tokens (List[str]): List of token sequences to analyze
            masks (List): List of importance masks corresponding to tokens (optional)
            instructions (str): Instructions for schema generation
            guidelines (str): Additional guidelines for schema format
            system_text (str): System prompt for the language model
            temperature (float): Sampling temperature for generation
            use_mask (bool): Whether to include importance masks in prompt

        Returns:
            The generated schema based on the language model's response
        """
        num_examples = len(tokens)
        if use_mask:
            user_text= f"""
    ###
    {instructions}
    ###
    I will now provide you with {num_examples} pairs of Tokens and a Mask. Follow the steps carefully, and return a JSON file in the correct format. 
        """

            for i in range(num_examples):
                example = f"""

    Example {i}:###
    Tokens: {tokens[i]}
    Mask: {masks[i]}
    ###

                """
                user_text += example
        else:
            user_text= f"""
    ###
    {instructions}
    ###
    I will now provide you with {num_examples} lists of Tokens. Follow the steps carefully, and return a JSON file in the correct format. 
        """

            for i in range(num_examples):
                example = f"""

    Example {i}:###
    Tokens: {tokens[i]}
    ###

                """
                user_text += example
        
        user_text += f"""
    ###
    {guidelines}
    ###
        """

        if model_name == "chatGPT":
            messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}]
        else:
            messages = system_text + user_text

        for i in range(10):
            if model_name == "chatGPT":
                response = call_chatgpt(self.client_oai,messages,temperature)
            elif model_name == "claude":
                response = call_claude(self.client_anthropic,messages,temperature)
            elif model_name == "hf":
                response = call_hf_model(self.model,self.tokenizer,messages,temperature)
            schema = self.extract_json_from_text(response)
            if schema is not None:
                break
            if schema is None:
                print("couldent find a schema!")

        return schema, system_text + user_text, response
    
    

    def create_unified_schema(self,
                            model_name: str, 
                            model: HookedTransformer,
                            tokenizer: Any,
                            schemas: List[str],
                            error_for_next_generation: List[str],
                            instructions: str,
                            guidelines: str, 
                            system_text: str,
                            tokens: List[List[str]],
                            masks: List[List[float]],
                            temperature: float,
                            use_mask: bool) -> Tuple[Optional[Dict], str, str]:
        
        """
        Creates a unified schema by combining multiple schemas and examples.

        Args:
            model_name (str): Name of the model to use ("chatGPT", "claude", or "hf")
            model (HookedTransformer): The transformer model (only used if model_name is "hf")
            tokenizer (Any): Tokenizer for the model (only used if model_name is "hf") 
            schemas (List[str]): List of previous schema attempts
            error_for_next_generation (List[str]): List of errors from previous attempts
            instructions (str): Instructions for schema generation
            guidelines (str): Additional guidelines for schema generation
            system_text (str): System prompt text
            tokens (List[List[str]]): List of token sequences to analyze
            masks (List[List[float]]): List of importance masks for tokens
            temperature (float): Temperature for model generation
            use_mask (bool): Whether to include masks in examples

        Returns:
            Tuple[Optional[Dict], str, str]: Tuple containing:
                - The generated schema as a dictionary (or None if failed)
                - The full prompt text used
                - The raw model response
        """
        num_examples = len(tokens)
        all_examples = ""

        if use_mask:
            for i in range(num_examples):
                example = f"""

    Example {i}:###
    Tokens: {tokens[i]}
    Mask: {masks[i]}
    ###
                """
                all_examples += example
        else:
            for i in range(num_examples):
                example = f"""

    Example {i}:###
    Tokens: {tokens[i]}
    ###
                """
                all_examples += example
            
        
        user_text = f"""
        {instructions}

    I will now provide you with {len(schemas)} Schemas :
        """
        for i in range(len(schemas)):
            user_text+= f"""
    ###
    Schema {i}:###
    {schemas[i]}
    ###
                """

        user_text += f"""
    ###
    All examples: ###
    {all_examples}
    ###

    ###
    {guidelines}
    ###
        """

        if len(error_for_next_generation) > 0:
            error_message = self.generate_error_message(error_for_next_generation)
            previous_attempt = f"""Previous attempt:


    In your previous attempt to generate a unified schema, applying it to the examples led to the following errors:
    {error_message}

    Please develop a new schema that resolves these errors!"""
            user_text += previous_attempt
            print(previous_attempt)
        
        if model_name == "chatGPT":
            messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}]
        else:
            messages = system_text + user_text

        for i in range(10):
            if model_name == "chatGPT":
                response = call_chatgpt(self.client_oai,messages,temperature)
            elif model_name == "claude":
                response = call_claude(self.client_anthropic,messages,temperature)
            elif model_name == "hf":
                response = call_hf_model(self.model,self.tokenizer,messages,temperature)
            schema = self.extract_json_from_text(response)
            if schema is not None:
                break
            if schema is None:
                print("couldent find a schema!")

        return schema, system_text + user_text, response




    def apply_schema(self, model_name: str, model: Optional[Any], tokenizer: Optional[Any], tokens_list: List[List[str]], schema: Dict[str, str], have_empty_span: bool, temperature: float) -> Tuple[List[Dict[str, List[str]]], Dict[str, List[Any]]]:
        """
        Applies a schema to a prompt.

        Args:
            model_name (str): Name of the model to use ('chatGPT', 'claude', or 'hf')
            model (Optional[Any]): The HuggingFace model object if using 'hf', otherwise None
            tokenizer (Optional[Any]): The HuggingFace tokenizer if using 'hf', otherwise None 
            tokens_list (List[List[str]]): List of token sequences to apply the schema to
            schema (Dict[str, str]): The schema defining how to split tokens into spans
            have_empty_span (bool): Whether empty spans are allowed
            temperature (float): Temperature parameter for model generation

        Returns:
            Tuple[List[Dict[str, List[str]]], Dict[str, List[Any]]]: 
                - List of dictionaries mapping span names to token lists
                - Dictionary containing logs of the schema application process
        """
        # Replace 'your-api-key' with your actual OpenAI API key
        
        results = []
        all_logs = {"all_logs":[], "final_results":[]}
        num_error = 0
        for tokens in tokens_list:
            token_log = {"tokens":tokens, "apply_schema":[]}
            apply_schema = {}
            if num_error > len(tokens_list)//5:
                return results, all_logs
            request = self.create_apply_schema_prompt(tokens, schema)        
            if model_name == "chatGPT":
                messege = [{"role": "user", "content": request}]
                response = call_chatgpt(self.client_oai,messege,temperature)
            elif model_name == "claude":
                response = call_claude(self.client_anthropic,request,temperature)
            elif model_name == "hf":
                response = call_hf_model(model,tokenizer,request,temperature)
            spans = self.extract_json_from_text(response)
            is_valid, errors = self.validate_spans(tokens, schema, spans, have_empty_span)
            token_log["apply_schema"].append({"request":request, "response":response, "spans":spans, "is_valid":is_valid, "errors":errors})
            apply_schema = {}
            # If not valid, explain the mistakes and retry
            retry_count = 1
            max_retries = 3
            while not is_valid and retry_count < max_retries:
                error_message = self.generate_error_message(errors)
                request = self.create_apply_schema_prompt(tokens, schema, error_message)
                if model_name == "chatGPT":
                    messege = [{"role": "user", "content": request}]
                    response = call_chatgpt(self.client_oai,messege,temperature)
                elif model_name == "claude":
                    response = call_claude(self.client_anthropic,request,temperature)
                elif model_name == "hf":
                    response = call_hf_model(self.model,self.tokenizer,request,temperature)
                spans = self.extract_json_from_text(response)
                is_valid, errors = self.validate_spans(tokens, schema, spans,have_empty_span)
                token_log["apply_schema"].append({"request":request, "response":response, "spans":spans, "is_valid":is_valid, "errors":errors})
                
                retry_count += 1
            if not is_valid:
                num_error += 1
                print(f"Failed to get valid spans after {retry_count} attempts. Errors: {errors}")
            if is_valid or retry_count == max_retries:
                idx = 0 # add spaces
                new_schema = defaultdict(list)
                for key, span_tokens in spans.items():
                    new_schema[key] = tokens[idx: idx+len(span_tokens)]
                    idx += len(span_tokens)
                spans = new_schema
                print(json.dumps(spans, indent=4))
            results.append((spans, errors))
            all_logs["all_logs"].append(token_log)
            all_logs["final_results"].append((spans, errors))
        return results, all_logs

    def create_apply_schema_prompt(self, tokens: List[str], schema: Dict[str, str], error_message: Optional[str] = None):
        """
        Creates the prompt to send to ChatGPT.
        """
        empty_schema = {}

        for key in schema:
            empty_schema[key] = []
        error = ""
        if error_message:
            error += f"\nPrevious attempt had the following issues:\n{error_message}\nPlease correct these in your new response."
        

        prompt = f"""
    You are an assistant that splits tokens into spans based on a given schema.

    Schema:
    {schema}

    Tokens:
    {tokens}

    Please split the tokens into the spans defined by the schema. Return the spans as a JSON object where each key is a span name and the value is the list of tokens in that span.

    {error}

    Format:

    ```json
        {{"span title": []}}
    ```

    Ensure that:
    - All the spans are present.
    - Every token is assigned to a span.
    - No new spans are added.
    - Punctuation marks should be included in the spans. If no specific span is assigned to a punctuation mark, it should be grouped with the preceding token.
    - Ensure that the last token is placed exclusively in the final span.
    - The spans are in the correct order as in the schema.
    - The tokens in each span are a continuous segment of the full prompt. 
    - The tokens are kept in the same order as they appear in the original prompt. 
    - Don't remove spaces from tokens inside the list.
    - If a span has no tokens in a specific example, leave it empty, but still include the span for consistency across all examples.

    """
        return prompt


    def validate_spans(self, tokens: List[str], schema: Dict[str, str], spans: Dict[str, List[str]], have_empty_span: bool):
        """
        Validates the spans according to the specified criteria.
        """
        errors = []
        if spans is None:
            errors.append("Response is not valid JSON.")
            return False, errors
        
        schema_spans = list(schema.keys())
        response_spans = list(spans.keys())
        
        # Check if all spans are present
        if set(schema_spans) != set(response_spans):
            missing_spans = set(schema_spans) - set(response_spans)
            extra_spans = set(response_spans) - set(schema_spans)
            if missing_spans:
                errors.append(f"Missing spans: {missing_spans}")
            if extra_spans:
                errors.append(f"Extra spans added: {extra_spans}")
        # Check order of spans
        elif schema_spans != response_spans:
            errors.append("Spans are not in the correct order.")
            
        # Check if tokens in each span are continuous and cover the full prompt
        all_tokens = []
        empty_spans = []
        token_str = ''.join([x.strip() for x in tokens])
        is_continuous = True
        for span_name, span_tokens in spans.items():
            if len(span_tokens) == 0:
                empty_spans.append(span_name)
            span_str = ''.join([x.strip() for x in span_tokens])
            if span_str not in token_str:
                errors.append(f"Tokens in span '{span_name}' are not a continuous segment of the prompt.")
                is_continuous=False
            else:
                all_tokens += span_tokens
            
        
        # Check if all tokens are covered
        if len(all_tokens) < len(tokens):
            missing_tokens = [x for x in [x.strip() for x in tokens] if x not in [x.strip() for x in all_tokens]]
            errors.append(F"Not all tokens are included in the spans. Add the Following tokens {missing_tokens}")
        elif len(all_tokens) > len(tokens):
            errors.append(F"Ensure that all tokens are included in only one span.")
        new_tokens = [x for x in [x.strip() for x in all_tokens] if x not in [x.strip() for x in tokens]]
        if len(new_tokens) > 0:
            errors.append(F"You have added to the spans new tokens that are not part of the prompt. Add to the spans tokens that are in the prompt.")
        elif ' '.join([x.strip() for x in all_tokens]) != ' '.join([x.strip() for x in tokens]) and is_continuous:
            errors.append(f"The tokens are not split into the spans in chronological order. Split the tokens into spans in chronological order") 
    

        if len(empty_spans)> 0 and not have_empty_span:
            errors.append(f"The following spans are empty: {empty_spans}. Check if the tokens were split correctly or if this span type is too uncommon.") 

        if len(spans[response_spans[-1]]) > 1:
            errors.append(f"The final token is not in its own span.") 

        if len(spans[response_spans[-1]]) == 0:
            errors.append(f"The final span is empty. The final token must be in the last span")
        return len(errors) == 0, errors

    def generate_error_message(self, errors: List[str]):
        """
        Generates an error message to send back to ChatGPT.
        """
        error_message = ""
        for error in errors:
            error_message += f"- {error}\n"
        return error_message



    def schema_generation(self,
                        model_to_analyze: HookedTransformer,
                        clean_prompts: List[str],
                        correct_tokens: List[str],
                        wrong_tokens: List[str],
                        save_path: str,
                        exp: object,
                        attribution_method: str,
                        model_to_generate: str,
                        use_mask: bool,
                        temperature: float,
                            ):
        
        schema_errors = [
            "The final token is not in its own span",
            "The following spans are empty",
            "The tokens are not split into the spans in chronological order",
            "Ensure that all tokens are included in only one span",
            "Not all tokens are included in the spans"
        ]
        instructions = INSTRUCTIONS if use_mask else INSTRUCTIONS_NO_MASK
        guidelines = GUIDELINES if use_mask else GUIDELINES_NO_MASK
        system_text = SYSTEM_TEXT
        final_instructions = FINAL_INSTRUCTIONS if use_mask else FINAL_INSTRUCTIONS_NO_MASK
        final_guidelines = FINAL_GUIDELINES
        tokens, masks = run_attribution_experiments(attribution_method,clean_prompts, correct_tokens, wrong_tokens, exp, model_to_analyze,1)
        tokenizer, model = None, None
        if "Llama" in model_to_generate:
            model_name ="meta-llama/Meta-Llama-3-70B-Instruct"  # You can change this to any Huggingface model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model =  AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.bfloat16)
            model.generation_config.pad_token_id = tokenizer.eos_token_id
        set_seed(64)
        current_time = datetime.now()
        time_string = current_time.strftime("%Y%m%d_%H%M%S")
        use_mask_text = "with_mask" if use_mask else "no_mask"
        logs = {"info":{}, "rounds":[]}
        batch_size = 5
        num_batch = len(tokens)//batch_size
        schemas = []
        examples = []
        requests = []
        responses = []
        logs["info"] = {"tokens": tokens, "masks": masks, "seed":exp.seed, "batch_size": batch_size, "num_batch": num_batch, "attribution_method": attribution_method, "use_mask": use_mask, "model_to_generate": model_to_generate, "model_to_analyze": model_to_analyze, "temperature": temperature}
        error_for_next_generation = []
        final_schema = None
        for use_empty_span in [False, True]:
            for i in range(5):
                log_round = {"round": i}
                for batch in range(num_batch):
                    cur_tokens = tokens[batch_size * batch: batch_size * (batch+1)]
                    cur_mask = masks[batch_size * batch: batch_size * (batch+1)]
                    schema, request, response = self.create_schema(model_to_generate, model, tokenizer, cur_tokens, cur_mask, instructions, guidelines, system_text, temperature, use_mask)
                    log_round[f"batch_{batch}"] = {"schema": schema, "request": request, "response": response}
                    print(f"schema {batch} is Done")
                    schemas.append(schema)
                    requests.append(request)
                    responses.append(response)
        
                final_schema, request, response = self.create_unified_schema(model_to_generate, model, tokenizer, schemas,list(set(error_for_next_generation)),final_schema, final_instructions, final_guidelines, system_text, tokens, masks, temperature, use_mask)
                log_round["final_schema"] = {"schema": final_schema, "request": request, "response": response}
                print(f"Final Sechem")
                print(json.dumps(final_schema, indent=4))
                results, apply_logs = self.apply_schema(model_to_generate, model, tokenizer, tokens, final_schema, use_empty_span, temperature)
                log_round["apply_schema"] = apply_logs
                error_rate = 0
                error_for_next_generation = []
                for spans, errors in results:
                    if len(errors) > 0:
                        error_for_next_generation += [e for e in errors if any(a.lower() in e.lower() for a in schema_errors)]
                        error_rate += 1
                logs["rounds"].append(log_round)
                logs["info"]["is_valid"] = False
                logs["info"]["error_reate"] = error_rate/len(results)
                logs["info"]["final_schema"] = final_schema
                
                with open(save_path, 'w') as json_file:
                    json.dump(logs, json_file, indent=4)
                
                print(error_for_next_generation)
                print("error rate", error_rate/len(results))
                if error_rate/len(results) <= 0.2:
                    print("finish succesfully")
                    logs["info"]["is_valid"] = True
                    logs["info"]["error_reate"] = error_rate/len(results)
                    return final_schema, results
        
        return final_schema, results


    def apply_schema_to_dataset(self, model_name: str, exp: Experiament, schema_path: str, save_path: str, have_empty_span: bool = False, temperature: float = 0.7) -> None:
        """
        Applies the schema to a dataset of prompts.
        """
        model = HookedTransformer.from_pretrained(
        model_name,
            center_writing_weights=False,
            center_unembed=False,
            trust_remote_code=True,
            fold_ln=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype="bf16" if "Llama" in model_name else "float32")
        df_clean, df_counterfactual = exp.create_datasets()
        schema = json.load(open(schema_path))["info"]["final_schema"]
        df_clean = df_clean.iloc[:500]
        results_list = []

        df_eval = exp.get_df_for_eval(500)
        results_list = []
        for index, row in df_eval.iterrows():
            row_dict = row.to_dict()
            tokens =  model.to_str_tokens(row["prompt"],prepend_bos=False)
            results, apply_logs = self.apply_schema("claude", None, None, [tokens], schema, have_empty_span, temperature)
            results_list.append({"spans":results, "apply_logs": apply_logs, "row_dict": row_dict})
        
            with open(save_path, "w") as json_file:
                json.dump({"schema":schema, "seed": exp.seed, "size": len(results_list), "dataset":results_list}, json_file, indent=4)
        
        

    def split_dataset_based_on_schema(schema_path: str, save_path: str, counter_data_path: str, save_path_counter: str, exp: object) -> None:
        """split the dataset into spans based on the schema.

        This function takes a schema file containing dataset specifications and creates two datasets:
        a clean dataset and a counterfactual dataset. It handles different dataset types including
        IOI (indirect object identification) and greater-than comparison tasks.

        Args:
            schema_path (str): Path to the schema JSON file
            save_path (str): Path where the clean dataset will be saved
            save_path_counter (str): Path where the counterfactual dataset will be saved  
            exp (object): Experiment object containing dataset creation methods

        Returns:
            None: The function saves the datasets to the specified paths but does not return anything
        """
        with open(schema_path) as json_file:
            schema_dataset = json.load(json_file)
        
        df_clean, df_counterfactual = exp.create_datasets()
        
        
        
        if "eval" in save_path:
            df_temp = exp.get_df_for_eval(500)
            df_counterfactual = pd.read_csv(counter_data_path)
      
            df_counterfactual = df_counterfactual.loc[df_temp.index]
            
        schema = schema_dataset["schema"]
        schema = {"_".join(key.split()): [] for key, value in schema.items()}
        dataset_orig = {}
        dataset_counter_orig = {}
        if "ioi" in schema_path:
            dataset_orig = {"prompt": [],
                    "prompt_id": [],
                    "length": [],
                    "wrong_token": [],
                    "correct_token":[],
                    "S1_token":[],
                    "S2_token": [],
                    "IO_token": [],
                    "label":[],
                    "split": [],
                    "top_answer": [],
                    "top_answer_prob": [],
                    "correct_prob": [],
                    "wrong_prob": []}
        elif "greater_than" in schema_path:
            dataset_orig = {"prompt": [], "length": [], "label": [], "split":[]}
        else:
            dataset_orig = {"prompt": [],
                    "id": [],
                    "pair_id": [],
                    "correct_proffesion_idx":[],
                    "correct_token":[],
                    "wrong_token":[],
                    "length":[],
                    "split":[],
                    "top_answer":[],
                    "top_answer_prob":[],
                    "correct_prob":[],
                    "wrong_prob":[]}
        
        dataset_counter_orig = copy.deepcopy(dataset_orig)
        dataset_idx = 0
        n_errors = 0
        for exp in schema_dataset["dataset"]:
            is_valid = True
            for e in exp["spans"][0][1]:
                if "The following spans are empty" not in e:
                    n_errors += 1
                    is_valid = False
                    break
            if not is_valid:
                dataset_idx += 1
                continue
            idx = 1
            for key, value in exp["spans"][0][0].items():
                schema["_".join(key.split())].append(idx)
                idx += len(value)
            for key in dataset_orig:
                if "eval" in save_path:
                    dataset_orig[key].append(exp["row_dict"][key])
                else:
                    dataset_orig[key].append(exp["row_dict_clean"][key])
                dataset_counter_orig[key].append(df_counterfactual.iloc[dataset_idx][key])
            dataset_idx += 1
        print("num errors:",n_errors)
        dataset = {**dataset_orig, **schema}
        dataset_counter = {**dataset_counter_orig, **schema}
        df = pd.DataFrame(dataset)
        df_counter = pd.DataFrame(dataset_counter)
        print("saving to", save_path)
        df.to_csv(save_path, index=False)
        df_counter.to_csv(save_path_counter, index=False)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['generate_schema', 'apply_schema'], help='Command to run')
    parser.add_argument('--exp', type=str, required=True, help='Experiment type (wino_bias, greater_than, or ioi)')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--ablation_type', type=str, required=True, help='Type of ablation')
    parser.add_argument('--clean', type=str, required=True, help='Path to clean dataset')
    parser.add_argument('--counter', type=str, required=True, help='Path to counterfactual dataset')
    parser.add_argument('--spans', type=str, required=True, help='Spans configuration')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--llm_name', type=str, required=True, help='Name of LLM')

    # Arguments specific to generate_schema command
    parser.add_argument('--model_to_analyze', type=str, help='Model to analyze')
    parser.add_argument('--clean_prompts', type=str, help='Clean prompts')
    parser.add_argument('--correct_tokens', type=str, help='Correct tokens')
    parser.add_argument('--wrong_tokens', type=str, help='Wrong tokens')
    parser.add_argument('--attribution_method', type=str, help='Attribution method')
    parser.add_argument('--model_to_generate', type=str, help='Model to generate')
    parser.add_argument('--use_mask', action='store_true', help='Use mask')
    parser.add_argument('--temperature', type=float, help='Temperature')

    # Arguments specific to apply_schema command
    parser.add_argument('--schema_path', type=str, help='Path to schema')
    parser.add_argument('--have_empty_span', action='store_true', help='Allow empty spans')

    args = parser.parse_args()

    if "wino_bias" in args.exp:
        exp = WinoBias(exp_name=args.exp,
                       model_name=args.model_name,
                       model_path=args.model_path,
                       ablation_type=args.ablation_type,
                       clean_dataset_path=args.clean,
                       counter_dataset_path=args.counter,
                       spans=args.spans,
                       metric=logit_diff,
                       seed=args.seed)
    elif "greater_than" in args.exp:
        exp = GreaterThan(exp_name=args.exp,
                        model_name=args.model_name,
                        model_path=args.model_path,
                         ablation_type=args.ablation_type,
                         clean_dataset_path=args.clean,
                         counter_dataset_path=args.counter,
                        spans=args.spans,
                         metric=prob_diff,
                         seed=args.seed)
    elif "ioi" in args.exp:
         exp = IOI(exp_name=args.exp,
                        model_name=args.model_name,
                        model_path=args.model_path,
                         ablation_type=args.ablation_type,
                         clean_dataset_path=args.clean,
                         counter_dataset_path=args.counter,
                        spans=args.spans,
                         metric=logit_diff,
                         seed=args.seed)
         
    if args.command == 'generate_schema':
        schema_generator = SchemaGenerator(api_key=args.api_key, llm_name=args.llm_name)
        schema_generator.schema_generation(
            model_to_analyze=args.model_to_analyze,
            clean_prompts=args.clean_prompts,
            correct_tokens=args.correct_tokens, 
            wrong_tokens=args.wrong_tokens,
            exp=exp,  
            attribution_method=args.attribution_method,
            model_to_generate=args.model_to_generate,
            use_mask=args.use_mask,
            temperature=args.temperature
        )

    elif args.command == 'apply_schema':
        schema_generator = SchemaGenerator(api_key=args.api_key, llm_name=args.llm_name)
        schema_generator.apply_schema_to_dataset(
            model_name=args.model_name,
            exp=args.exp,
            schema_path=args.schema_path,
            have_empty_span=args.have_empty_span,
            temperature=args.temperature
        )
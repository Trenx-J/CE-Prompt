# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch

from peft.utils.integrations import gather_params_ctx

from .config import PromptTuningInit


class CEPromptEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()


        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)

        self.inference_mode = config.inference_mode
        self.prompt_head = config.prompt_head
        if not config.inference_mode and config.prompt_head != 1:
            total_token_dim = config.token_dim * config.prompt_head
            self.embedding = torch.nn.Embedding(total_virtual_tokens, total_token_dim)
            # self.linear = torch.nn.Linear(total_token_dim, config.token_dim)
        # self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        
        
    
    def split_and_sum(self, tensor, n):
        last_dim_size = tensor.size(-1)
        if last_dim_size % n != 0:
            raise ValueError(f"The last dimension of size {last_dim_size} cannot be evenly divided into {n} parts.")
        split_size = last_dim_size // n
        splits = torch.split(tensor, split_size, dim=-1)
        result = sum(splits)
        result = result / n 
        return result
    
    def forward(self, indices):
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        if not self.inference_mode and self.prompt_head != 1:
            # prompt_embeddings = self.linear(prompt_embeddings)
            prompt_embeddings = self.split_and_sum(prompt_embeddings,self.prompt_head)
        return prompt_embeddings

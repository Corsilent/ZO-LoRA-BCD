import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.nn.init as init
from torch.nn.init import orthogonal_

def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class LoRALinear(nn.Linear):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            bfloat16: bool = False,
            bcd: bool = False,
            # Not sure if this will affect saving/loading models so just set it to be False
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        
        # for reset parameters
        self.lora_A_mask_record = None
        self.lora_B_mask_record = None
        self.bcd_activation = False
        self.bcd = bcd
        self.tolerance = 3
        
        # history lora
        self.lora_A_history_data = None
        self.lora_B_history_data = None
        
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        # self.reset_orthogonal()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        
        # statistics magnitude
        self.temporal_activation_sum = {}  # 存储激活统计
        self.lora_A_importance_mask = None
        self.lora_B_importance_mask = None
        
        self.lora_A_constant_mask1 = None
        self.lora_A_constant_mask2 = None
        self.lora_B_constant_mask1 = None
        self.lora_B_constant_mask2 = None
        self.mask1_activation = False
        self.mask2_activation = False
        

        # training steps
        self.step = 0
    
    def get_directions(self, A, B):
        """
        Compute and return the direction matrices for lora_A and lora_B.
        Each column of the direction matrix is a unit vector.
        """
        if not hasattr(self, 'lora_A') or not hasattr(self, 'lora_B'):
            raise AttributeError("LoRALinear layer must have lora_A and lora_B attributes.")
        
        # Compute direction for lora_A
        # lora_A = self.lora_A.data
        lora_A = A
        norm_A = torch.norm(lora_A, dim=0, keepdim=True)  # Compute norm across rows for each column
        direction_A = lora_A / norm_A
        
        # Compute direction for lora_B
        # lora_B = self.lora_B.data
        lora_B = B
        norm_B = torch.norm(lora_B, dim=0, keepdim=True)  # Compute norm across rows for each column
        direction_B = lora_B / norm_B
        
        return direction_A, direction_B
    
    def compute_cosine_similarity(self, direction_A, direction_B):
        """
        Compute the cosine similarity between two direction matrices.
        Returns a tensor of shape (r,) where r is the number of columns.
        """
        # Ensure the matrices have the same shape
        if direction_A.shape != direction_B.shape:
            raise ValueError("The two direction matrices must have the same shape.")
        
        # Compute cosine similarity for each column
        # direction_A: [r, in_features]
        # direction_B: [r, in_features]
        # similarity: [r]
        similarity = torch.sum(direction_A * direction_B, dim=1)  # sum across rows
        return similarity
    
    def compute_overall_direction_similarity(self, direction_A, direction_B):
        """
        Compute and return a single value representing the overall direction similarity between two matrices.
        This is achieved by taking the average of the cosine similarities of their corresponding columns.
        """
        similarity = self.compute_cosine_similarity(direction_A, direction_B)
        overall_similarity = torch.mean(similarity)
        return overall_similarity
    
    def bcd_compute_importance_mask(self, ini_threshold):
        """Compute the importance mask based on the provided method."""
        if not self.bcd_activation:
            lora_A_threshold = torch.quantile(self.temporal_activation_sum['lora_A'].to(dtype=torch.float32), ini_threshold)
            lora_B_threshold = torch.quantile(self.temporal_activation_sum['lora_B'].to(dtype=torch.float32), ini_threshold)
            self.lora_A_importance_mask = (self.temporal_activation_sum['lora_A'] >= lora_A_threshold).float()
            self.lora_B_importance_mask = (self.temporal_activation_sum['lora_B'] >= lora_B_threshold).float()
        else:
            if self.mask1_activation:
                self.lora_A_importance_mask = self.lora_A_constant_mask1
                self.lora_B_importance_mask = self.lora_B_constant_mask1
            elif self.mask2_activation:
                self.lora_A_importance_mask = self.lora_A_constant_mask2
                self.lora_B_importance_mask = self.lora_B_constant_mask2
            else:
                raise NotImplementedError("mask activation error")
                

    
    def bcd_apply_importance_mask(self):          
        """Apply the importance mask to the LoRA parameters."""      
        self.lora_A.grad *= self.lora_A_importance_mask.unsqueeze(dim=-1)
        self.lora_B.grad *= self.lora_B_importance_mask.unsqueeze(dim=-1)
        
    def reset_activation_stats(self, ini_threshold):
        """重置激活统计数据"""
        self.temporal_activation_sum.clear()
        self.step += 1
        if self.bcd:
            if not self.bcd_activation:
                if self.lora_A_mask_record is None:
                    self.lora_A_mask_record = self.lora_A_importance_mask
                    self.lora_B_mask_record = self.lora_B_importance_mask
                else:
                    self.lora_A_mask_record += self.lora_A_importance_mask
                    self.lora_B_mask_record += self.lora_B_importance_mask
                # 在参数选择期，让模型见过所有训练数据 16 * 100 > 1000
                if self.step % 50 == 0:
                    self.bcd_activation = True
                    
                    lora_A_threshold = torch.quantile(self.lora_A_mask_record.to(dtype=torch.float32), ini_threshold)
                    lora_B_threshold = torch.quantile(self.lora_B_mask_record.to(dtype=torch.float32), ini_threshold)
                    lora_A_constant_mask = (self.lora_A_mask_record >= lora_A_threshold).float()
                    lora_B_constant_mask = (self.lora_B_mask_record >= lora_B_threshold).float()
                    
                    # 找出值为 1 的位置
                    ones_indices = torch.nonzero(lora_A_constant_mask == 1).flatten()
                    # 随机选择一半的位置
                    num_to_select = len(ones_indices) // 2
                    selected_indices = ones_indices[torch.randperm(len(ones_indices))[:num_to_select]]
                    
                    self.lora_A_constant_mask1 = lora_A_constant_mask.clone()
                    self.lora_A_constant_mask1[selected_indices] = 0  # 将选中的位置置零
                    
                    self.lora_A_constant_mask2 = torch.zeros_like(lora_A_constant_mask)  # 创建一个全零 tensor
                    self.lora_A_constant_mask2[selected_indices] = 1  # 将选中的位置设为 1
                    
                    # ---------------------------------------------------------------------------
                    # 找出值为 1 的位置
                    ones_indices = torch.nonzero(lora_B_constant_mask == 1).flatten()
                    # 随机选择一半的位置
                    num_to_select = len(ones_indices) // 2
                    selected_indices = ones_indices[torch.randperm(len(ones_indices))[:num_to_select]]
                    
                    self.lora_B_constant_mask1 = lora_B_constant_mask.clone()
                    self.lora_B_constant_mask1[selected_indices] = 0  # 将选中的位置置零
                    
                    self.lora_B_constant_mask2 = torch.zeros_like(lora_B_constant_mask)  # 创建一个全零 tensor
                    self.lora_B_constant_mask2[selected_indices] = 1  # 将选中的位置设为 1
                    
                    # 设置 mask activation
                    self.mask1_activation = True
                    self.mask2_activation = False
                    
                    # 重置 record
                    self.lora_A_mask_record = None
                    self.lora_B_mask_record = None
            else:
                if (self.step - 50) % 400 == 0:
                    self.bcd_activation = False
                    self.mask1_activation = False
                    self.mask2_activation = False
                    self.lora_A_constant_mask2 = None
                    self.lora_B_constant_mask2 = None
                elif (self.step - 50) % 200 == 0:
                    self.mask1_activation = False
                    self.mask2_activation = True
                    self.lora_A_constant_mask1 = None
                    self.lora_B_constant_mask1 = None
        
        
    def reset_orthogonal(self):      
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            orthogonal_(self.lora_A, gain=0.5) 
            orthogonal_(self.lora_B, gain=0.5) 


    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
                print("successfully unmerge weights")
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True
                print("successfully merge weights")

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            
            lora_A_act = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
            lora_A_act = lora_A_act.reshape(-1, self.r).abs().sum(dim=0)  # [r]
            
            lora_B_act = lora_A_act @ self.lora_B.transpose(0, 1)  # [B, seq_len, out_features]
            lora_B_act = lora_B_act.reshape(-1, self.out_features).abs().sum(dim=0)  # [out_features]
            
            
            if "lora_A" not in self.temporal_activation_sum:
                self.temporal_activation_sum["lora_A"] = lora_A_act.detach()
                self.temporal_activation_sum["lora_B"] = lora_B_act.detach()
            else:
                self.temporal_activation_sum["lora_A"] += lora_A_act.detach()
                self.temporal_activation_sum["lora_B"] += lora_B_act.detach()
            # else:
            #     self.temporal_activation_sum["lora_A"] = torch.abs(self.temporal_activation_sum["lora_A"] - lora_A_act.detach())
            #     self.temporal_activation_sum["lora_B"] = torch.abs(self.temporal_activation_sum["lora_B"] - lora_B_act.detach())
                
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
    # def forward(self, x: torch.Tensor):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w
        
    #     lora_A_fp32 = self.lora_A.float()  # 显式转为FP32
    #     A_mean = lora_A_fp32.mean(dim=1, keepdim=True)
    #     A_centered = lora_A_fp32 - A_mean
    #     A_std = A_centered.std(dim=1, keepdim=True) + 1e-5
    #     A = (A_centered / A_std).to(self.lora_A.dtype)  # 转回原始数据类型
        
    #     # Weight Standardization for B (FP16兼容)
    #     lora_B_fp32 = self.lora_B.float()  # 显式转为FP32
    #     B_mean = lora_B_fp32.mean(dim=1, keepdim=True)
    #     B_centered = lora_B_fp32 - B_mean
    #     B_std = B_centered.std(dim=1, keepdim=True) + 1e-5
    #     B = (B_centered / B_std).to(self.lora_B.dtype)  # 转回原始数据类型
        
        

    #     if self.r > 0 and not self.merged:
    #         result = F.linear(x, T(self.weight), bias=self.bias)
    #         if self.r > 0:
    #             result += (self.lora_dropout(x) @ A.transpose(0, 1) @ self.lora_B.transpose(0,1)) * self.scaling
    #         return result
    #     else:
    #         return F.linear(x, T(self.weight), bias=self.bias)
    
    # def forward(self, x: torch.Tensor):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w

    #     if self.r > 0 and not self.merged:
    #         result = F.linear(x, T(self.weight), bias=self.bias)
    #         if self.r > 0:
    #             result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,1)) * self.scaling
    #         return result
    #     else:
    #         return F.linear(x, T(self.weight), bias=self.bias)
    


class LoRA:

    def __init__(self, model, r, alpha, bcd, args):
        """
        Input:
        r, alpha: LoRA hyperparameters
        float16: Whether the model parameters are float16 or not
        """

        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.bcd = bcd
        self.args = args

        if model.config.model_type in ["opt", "qwen2"]:
            attention_name = "attn"
        elif model.config.model_type == "roberta":
            attention_name = "self"
        elif model.config.model_type in ["llama", "mistral"]:
            attention_name = "self_attn"
        else:
            raise NotImplementedError

        # Insert LoRA
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                logger.info(f"Inject lora to: {key}")
                _, _, attn = find_module(model, key)

                if model.config.model_type == "opt":
                    original_q_weight = attn.q_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data
                    original_v_weight = attn.v_proj.weight.data
                    original_v_bias = attn.v_proj.bias.data
                    attn.q_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha,
                                             bias=model.config.enable_bias, bcd=self.bcd).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha,
                                             bias=model.config.enable_bias, bcd=self.bcd).to(original_v_weight.device)
                    if self.args.load_float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.q_proj.bias.data = original_q_bias
                    attn.v_proj.weight.data = original_v_weight
                    attn.v_proj.bias.data = original_v_bias
                    
                elif model.config.model_type == "roberta":
                    original_q_weight = attn.query.weight.data
                    original_q_bias = attn.query.bias.data
                    original_v_weight = attn.value.weight.data
                    original_v_bias = attn.value.bias.data
                    attn.query = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha,
                                             bias=True, bcd=self.bcd).to(original_q_weight.device)
                    attn.value = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha,
                                             bias=True, bcd=self.bcd).to(original_v_weight.device)
                    if self.args.load_float16:
                        attn.query.half()
                        attn.value.half()
                    attn.query.weight.data = original_q_weight
                    attn.query.bias.data = original_q_bias
                    attn.value.weight.data = original_v_weight
                    attn.value.bias.data = original_v_bias
                    
                elif model.config.model_type == "qwen2":
                    original_q_weight = attn.q_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data
                    original_v_weight = attn.v_proj.weight.data
                    original_v_bias = attn.v_proj.bias.data
                    attn.q_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size, r=r, lora_alpha=alpha,
                                             bias=True).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(model.config.hidden_size, model.config.hidden_size // 7, r=r, lora_alpha=alpha,
                                             bias=True).to(original_v_weight.device)
                    if self.args.load_bfloat16:
                        attn.q_proj.bfloat16()
                        attn.v_proj.bfloat16()
                    attn.q_proj.weight.data = original_q_weight
                    attn.q_proj.bias.data = original_q_bias
                    attn.v_proj.weight.data = original_v_weight
                    attn.v_proj.bias.data = original_v_bias
                    
                    
                elif model.config.model_type == "llama" and "llama-3.2-3B" in self.args.model_name:
                    # in early version of transformers, llama attention bias is hard coded to False
                    attention_bias = False if not hasattr(model.config, "attention_bias") else model.config.attention_bias
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data if attention_bias else None
                    original_v_bias = attn.v_proj.bias.data if attention_bias else None
                    attn.q_proj = LoRALinear(
                        model.config.hidden_size,
                        model.config.hidden_size,
                        r=r, lora_alpha=alpha, bias=attention_bias, bcd=self.bcd
                    ).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(
                        model.config.hidden_size,
                        model.config.hidden_size // 3,
                        r=r, lora_alpha=alpha, bias=attention_bias, bcd=self.bcd
                    ).to(original_v_weight.device)
                    if self.args.load_bfloat16:
                        attn.q_proj.bfloat16()
                        attn.v_proj.bfloat16()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                    if attention_bias:
                        attn.q_proj.bias.data = original_q_bias
                        attn.v_proj.bias.data = original_v_bias
                        
                elif model.config.model_type == "llama" and "llama-2-7b" in self.args.model_name:
                    # in early version of transformers, llama attention bias is hard coded to False
                    attention_bias = False if not hasattr(model.config, "attention_bias") else model.config.attention_bias
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data if attention_bias else None
                    original_v_bias = attn.v_proj.bias.data if attention_bias else None
                    attn.q_proj = LoRALinear(
                        model.config.hidden_size,
                        model.config.hidden_size,
                        r=r, lora_alpha=alpha, bias=attention_bias, bcd=self.bcd
                    ).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(
                        model.config.hidden_size,
                        model.config.hidden_size,
                        r=r, lora_alpha=alpha, bias=attention_bias, bcd=self.bcd
                    ).to(original_v_weight.device)
                    # if self.args.load_float16:
                    #     attn.q_proj.half()
                    #     attn.v_proj.half()
                    if self.args.load_bfloat16:
                        attn.q_proj.bfloat16()
                        attn.v_proj.bfloat16()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                    if attention_bias:
                        attn.q_proj.bias.data = original_q_bias
                        attn.v_proj.bias.data = original_v_bias
                
                elif model.config.model_type == "llama" and "llama-3.1-8B" in self.args.model_name:
                    # in early version of transformers, llama attention bias is hard coded to False
                    attention_bias = False if not hasattr(model.config, "attention_bias") else model.config.attention_bias
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    original_q_bias = attn.q_proj.bias.data if attention_bias else None
                    original_v_bias = attn.v_proj.bias.data if attention_bias else None
                    attn.q_proj = LoRALinear(
                        model.config.hidden_size,
                        model.config.hidden_size,
                        r=r, lora_alpha=alpha, bias=attention_bias, bcd=self.bcd
                    ).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(
                        model.config.hidden_size,
                        model.config.hidden_size // 4,
                        r=r, lora_alpha=alpha, bias=attention_bias, bcd=self.bcd
                    ).to(original_v_weight.device)
                    if self.args.load_bfloat16:
                        attn.q_proj.bfloat16()
                        attn.v_proj.bfloat16()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                    if attention_bias:
                        attn.q_proj.bias.data = original_q_bias
                        attn.v_proj.bias.data = original_v_bias
                        
                elif model.config.model_type == "mistral":
                    # in early version of transformers, llama attention bias is hard coded to False
                    config = model.config
                    original_q_weight = attn.q_proj.weight.data
                    original_v_weight = attn.v_proj.weight.data
                    head_dim = config.hidden_size // config.num_attention_heads
                    attn.q_proj = LoRALinear(
                        config.hidden_size,
                        config.hidden_size,
                        r=r, lora_alpha=alpha
                    ).to(original_q_weight.device)
                    attn.v_proj = LoRALinear(
                        config.hidden_size,
                        config.num_key_value_heads * head_dim,
                        r=r, lora_alpha=alpha
                    ).to(original_v_weight.device)
                    if self.args.load_float16:
                        attn.q_proj.half()
                        attn.v_proj.half()
                    attn.q_proj.weight.data = original_q_weight
                    attn.v_proj.weight.data = original_v_weight
                else:
                    raise NotImplementedError

        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            # if "lora_B" not in n:
            if "lora" not in n:
                p.requires_grad = False

import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer, r=4, alpha=16, dropout=0.0):
        super(LoRALinear, self).__init__()
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        self.lora_A = nn.Parameter(torch.zeros((self.in_features, r)))
        self.lora_B = nn.Parameter(torch.zeros((r, self.out_features)))
        self.dropout = nn.Dropout(p=dropout)
        
        self.reset_parameters()

    @property
    def weight(self):
        return self.original_layer.weight
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        original_out = self.original_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out

def inject_lora(model, r=4, alpha=16, dropout=0.0, target_module_names=None):
    """
    Replaces nn.Linear layers with LoRALinear layers in place.
    If target_module_names is provided (list of strings), only modules matching those names will be replaced.
    Otherwise, all nn.Linear layers are replaced.
    """
    reassignment = []
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if target_module_names is None or any(t in name for t in target_module_names):
                reassignment.append((name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout)))
        else:
            # Recursively apply
            inject_lora(module, r=r, alpha=alpha, dropout=dropout, target_module_names=target_module_names)
    
    for name, new_module in reassignment:
        setattr(model, name, new_module)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def get_lora_params(model):
    params = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params

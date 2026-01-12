import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import random

def apply_pc_grad(task_grads_list, task_names=None, logger=None, t_env=None):
    """
    task_grads_list: List[List[Tensor]]. Each element is a list of gradient tensors for one task.
                     Assumes all tasks have same parameter structure.
    task_names: List[str]. Optional list of task names for logging.
    logger: Optional logger object for logging stats.
    t_env: Optional timestep for logging.
    
    Returns: List[Tensor] (the summed projected gradient ready for step, matching structure of one task's grads)
    """
    
    # Flatten gradients for each task for vector operations
    flat_task_grads = []
    
    # Check structure match and flatten
    # We essentially filter out None gradients (frozen params) but need to be careful to keep alignment?
    # Usually p.grad is None if not trained. 
    # We assume 'template' comes from the first task's params. 
    # If a param is trained in task A but not task B, B's grade might be None?
    # In this specific codebase, we froze common parts and injected LoRA, 
    # so active parameters should be consistent across tasks (the LoRA adapters).
    
    # We'll use the first task that has gradients as the template for reconstruction
    # flattened vector will only contain non-None gradients.
    
    active_indices = []
    for i, grads in enumerate(task_grads_list):
        # Flatten all NON-NONE gradients
        valid_grads = [g for g in grads if g is not None]
        if valid_grads:
            flat_task_grads.append(torch.cat([g.view(-1) for g in valid_grads]))
            active_indices.append(i)
            
    if not flat_task_grads:
        return [None] * len(task_grads_list[0]) # No gradients
        
    grad_vecs = flat_task_grads # List of 1D tensors
    num_tasks = len(grad_vecs)
    
    # PCGrad Algorithm
    # We work on copies to avoid modifying originals during comparisons
    pc_grad_vecs = [g.clone() for g in grad_vecs]
    
    for i in range(num_tasks):
        indices = list(range(num_tasks))
        random.shuffle(indices)
        
        for j in indices:
            if i == j:
                continue
            
            g_i = pc_grad_vecs[i]
            g_j = grad_vecs[j] # Project against ORIGINAL gradient of task j
            
            dot = torch.dot(g_i, g_j)
            if dot < 0:
                # Subtract projection
                denom = torch.dot(g_j, g_j)
                factor = dot / (denom + 1e-8)
                g_i -= factor * g_j
                pc_grad_vecs[i] = g_i
    
    # Logging if requested
    if logger is not None and task_names is not None and len(task_names) == len(task_grads_list):
        # Map active indices back to names
        active_names = [task_names[i] for i in active_indices]
        
        for i in range(len(active_names)):
            for j in range(i + 1, len(active_names)):
                t1 = active_names[i]
                t2 = active_names[j]
                g1 = pc_grad_vecs[i]
                g2 = pc_grad_vecs[j]
                
                sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
                logger.log_stat(f"pcgrad_sim/{t1}_vs_{t2}", sim, t_env if t_env is not None else 0)

    # Sum the projected gradients from all tasks
    if not pc_grad_vecs:
        final_flat_grad = torch.zeros_like(grad_vecs[0]) if grad_vecs else None
    else:
        final_flat_grad = torch.stack(pc_grad_vecs).sum(dim=0)
        
    # Unflatten back to param structure using the first active task as template
    template_grads = task_grads_list[active_indices[0]]
    final_grads = []
    idx = 0
    for g in template_grads:
        if g is None:
            final_grads.append(None)
        else:
            numel = g.numel()
            final_grads.append(final_flat_grad[idx:idx+numel].view_as(g))
            idx += numel
            
    return final_grads

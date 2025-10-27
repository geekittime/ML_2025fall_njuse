"""
Neural Network architectures for Lab2 deep learning models
"""
import torch
import torch.nn as nn


# ==================== Task 1: Regression Models ====================

class MLPRegressor(nn.Module):
    """Multi-Layer Perceptron for Regression"""
    
    def __init__(self, input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.2):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class WideAndDeepRegressor(nn.Module):
    """Wide & Deep model for Regression"""
    
    def __init__(self, input_dim, deep_dims=[128, 64], dropout_rate=0.2):
        super(WideAndDeepRegressor, self).__init__()
        
        # Wide part: Simple linear transformation
        self.wide_part = nn.Linear(input_dim, 1)
        
        # Deep part: MLP for generalization
        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in deep_dims:
            deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep_part = nn.Sequential(*deep_layers)
    
    def forward(self, x):
        wide_out = self.wide_part(x)
        deep_out = self.deep_part(x)
        return wide_out + deep_out


class CrossLayer(nn.Module):
    """Cross Layer for Deep & Cross Network"""
    
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))
    
    def forward(self, x0, xl):
        """
        x0: Initial input
        xl: Output from previous layer
        """
        # Cross product: x0 * (xl^T * w) + bias + xl
        cross = torch.matmul(xl, self.weight)  # (batch, 1)
        cross = x0 * cross + self.bias + xl   # Residual connection
        return cross


class DeepCrossRegressor(nn.Module):
    """Deep & Cross Network for Regression"""
    
    def __init__(self, input_dim, num_cross_layers=3, deep_dims=[128, 64], dropout_rate=0.2):
        super(DeepCrossRegressor, self).__init__()
        
        # Cross Network
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        ])
        
        # Deep Network (MLP)
        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in deep_dims:
            deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.deep_part = nn.Sequential(*deep_layers)
        
        # Final output layer (concatenate cross and deep)
        self.final = nn.Linear(input_dim + deep_dims[-1], 1)
    
    def forward(self, x):
        # Cross Network
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        
        # Deep Network
        deep_out = self.deep_part(x)
        
        # Concatenate and output
        combined = torch.cat([xl, deep_out], dim=1)
        return self.final(combined)


class SharedBottomRegressor(nn.Module):
    """Shared-Bottom Multi-Task Learning for Regression
    (Can be used with single task by ignoring extra outputs)
    """
    
    def __init__(self, input_dim, shared_dims=[128, 64], task_dims=[32], 
                 num_tasks=1, dropout_rate=0.2):
        super(SharedBottomRegressor, self).__init__()
        
        # Shared bottom layers
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in shared_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.shared_bottom = nn.Sequential(*shared_layers)
        
        # Task-specific towers
        self.task_towers = nn.ModuleList()
        for _ in range(num_tasks):
            tower_layers = []
            tower_prev_dim = shared_dims[-1]
            for hidden_dim in task_dims:
                tower_layers.append(nn.Linear(tower_prev_dim, hidden_dim))
                tower_layers.append(nn.ReLU())
                tower_layers.append(nn.Dropout(dropout_rate))
                tower_prev_dim = hidden_dim
            tower_layers.append(nn.Linear(tower_prev_dim, 1))
            self.task_towers.append(nn.Sequential(*tower_layers))
    
    def forward(self, x):
        shared_output = self.shared_bottom(x)
        task_outputs = [tower(shared_output) for tower in self.task_towers]
        return task_outputs if len(task_outputs) > 1 else task_outputs[0]


class MMoERegressor(nn.Module):
    """Multi-gate Mixture-of-Experts for Multi-Task Learning"""
    
    def __init__(self, input_dim, num_experts=3, expert_dims=[64, 32], 
                 task_dims=[32], num_tasks=1, dropout_rate=0.2):
        super(MMoERegressor, self).__init__()
        
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # Expert networks
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert_layers = []
            prev_dim = input_dim
            for hidden_dim in expert_dims:
                expert_layers.append(nn.Linear(prev_dim, hidden_dim))
                expert_layers.append(nn.ReLU())
                expert_layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            self.experts.append(nn.Sequential(*expert_layers))
        
        # Gate networks (one per task)
        self.gates = nn.ModuleList()
        for _ in range(num_tasks):
            self.gates.append(nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=1)
            ))
        
        # Task-specific towers
        self.task_towers = nn.ModuleList()
        for _ in range(num_tasks):
            tower_layers = []
            tower_prev_dim = expert_dims[-1]
            for hidden_dim in task_dims:
                tower_layers.append(nn.Linear(tower_prev_dim, hidden_dim))
                tower_layers.append(nn.ReLU())
                tower_layers.append(nn.Dropout(dropout_rate))
                tower_prev_dim = hidden_dim
            tower_layers.append(nn.Linear(tower_prev_dim, 1))
            self.task_towers.append(nn.Sequential(*tower_layers))
    
    def forward(self, x):
        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, expert_dim)
        
        task_outputs = []
        for task_idx in range(self.num_tasks):
            # Get gate weights for this task
            gate_weights = self.gates[task_idx](x).unsqueeze(-1)  # (batch, num_experts, 1)
            
            # Weighted sum of expert outputs
            gated_output = torch.sum(expert_outputs * gate_weights, dim=1)  # (batch, expert_dim)
            
            # Task-specific tower
            task_output = self.task_towers[task_idx](gated_output)
            task_outputs.append(task_output)
        
        return task_outputs if len(task_outputs) > 1 else task_outputs[0]


# ==================== Task 2: Classification Models ====================

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for Binary Classification"""
    
    def __init__(self, input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class WideAndDeepClassifier(nn.Module):
    """Wide & Deep model for Binary Classification"""
    
    def __init__(self, input_dim, deep_dims=[128, 64], dropout_rate=0.2):
        super(WideAndDeepClassifier, self).__init__()
        
        # Wide part
        self.wide_part = nn.Linear(input_dim, 1)
        
        # Deep part
        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in deep_dims:
            deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        deep_layers.append(nn.Linear(prev_dim, 1))
        self.deep_part = nn.Sequential(*deep_layers)
    
    def forward(self, x):
        wide_out = self.wide_part(x)
        deep_out = self.deep_part(x)
        return wide_out + deep_out


class DeepCrossClassifier(nn.Module):
    """Deep & Cross Network for Binary Classification"""
    
    def __init__(self, input_dim, num_cross_layers=3, deep_dims=[128, 64], dropout_rate=0.2):
        super(DeepCrossClassifier, self).__init__()
        
        # Cross Network
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_cross_layers)
        ])
        
        # Deep Network
        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in deep_dims:
            deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.deep_part = nn.Sequential(*deep_layers)
        
        # Final output layer
        self.final = nn.Linear(input_dim + deep_dims[-1], 1)
    
    def forward(self, x):
        # Cross Network
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        
        # Deep Network
        deep_out = self.deep_part(x)
        
        # Concatenate and output
        combined = torch.cat([xl, deep_out], dim=1)
        return self.final(combined)


# ==================== Multi-Task Models ====================

class MultiTaskModel(nn.Module):
    """Multi-Task Learning model for both regression and classification"""
    
    def __init__(self, input_dim, shared_dims=[128, 64], 
                 regression_dims=[32], classification_dims=[32], dropout_rate=0.2):
        super(MultiTaskModel, self).__init__()
        
        # Shared bottom layers
        shared_layers = []
        prev_dim = input_dim
        for hidden_dim in shared_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        self.shared_bottom = nn.Sequential(*shared_layers)
        
        # Regression tower (Task 1: TTC prediction)
        regression_layers = []
        reg_prev_dim = shared_dims[-1]
        for hidden_dim in regression_dims:
            regression_layers.append(nn.Linear(reg_prev_dim, hidden_dim))
            regression_layers.append(nn.ReLU())
            regression_layers.append(nn.Dropout(dropout_rate))
            reg_prev_dim = hidden_dim
        regression_layers.append(nn.Linear(reg_prev_dim, 1))
        self.regression_tower = nn.Sequential(*regression_layers)
        
        # Classification tower (Task 2: Merge prediction)
        classification_layers = []
        cls_prev_dim = shared_dims[-1]
        for hidden_dim in classification_dims:
            classification_layers.append(nn.Linear(cls_prev_dim, hidden_dim))
            classification_layers.append(nn.ReLU())
            classification_layers.append(nn.Dropout(dropout_rate))
            cls_prev_dim = hidden_dim
        classification_layers.append(nn.Linear(cls_prev_dim, 1))
        self.classification_tower = nn.Sequential(*classification_layers)
    
    def forward(self, x):
        shared_output = self.shared_bottom(x)
        regression_output = self.regression_tower(shared_output)
        classification_output = self.classification_tower(shared_output)
        return regression_output, classification_output

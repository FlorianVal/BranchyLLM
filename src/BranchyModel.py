from transformers import PreTrainedModel
import torch
import copy

class BranchyModel(PreTrainedModel):
    """
    This class is a wrapper for transformer models with added functionality for branchy networks.
    It uses BranchyConfig to initialize a model and later will be extended to add branches.

    Args:
        config (BranchyLLMConfig): The configuration to initialize the model with.
        model (PreTrainedModel): The underlying transformer model to wrap.

    Returns:
        A model instance with the given configuration.
    """

    def __init__(self, config, model):
        super().__init__(model.config)
        # Initialize the base transformer model
        self.model = model
        
        # Get args for branchy model
        self.self_supervised_training = config.self_supervision
        self.num_branches = config.num_branches
        self.branch_locations = config.branch_locations
        
        # Get details on layering inside the model
        if hasattr(self.model.config, "n_layer") or hasattr(self.model.config, "num_hidden_layers"):  # If there is no n_layer in the config, there might be ways to get it from the model itself
            self.num_layers = self.model.config.n_layer if hasattr(self.model.config, "n_layer") else self.model.config.num_hidden_layers
            assert self.num_layers > 0, "The number of layers must be greater than 0"
            assert self.num_branches < self.num_layers, "The number of branches must be less than the number of layers"
            assert all([0 <= i < self.num_layers for i in self.branch_locations]), "The branch locations must be between 0 and num_layers"
        else:
            raise ValueError("cannot find n_layer in config")
            
        # Make sure the base model is frozen
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Instantiate heads. Default: heads are copies of the lm_head
        self.model.heads = torch.nn.ModuleList([copy.deepcopy(self.model.lm_head) for _ in range(self.num_branches)])

        # initialize heads
        for head in self.model.heads:
            head.apply(self.model._init_weights)
            # Make them trainable
            for param in head.parameters():
                param.requires_grad = True

        self.post_init()
        
    def generate(self, *args, **kwargs):
        # TODO it might be necessary to implement this function to be able to generate text from the model
        # Because we need to choose decoding method [https://huggingface.co/blog/how-to-generate] 
        # so it's either we override this function or trick inside transformers.PretrainedModel.generate
        return self.model.generate(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        # TODO
        # For now, simply forward the call to the underlying transformer model
        return self.model(*args, **kwargs)

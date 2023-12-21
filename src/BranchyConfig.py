from typing import Optional


class BranchyConfig:
    """
    This is the configuration class to store the configuration of a Branchy model.
    It is used to instantiate the Branchy model according to the specified arguments,
    defining the model architecture.

    Args:
        self_supervision (bool, optional): Indicates whether to use self-supervision 
                                            for training branch heads. Defaults to False.
        branch_locations (list of int, optional): Specifies the layers where branches should be added. 
                                           Defaults to None, which can be interpreted as evenly 
                                           distributing branches across the model.
        **kwargs: Additional keyword arguments passed to the base class (`PretrainedConfig`) 
                  and used to define model-specific configurations.
    """

    def __init__(self,
                 self_supervision: bool = False,
                 branch_locations: Optional[list] = None,
                 ):
        self.self_supervision = self_supervision
        self.branch_locations = branch_locations if branch_locations is not None else []
        
        assert type(self.branch_locations) == list, "branch_locations must be a list"
        assert type(self.self_supervision) == bool, "self_supervision must be a boolean"
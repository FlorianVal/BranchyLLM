def print_model_parameter_distribution(model, indent=0, parent_total_params=0):
    """
    Recursively prints the parameter distribution of a PyTorch model.

    Args:
        model (torch.nn.Module): The model or submodule to analyze.
        indent (int): The indentation level (for nested submodules).
        parent_total_params (int): The total number of parameters of the parent module.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if parent_total_params == 0:
        parent_total_params = total_params

    indent_str = '    ' * indent
    if parent_total_params == 0:
        perc_of_parent = 100.0
    else:
        perc_of_parent = 100.0 * total_params / parent_total_params
    print(f"{indent_str}({model.__class__.__name__}): {perc_of_parent:.2f}% -> {total_params} Parameters")

    for _, submodule in model.named_children():
        print_model_parameter_distribution(submodule, indent + 1, total_params)

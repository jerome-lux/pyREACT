def diffusivity_model(porosity, d0: float = 1, exponent: float = 1.5):
    """
    Map porosity values to diffusion coefficients using a power-law relationship.

    Args:
        porosity (numpy.ndarray): Porosity field.
        exponent (float): Exponent for the power-law relationship.

    Returns:
        numpy.ndarray: Diffusion coefficient field.
    """
    return d0 * (porosity**exponent)
from modules import calculate_sigma_profile, calculate_gamma


def calculate_binary_gamma(
    SMILES1: str,
    SMILES2: str,
    x1: float,
    x2: float,
    T: float,
) -> list:

    return calculate_gamma(
        [calculate_sigma_profile(SMILES1), calculate_sigma_profile(SMILES2)],
        [x1, x2],
        T,
    )

import numpy as np
from scipy.stats import chi2 as chi2
from typing import Optional, Callable

# TODO: Add chi-squared for distribution.


def chi_sq_fit(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    y_error: np.ndarray,
    f: Callable[[], float],
    n_params: int,
    alpha: Optional[float] = 0.05,
) -> None:
    """
    This function performs a chi-squared test
    to check the validity of a fit.

    Parameters
    ---
    x_obs: numpy.ndarray
        The measured x-values (used for the fit).
    y_obs: numpy.ndarray
        The measured y-values (used for the fit).
    y_err: numpy.ndarray
        The errors on the y-values.
    f: function
        The function used for the fit.
    n_params: int
        The number of parameters obtained from the fit.

    Optional Parameters
    ---
    alpha: float
        The significance level of the test.
        It is set to 5% by default.
    """

    if alpha > 1:
        raise Exception("The significance level must be less than 100%")

    dof = len(x_obs) - n_params

    chi_2 = (((y_obs - f(x_obs)) / y_error) ** 2).sum()
    chi_2_reduced = chi_2 / dof
    chi_2_left = chi2.ppf(alpha / 2, dof)
    chi_2_right = chi2.ppf(1 - alpha / 2, dof)
    p_value = 1 - chi2.cdf(chi_2, df=dof)

    print("================================================")
    print("=====       CHI-SQUARED TEST RESULTS       =====")
    print("================================================\n")
    print(f"{'Degrees of freedom:':<20} {dof:>27}")
    print(f"{'Significance level:':<20} {f'{alpha*100} %':>27}")
    print(
        f"{'Critical values:':<20} {f'{np.round(chi_2_left, 2)}, {np.round(chi_2_right, 2)}':>27}\n"
    )
    print(f"{'Chi-squared:':<20} {np.round(chi_2, 2):>27}")
    print(f"{'Reduced chi-squared:':<20} {np.round(chi_2_reduced, 2):>27}")
    print(f"{'P-value:':<20} {np.round(p_value, 3):>27}\n")

    if p_value > alpha / 2 and p_value < 1 - alpha / 2:
        print("Null hypothesis H0 cannot be rejected")
    elif p_value < alpha / 2:
        print("Null hypothesis H0 must be rejected (right tail)")
    else:
        print("Null hypothesis H0 must be rejected (left tail)")

    print("================================================")

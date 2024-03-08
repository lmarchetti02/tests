import numpy as np
from numba import njit
from scipy.stats import chi2 as chi2
from typing import Optional, Callable


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
    to check the goodness of a regression.

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


@njit
def _check_bins(bin_values: np.ndarray) -> bool:
    """
    This helper function checks whether an array of
    n bins of an histogram contains only values greater
    than 5.
    """
    for value in bin_values:
        if value <= 5:
            return False

    return True


@njit
def _get_bin_center(bins_edges: np.ndarray) -> np.ndarray:
    """
    This helper function takes an (n+1)-array, containing
    the edges of the bins of an histogram, and returns
    an n-array with the centers of the bins.

    Parameters
    ---
    bins_edges: numpy.ndarray
        The array with the n+1 edges.
    """
    res = np.empty(len(bins_edges) - 1)

    for i in range(len(bins_edges) - 1):
        res[i] = bins_edges[i] + abs((bins_edges[i] - bins_edges[i + 1])) / 2

    return res


def chi_sq_dist(
    bins_values: np.ndarray,
    bins_edges: np.ndarray,
    f: Callable[[], float],
    f_continuous: bool,
    n_params: int,
    alpha: Optional[float] = 0.05,
) -> None:
    """
    This function performs a chi-squared test
    to check the goodness of a distribution fit.

    Parameters
    ---
    bins_values: numpy.ndarray
        The array with the values of each bin of
        the histogram.
    bins_edges: numpy.ndarray
        The array with the edges of each bin of
        the histogram. It can also be the array with
        the centers of the bins.
    f: function
        The function that describes the pdf of
        the expected distribution.
    f_continuous: bool
        `True` if the expected distribution is continuous,
        `False` if it is discrete.
    n_params: int
        The number of parameters of the expected distribution.

    Optional Parameters
    ---
    alpha: float
        The significance level of the test.
        It is set to 5% by default.
    """

    if alpha > 1:
        raise ValueError("The significance level must be less than 100%")

    # get centers of bins
    if len(bins_values) == len(bins_edges) - 1:
        bins_centers = _get_bin_center(bins_edges)
    else:
        bins_centers = bins_edges

    dof = len(bins_values) - n_params - 1
    total_events = bins_values.sum()
    expected_values = total_events * f(bins_centers)

    # get correct expected frequencies
    if f_continuous:
        bins_width = bins_centers[1] - bins_centers[0]
        expected_values *= bins_width

    if not _check_bins(expected_values):
        raise RuntimeError("The expected frequencies must all have a value greater than 5.")

    chi_2 = (((bins_values - expected_values) / expected_values) ** 2).sum()
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

import numpy as np
from scipy.stats import norm as norm
from typing import Optional


def test_z(m_1: float, m_2: float, e_1: float, e_2: float, alpha: Optional[float] = 0.05) -> None:
    """
    This function performs a z-test to compare
    two measure in the form: (m_1 +- e_1) and
    (m_2 +- e_2).

    Parameters
    ---
    m_1: float
        First measure.
    m_2: float
        Second measure.
    e_1: float
        First uncertainty.
    e_2: float
        Second uncertainty.

    Optional Parameters
    ---
    alpha: float
        The significance level of the test.
        It is set to 5% by default.
    """
    if alpha > 1:
        raise Exception("The significance level must be less than 100%")

    z_calc = (m_1 - m_2) / np.sqrt(e_1**2 + e_2**2)
    z_cr = norm.ppf(alpha / 2)
    p_value = norm.cdf(-abs(z_calc)) * 2

    print("======================================")
    print("===         Z-TEST RESULTS         ===")
    print("======================================\n")

    print(f"{'Significance level:':<20} {f'{alpha*100} %':>17}")
    print(f"{'|z critical|:':<20} {np.round(abs(z_cr), 2):>17}\n")
    print(f"{'|z from data|:':<20} {np.round(abs(z_calc), 2):>17}")
    print(f"{'P-value:':<20} {np.round(p_value, 3):>17}\n")

    if abs(z_calc) <= abs(z_cr):
        print("Null hypothesis H0 cannot be rejected")
    else:
        print("Null hypothesis H0 must be rejected")

    print("=" * 38)

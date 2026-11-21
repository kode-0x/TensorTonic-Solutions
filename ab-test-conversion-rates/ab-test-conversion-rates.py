import math
from statistics import NormalDist

def ab_test_conversion(control_conversions: int, control_n: int, treatment_conversions: int, treatment_n: int, alpha: float = 0.05) -> dict:
    """
    Returns lift, z, p_value, ci_low, ci_high and significant for a two-proportion z-test.
    """
    control_rate = control_conversions / control_n
    treatment_rate = treatment_conversions / treatment_n
    lift = treatment_rate - control_rate

    pooled_rate = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    pooled_var = pooled_rate * (1.0 - pooled_rate) * (1.0 / control_n + 1.0 / treatment_n)

    dist = NormalDist()
    if pooled_var == 0.0:
        z = 0.0
        p_value = 1.0
    else:
        pooled_se = math.sqrt(pooled_var)
        z = lift / pooled_se
        p_value = 2.0 * dist.cdf(-abs(z))

    unpooled_se = math.sqrt(
        control_rate * (1.0 - control_rate) / control_n
        + treatment_rate * (1.0 - treatment_rate) / treatment_n
    )
    z_crit = dist.inv_cdf(1.0 - alpha / 2.0)
    margin = z_crit * unpooled_se

    ci_low = lift - margin
    ci_high = lift + margin
    significant = bool(p_value < alpha)

    return {
        "lift": float(lift),
        "z": float(z),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": significant,
    }
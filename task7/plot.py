import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

from numpy import sign
import numpy as np

def cubic_roots_real_part(a, b, c) -> list[float]:
    """
    Возвращает список действительных частей корней кубического уравнения
    вида x^3 + ax^2 + bx + c
    """
    q = (a**2 - 3.0 * b) / 9.0
    r = (2.0 * a**3 - 9.0 * a * b + 27.0 * c) / 54.0
    q_pow_3 = q**3
    s = q_pow_3 - r**2

    if s > 0.0:
        angle = math.acos(r / math.sqrt(q_pow_3)) / 3.0
        sqrt_q = math.sqrt(q)
        return [-2.0 * sqrt_q * math.cos(angle) - a / 3.0,
                -2.0 * sqrt_q * math.cos(angle + 2.0 * math.pi / 3.0) - a / 3.0,
                -2.0 * sqrt_q * math.cos(angle - 2.0 * math.pi / 3.0) - a / 3.0]

    if abs(s) < 1.0e-10:
        cbrt_r = math.cbrt(r)
        x2 = cbrt_r - a / 3.0
        return [-2.0 * cbrt_r - a / 3.0, x2]

    if q > 0.0:
        angle = math.acosh(abs(r) / math.sqrt(q_pow_3)) / 3.0
        sqrt_q = math.sqrt(q)
        coeff = sign(r) * sqrt_q * math.cosh(angle)
        x2 = coeff - a / 3.0
        return [-2.0 * coeff - a / 3.0,
                x2,
                x2]

    if q < 0.0:
        angle = math.asinh(abs(r) / math.sqrt(abs(q_pow_3))) / 3.0
        coeff = sign(r) * math.sqrt(abs(q)) * math.sinh(angle)
        x2 = coeff - a / 3.0
        return [-2.0 * coeff - a / 3.0,
                x2,
                x2]

    x1 = -math.cbrt(c - a**3 / 27.0) - a / 3.0
    x2 = -(a + x1) / 2.0
    return [x1, x2, x2]

def eigenvalues_real_parts(prandtl: float, rayleigh: float) -> list[float]:
    """
    Задает коэффициенты куб. уравнения по числам Прандтля и Рэлея
    и возвращает реальные части всех корней
    """
    return cubic_roots_real_part(
        a = prandtl + 2.0,
        b = prandtl + rayleigh,
        c = 2.0 * prandtl * (rayleigh - 1.0)
    )

def is_stable(prandtl: float, rayleigh: float) -> bool:
    """
    True: все реальные части отрицательные (устойчивое равновесие, решение возвращается к стационарной точке)
    False: хотя бы одна положительная (неустойчивое равновесие)
    """
    real_parts = eigenvalues_real_parts(prandtl = prandtl, rayleigh = rayleigh)
    for real_part in real_parts:
        if real_part > 0.0:
            return False
    return True

if __name__=='__main__':
    # lets discover stability
    rayleighs = np.linspace(0.001, 2.0)
    prandtls = np.linspace(0.001, 2.0)

    def bool_to_int(val: bool) -> int:
        return 1 if val else -1

    stability_region = np.zeros(shape = (len(rayleighs), len(prandtls)))
    for idx_p, p in enumerate(prandtls):
        for idx_r, r in enumerate(rayleighs):
            stability_region[idx_p][idx_r] = bool_to_int(is_stable(prandtl=p, rayleigh=r))

    plt.figure()
    ax = plt.gca()
    img = ax.imshow(stability_region,
                    extent=(rayleighs.min(), rayleighs.max(), prandtls.min(), prandtls.max()),
                    cmap = ListedColormap(['darkorange', 'lightseagreen']),
                    origin='lower',
                    interpolation='none')
    fig = plt.gcf()
    cb = fig.colorbar(img, ax=ax, ticks = [-1, 1])
    cb.ax.set_yticklabels([r'$Re(\lambda) > 0$', r'$Re(\lambda) < 0$'])
    ax.set_title(r'$Re(\lambda)$')
    ax.set_xlabel('Число Рэлея')
    ax.set_ylabel('Число Прандтля')
    plt.show()

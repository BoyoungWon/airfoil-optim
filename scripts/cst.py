import numpy as np
import math
from typing import Tuple

def cosine_spacing(n: int) -> np.ndarray:
    i = np.arange(n)
    return 0.5 * (1.0 - np.cos(np.pi * i / (n - 1)))

def bernstein_matrix(x: np.ndarray, n: int) -> np.ndarray:
    """
    Returns matrix B of shape (len(x), n+1) where B[:,i] = Bernstein(n,i)(x).
    """
    B = np.zeros((len(x), n + 1), dtype=float)
    for i in range(n + 1):
        B[:, i] = math.comb(n, i) * (x ** i) * ((1.0 - x) ** (n - i))
    return B

def class_function(x: np.ndarray, N1: float = 0.5, N2: float = 1.0) -> np.ndarray:
    # Avoid endpoint issues (C(0)=C(1)=0 anyway)
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    return (x ** N1) * ((1.0 - x) ** N2)

def parse_selig_dat(text: str) -> Tuple[str, np.ndarray]:
    """
    Parses a .dat-like airfoil text. Returns (name, pts) where pts is (N,2).
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    name = lines[0]
    pts = []
    for ln in lines[1:]:
        parts = ln.replace("D", "E").split()  # handle Fortran D exponent if present
        if len(parts) >= 2:
            x = float(parts[0])
            y = float(parts[1])
            pts.append((x, y))
    return name, np.array(pts, dtype=float)

def split_upper_lower(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits TE->upper->LE->lower->TE formatted points into upper and lower arrays.
    Returns (upper, lower) each sorted by x increasing (LE->TE).
    """
    # Find leading edge index = minimum x (if multiple, take first min)
    i_le = int(np.argmin(pts[:, 0]))

    upper = pts[: i_le + 1]      # TE -> ... -> LE
    lower = pts[i_le: ]          # LE -> ... -> TE

    # Convert both to x increasing for fitting convenience
    upper = upper[::-1]          # LE -> ... -> TE
    # lower is already LE -> ... -> TE in typical files

    # If duplicates at LE/TE exist, that’s fine.
    # Sort by x to ensure monotonic for interpolation
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    return upper, lower

def resample_curve_xy(curve: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Resample y(x) by linear interpolation. Assumes curve[:,0] spans [0,1].
    """
    x = curve[:, 0]
    y = curve[:, 1]
    # Remove potential duplicate x values for interp
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    return np.interp(x_new, x_unique, y_unique)

def fit_cst_surface(
    x: np.ndarray, y: np.ndarray, n: int,
    N1: float = 0.5, N2: float = 1.0,
    dte: float = 0.0, upper: bool = True
) -> np.ndarray:
    """
    Fit CST weights A (length n+1) for one surface: y = C(x)*B(x)A + te_term
    """
    C = class_function(x, N1, N2)
    B = bernstein_matrix(x, n)                 # (m, n+1)
    te = x * (dte * 0.5) * (1.0 if upper else -1.0)

    # Solve: y - te = diag(C) @ B @ A
    M = (C[:, None] * B)                       # (m, n+1)
    rhs = y - te
    A, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    return A

def cst_eval_surface(x: np.ndarray, A: np.ndarray, N1=0.5, N2=1.0, dte=0.0, upper=True) -> np.ndarray:
    n = len(A) - 1
    C = class_function(x, N1, N2)
    B = bernstein_matrix(x, n)
    te = x * (dte * 0.5) * (1.0 if upper else -1.0)
    return (C * (B @ A)) + te

def fit_cst_airfoil(
    text: str,
    n: int = 8,
    n_fit_points: int = 200,
    N1: float = 0.5,
    N2: float = 1.0,
    dte: float = 0.0
):
    """
    Fits CST to an airfoil defined in Selig format text.
    Returns dict with Au, Al, x_fit, yu_fit, yl_fit, etc.
    """
    name, pts = parse_selig_dat(text)
    upper, lower = split_upper_lower(pts)

    # Fit on cosine-spaced x grid (recommended)
    x_fit = cosine_spacing(n_fit_points)

    yu = resample_curve_xy(upper, x_fit)
    yl = resample_curve_xy(lower, x_fit)

    Au = fit_cst_surface(x_fit, yu, n, N1, N2, dte=dte, upper=True)
    Al = fit_cst_surface(x_fit, yl, n, N1, N2, dte=dte, upper=False)

    # Evaluate fitted curves (for error checking)
    yu_hat = cst_eval_surface(x_fit, Au, N1, N2, dte=dte, upper=True)
    yl_hat = cst_eval_surface(x_fit, Al, N1, N2, dte=dte, upper=False)

    rms_u = float(np.sqrt(np.mean((yu_hat - yu) ** 2)))
    rms_l = float(np.sqrt(np.mean((yl_hat - yl) ** 2)))

    return {
        "name": name,
        "n": n,
        "Au": Au,
        "Al": Al,
        "x_fit": x_fit,
        "yu_data": yu,
        "yl_data": yl,
        "yu_fit": yu_hat,
        "yl_fit": yl_hat,
        "rms_upper": rms_u,
        "rms_lower": rms_l,
    }

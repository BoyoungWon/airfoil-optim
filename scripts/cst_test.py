import os
import numpy as np
import math

# ===================== CST CORE =====================

def bernstein_matrix(x: np.ndarray, n: int) -> np.ndarray:
    B = np.zeros((len(x), n + 1))
    for i in range(n + 1):
        B[:, i] = math.comb(n, i) * x**i * (1 - x)**(n - i)
    return B

def class_function(x, N1=0.5, N2=1.0):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return (x**N1) * ((1 - x)**N2)

def cosine_spacing(n):
    i = np.arange(n)
    return 0.5 * (1 - np.cos(np.pi * i / (n - 1)))

def cst_surface(x, A, upper=True, dte=0.0):
    n = len(A) - 1
    C = class_function(x)
    B = bernstein_matrix(x, n)
    te = x * dte * 0.5 * (1 if upper else -1)
    return C * (B @ A) + te

def cst_airfoil(Au, Al, n_points=200, dte=0.0):
    x = cosine_spacing(n_points)
    yu = cst_surface(x, Au, upper=True, dte=dte)
    yl = cst_surface(x, Al, upper=False, dte=dte)

    upper = np.column_stack([x, yu])[::-1]   # TE → LE
    lower = np.column_stack([x, yl])[1:]     # LE → TE
    return np.vstack([upper, lower])

# ===================== FITTING =====================

def parse_dat(text):
    lines = text.strip().splitlines()
    name = lines[0].strip() if lines else "AIRFOIL"
    pts = []
    for ln in lines[1:]:
        p = ln.replace("D", "E").split()
        if len(p) == 2:
            pts.append([float(p[0]), float(p[1])])
    pts = np.array(pts, dtype=float)
    if pts.shape[0] < 10:
        raise ValueError("Not enough points parsed from the .dat file.")
    return name, pts

def split_upper_lower(pts):
    i_le = np.argmin(pts[:, 0])
    upper = pts[: i_le + 1][::-1]
    lower = pts[i_le:]
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    return upper, lower

def split_upper_lower_with_x(pts):
    i_le = np.argmin(pts[:, 0])

    upper = pts[: i_le + 1]      # TE → LE
    lower = pts[i_le:]           # LE → TE

    # sort by x increasing for evaluation
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]

    return upper, lower


def interp_y(curve, x_new):
    x, y = curve[:, 0], curve[:, 1]
    x, idx = np.unique(x, return_index=True)
    return np.interp(x_new, x, y[idx])

def fit_surface(x, y, n, upper=True):
    C = class_function(x)
    B = bernstein_matrix(x, n)
    M = C[:, None] * B
    A, *_ = np.linalg.lstsq(M, y, rcond=None)
    return A

def fit_cst(text, n=8, fit_points=200):
    name, pts = parse_dat(text)
    upper, lower = split_upper_lower(pts)
    x = cosine_spacing(fit_points)
    yu = interp_y(upper, x)
    yl = interp_y(lower, x)
    Au = fit_surface(x, yu, n, upper=True)
    Al = fit_surface(x, yl, n, upper=False)
    return name, Au, Al, 

def cst_airfoil_on_original_x(Au, Al, upper, lower, dte=0.0):
    xu = upper[:, 0]
    xl = lower[:, 0]

    yu = cst_surface(xu, Au, upper=True,  dte=dte)
    yl = cst_surface(xl, Al, upper=False, dte=dte)

    # restore original ordering
    upper_new = np.column_stack([xu, yu])[::-1]   # TE → LE
    lower_new = np.column_stack([xl, yl])[1:]     # skip duplicate LE

    return np.vstack([upper_new, lower_new])

# ===================== IO =====================

def to_dat(name, coords):
    lines = [name]
    for x, y in coords:
        lines.append(f"{x:.8f} {y:.8f}")
    return "\n".join(lines)

# ===================== RANDOM SAMPLING =====================

def random_modify_drastic(Au, Al, rng,
                          p_big=0.15, p_medium=0.35,
                          sig_small_t=0.006, sig_small_c=0.003,
                          sig_med_t=0.020,  sig_med_c=0.010,
                          sig_big_t=0.060,  sig_big_c=0.030,
                          idx_mode="safe"):
    """
    Mixture sampler:
      - small (default)
      - medium
      - big/drastic (rare)

    Thickness-mode:  Au += dt, Al -= dt
    Camber-mode:     Au += dc, Al += dc
    """

    Au_mod = Au.copy()
    Al_mod = Al.copy()

    n = len(Au) - 1  # CST order

    # Choose which coefficients to perturb
    if idx_mode == "safe":
        # avoid very LE/TE sensitive coefficients
        idx = np.arange(2, n-1)   # e.g., for n=8 -> indices 2..6
    elif idx_mode == "all_mid":
        idx = np.arange(1, n)     # 1..n-1
    else:
        idx = np.arange(len(Au))

    # Pick scale regime
    u = rng.random()
    if u < p_big:
        sig_t, sig_c = sig_big_t, sig_big_c
    elif u < p_big + p_medium:
        sig_t, sig_c = sig_med_t, sig_med_c
    else:
        sig_t, sig_c = sig_small_t, sig_small_c

    # Optional: bias where changes happen (mid + aft gets more weight)
    # This makes "drastic" changes look like real airfoil families rather than noise.
    w = np.ones_like(idx, dtype=float)
    # heavier on mid/aft coefficients (roughly corresponds to mid/aft chord shaping)
    w *= np.linspace(0.8, 1.4, len(idx))

    dt = rng.normal(0.0, sig_t, size=len(idx)) * w
    dc = rng.normal(0.0, sig_c, size=len(idx)) * w

    # Apply thickness + camber components
    Au_mod[idx] += dt + dc
    Al_mod[idx] += -dt + dc

    # Add an occasional "global camber kick" (very drastic but coherent)
    if u < p_big:
        kick = rng.normal(0.0, sig_big_c)  # same shift both surfaces
        Au_mod[idx] += 0.5 * kick
        Al_mod[idx] += 0.5 * kick

    return Au_mod, Al_mod


def is_valid_airfoil(Au, Al, n_points=200, dte=0.0,
                     edge_clip=0.02,
                     min_thickness=5e-5,
                     max_thickness=0.25,   # <-- add
                     max_abs_y=0.6):
    x = cosine_spacing(n_points)
    yu = cst_surface(x, Au, upper=True,  dte=dte)
    yl = cst_surface(x, Al, upper=False, dte=dte)

    if np.any(np.isnan(yu)) or np.any(np.isnan(yl)):
        return False

    if np.max(np.abs(yu)) > max_abs_y or np.max(np.abs(yl)) > max_abs_y:
        return False

    t = yu - yl
    mask = (x > edge_clip) & (x < 1.0 - edge_clip)

    if np.min(t[mask]) < min_thickness:
        return False
    if np.max(t[mask]) > max_thickness:
        return False

    return True




# ===================== MAIN =====================

if __name__ == "__main__":
    # ---- config ----
    base_dat_path = "naca4412.dat"
    out_dir = "airfoil_samples_1000"
    n_samples = 100
    n_cst = 8                 # your n=8 -> 9 coefficients per surface
    n_points = 200
    dte = 0.0

    # randomization controls
    seed = 42
    sigma_thickness = 0.02
    sigma_camber = 0.01

    # validity / retries
    max_tries_per_sample = 200

    os.makedirs(out_dir, exist_ok=True)

    with open(base_dat_path, "r") as f:
        text = f.read()

    name, Au, Al = fit_cst(text, n=n_cst)

    # save the fitted baseline once
    base_coords = cst_airfoil(Au, Al, n_points=n_points, dte=dte)
    with open(os.path.join(out_dir, "0000_base_cst_fit.dat"), "w") as f:
        f.write(to_dat(name + " CST_FIT", base_coords))

    rng = np.random.default_rng(seed)

    saved = 0
    for k in range(1, n_samples + 1):
        ok = False
        for _ in range(max_tries_per_sample):
            Au_mod, Al_mod = random_modify_drastic(Au, Al, rng)

            if is_valid_airfoil(Au_mod, Al_mod, n_points=n_points, dte=dte,
                    min_thickness=1e-4, edge_clip=0.02):
                ok = True
                break


        if not ok:
            raise RuntimeError(
                f"Failed to generate a valid airfoil for sample {k} after {max_tries_per_sample} tries. "
                f"Try reducing sigma_thickness/sigma_camber or relaxing validity thresholds."
            )

        coords = cst_airfoil_on_original_x(Au_mod, Al_mod, upper, lower, dte=dte)

        fname = os.path.join(out_dir, f"{k:04d}_cst_random.dat")
        with open(fname, "w") as f:
            f.write(to_dat(f"{name} CST_RAND_{k:04d}", coords))

        saved += 1

    print(f"Saved {saved} airfoil .dat files to: {out_dir}")

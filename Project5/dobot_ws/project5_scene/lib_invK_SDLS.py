import numpy as np
import io, sys
from numpy.linalg import norm
from typing import Dict, Tuple, Optional, Union
from lib_forwardK import ht_compute
from dataclasses import dataclass
from contextlib import redirect_stdout

@dataclass
class SDLSParams:
    ee_frame: str = "T06"  #* 'T06' (flange) or 'T09' (gripper tip)
    w_pos: float = 1.0    #* Weight for position error (meters)
    w_ori: float = 1.0    #* Weight for orientation error (radians)
    r_eff: float = 0.25   #* meters per rad, effective radius of end-effector for orientation weighting
    dmax_pos: Optional[float] = 0.10 #* ClampMag for position error (meters)
    dmax_ori: Optional[float] = None #* ClampMag for orientation error (radians)
    gamma_max_deg: float = 45.0      #* Global per-step joint change limit (degrees)
    v_max_deg_step: Optional[Union[float, np.ndarray]] = None #* Per-joint max change per step (degrees)
    joint_limits_deg: Optional[Tuple[np.ndarray, np.ndarray]] = None #* (lower, upper) in degrees or None
    sigma_eps: float = 1e-9  #* Threshold below which singular values are treated as zero
    #* Stopping tolerances
    eps_pos: float = 1e-4   #* Position tolerance (meters)
    eps_ori: float = 1e-3    #* Orientation tolerance (radians)

@dataclass
class SDLSSolveConfig:
    max_iters: int = 200
    tol_improve:float = 1e-6 #* Minimum improvement in position+orientation error norm to count as progress
    verbose: bool = False

def fk_cr3(q_deg: np.ndarray, ee_frame: str = "T06", quiet: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute forward kinematics for the CR3 robot.
    Args:
        q_deg (np.ndarray): Joint angles in degrees, shape (6,).
        ee_frame (str): End-effector frame to return ("T06" or "T09").
    Returns:
        T (Dict[str, np.ndarray]): Homogeneous transformation matrices from base to each joint and end-effector.
        p_list (np.ndarray): Origins of each joint frame in base frame, shape (7, 3).
        z_list (np.ndarray): Z axes of each joint frame in base frame, shape (6, 3).
        T0e (np.ndarray): Homogeneous transformation matrix from base to end-effector frame
    """
    q_deg = np.asarray(q_deg, dtype=float).reshape(6,)
    #* Running lib_forwardK returns (T01..T09)
    #* We assume ht_compute(J_deg) -. T01..T09 (each 4x4)
    #* Sometimes we don't want to see the printout, we can suppress it here
    if quiet:
        sink = io.StringIO()
        with redirect_stdout(sink):
            T01, T02, T03, T04, T05, T06, T07, T08, T09 = ht_compute(q_deg)
    else:
        T01, T02, T03, T04, T05, T06, T07, T08, T09 = ht_compute(q_deg)

    T = {
        "T00": np.eye(4),
        "T01": T01, "T02": T02, "T03": T03,
        "T04": T04, "T05": T05, "T06": T06,
        "T07": T07, "T08": T08, "T09": T09,
    }
    #* Select end-effector frame
    if ee_frame not in ("T06", "T09"):
        raise ValueError("ee_frame must be 'T06' (flange) or 'T09' (gripper tip)")
    T0e = T[ee_frame]
    #* Extract origings p_i and z_i for jacobian computation
    #* P_0 is [0,0,0], z_0 is the base Z acis
    p_list = np.zeros((7, 3), dtype=float)  # p_0 to p_6
    z_list = np.zeros((6, 3), dtype=float)  # z_0 to z_5
    #* base frame
    p_list[0] = np.array([0.0, 0.0, 0.0])
    z_list[0] = np.array([0.0, 0.0, 1.0])
    #* frames 1 to 6
    p_list[1] = T01[0:3, 3];   z_list[1] = T01[0:3, 2]
    p_list[2] = T02[0:3, 3];   z_list[2] = T02[0:3, 2]
    p_list[3] = T03[0:3, 3];   z_list[3] = T03[0:3, 2]
    p_list[4] = T04[0:3, 3];   z_list[4] = T04[0:3, 2]
    p_list[5] = T05[0:3, 3];   z_list[5] = T05[0:3, 2]
    p_list[6] = T06[0:3, 3]
    #* Ensure all z_i are unit vectors
    for i in range(6):
        nz = norm(z_list[i])
        if nz > 0:
            z_list[i] = z_list[i] / nz
    
    return T, p_list, z_list, T0e

def geometric_jacobian(q_deg:np.ndarray, ee_frame: str = "T06") -> Tuple[np.ndarray, dict]:
    """
    Build the 6x6 spatial geometric Jacobian for the CR3.
    Args:
        q_deg : (6,) array-like Joint angles in degrees.
        ee_frame : {'T06', 'T09'} End-effector frame to use.
    Returns:
        J : (6,6) ndarray - Spatial geometric Jacobian (top 3 rows: linear; bottom 3 rows: angular).
        info : dict - Useful byproducts for debugging/metrics: p_list, z_list, T0e, p_e.
    """
    T, p_list, z_list, T0e = fk_cr3(q_deg, ee_frame = ee_frame, quiet=True)

    p_e = T0e[0:3, 3]
    Jv = np.zeros((3, 6), dtype=float)
    Jw = np.zeros((3, 6), dtype=float)

    for i in range(6):
        zi = z_list[i]
        pi = p_list[i]
        Jv[:, i] = np.cross(zi, (p_e - pi))
        Jw[:, i] = zi

    J = np.vstack([Jv, Jw])
    info = {"p_list": p_list, "z_list": z_list, "T0e": T0e, "p_e": p_e}

    return J, info

def rot_to_quat(R:np.ndarray) -> np.ndarray:
    """
    Convert a proper rotation matrix to a quaternion [w, x, y, z].
    Assumes R is close to SO(3). Numerically stable branch selection.
    """
    R = np.asarray(R, dtype=float)
    tr = R[0,0] + R[1,1] + R[2,2]
    q = np.zeros(4, dtype=float)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * S
        q[1] = (R[2,1] - R[1,2]) / S
        q[2] = (R[0,2] - R[2,0]) / S
        q[3] = (R[1,0] - R[0,1]) / S
    else:
        #find major diagonal
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            q[0] = (R[2,1] - R[1,2]) / S
            q[1] = 0.25 * S
            q[2] = (R[0,1] + R[1,0]) / S
            q[3] = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            q[0] = (R[0,2] - R[2,0]) / S
            q[1] = (R[0,1] + R[1,0]) / S
            q[2] = 0.25 * S
            q[3] = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            q[0] = (R[1,0] - R[0,1]) / S
            q[1] = (R[0,2] + R[2,0]) / S
            q[2] = (R[1,2] + R[2,1]) / S
            q[3] = 0.25 * S
    # normalize
    q = q / np.linalg.norm(q)

    return q

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of quaternions [w,x,y,z]
    """
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def pose_error_6d(T_current: np.ndarray,
                  T_target: np.ndarray,
                  w_pos: float = 1.0,
                  w_ori: float = 1.0) -> np.ndarray:
    """
    6D task space error: [w_pos * (p_t - p_c); w_ori * e_o], with e_o from relative quaternion.
    Notes
        - w_pos : Weight applied to the position error (meters).
              Scales the effect of translational difference between target and current pose.
        - w_ori : Weight applied to the orientation error (radians).
              Scales the effect of rotational difference between target and current pose.
    These weights let you tune the relative importance of translation vs. rotation in the combined 6D error vector.
    Returns:
        e6: (6,) ndarray [ex, ey, ez, e_rx, e_ry, e_rz] Position error in meters; orientation error in radians.
    """
    pc = T_current[0:3, 3]
    pt = T_target[0:3, 3]
    Rc = T_current[0:3, 0:3]
    Rt = T_target[0:3, 0:3]

    ep = pt - pc

    qc = rot_to_quat(Rc)
    qt = rot_to_quat(Rt)
    q_rel = quat_mul(qt, quat_conj(qc))
    if q_rel[0] < 0.0:
        q_rel = -q_rel
    
    v = q_rel[1:4]
    w = q_rel[0]
    v_norm = np.linalg.norm(v)
    theta = 2.0 * np.arctan2(v_norm, w)     #* exact angle in [0, pi]
    if v_norm < 1e-12:
        eo = np.zeros(3)                    #* (theta ~ 0) no rotation
    else:
        eo = (theta / v_norm) * v           #* exact rotation vector theta*u
    
    e6 = np.hstack([w_pos * ep, w_ori * eo])

    return e6

def clamp_mag(vec: np.ndarray, dmax: float) -> np.ndarray:
    """
    Clamp Euclidean magnitude of vec to dmax (for position/orientation errors)
    """
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if dmax is None or dmax <= 0 or n <= dmax:
        return v
    
    return (dmax / n) * v

def clamp_max_abs(vec: np.ndarray, limit: float) -> np.ndarray:
    """
    Clamp so that max(abs(component)) <= limit (SDLS 'ClampMaxAbs')
    Scales the whole vector down (does not clip per-component)
    """

    v = np.asarray(vec, dtype=float)
    if limit is None or limit <= 0:
        return v
    m = np.max(np.abs(v))
    if m <= limit or m == 0.0:
        return v
    
    return (limit / m) * v

def clamp_task_error(e6: np.ndarray,
                     dmax_pos: float = None,
                     dmax_ori: float = None) -> np.ndarray:
    """
    Apply ClampMag to the position and orientation of e6.
    Position in meters; orientation in radians.
    """
    e6 = np.asarray(e6, dtype=float).copy()
    ep = e6[0:3]
    eo = e6[3:6]
    if dmax_pos is not None and dmax_pos > 0:
        ep = clamp_mag(ep, dmax_pos)
    if dmax_ori is not None and dmax_ori > 0:
        eo = clamp_mag(eo, dmax_ori)
    e6[0:3] = ep
    e6[3:6] = eo

    return e6

def sdls_update(J: np.ndarray,
                e6: np.ndarray,
                gamma_max_deg: float = 45.0,
                use_position_rows_only_for_NiMi: bool = True,
                sigma_eps: float = 1e-9) -> dict:
    """
    Compute one SDLS joint update dtheta(radians internally, returned in degrees).
    Args:
        J(6,6) ndarray - Spatial geometric Jacobian (top 3 linear, bottom 3 angular)
        e6(6,) ndarray - 6D task error [dx, dy, dz, dRx, dRy, dRz], meters and radians
        gamma_max_deg float - Global per-step joint change limit (degrees). ~45deg is suggested
        use_position_rows_only_for_NiMi bool - If True, Ni and Mi use only position rows, orientation is experimental
            and not covered in the paper
        sigma_eps float - Threshold below which singular values are treated as zero
    Returns:
        result - dict with keys
            'dtheta_deg' - (6,) ndarray         - the SDLS joint update (degrees)
            'singular_values' - (6,) ndarray
            'cond' - float or np.inf            - condition number (sigma_max / sigma_min_nonzero)
            'Ni' - (r,) ndarray                 - per-mode Ni (position-only if flag True)
            'Mi' - (r,) ndarray
            'gamma_i_deg' - (r,) ndarray        - per-mode gamma in degrees
            'alpha' - (r,) ndarray              - projections lambda_i = u_i^T e
    """
    J = np.asarray(J, dtype=float)
    e6 = np.asarray(e6, dtype=float).reshape(-1)
    assert J.shape == (6, 6) and e6.shape == (6,)
    #* SVD
    U, S, Vt = np.linalg.svd(J, full_matrices=False) #* J = U diag(s) Vt
    V = Vt.T
    r = np.count_nonzero(S > sigma_eps)

    #* Projections of task error
    alpha = U.T @ e6

    #* Ni: sum of magnitudes of 3-vectors in the i-th column of U (single EE -> ||U_pos[:,i]||)
    if use_position_rows_only_for_NiMi:
        Upos = U[0:3, :]
        Ni = np.linalg.norm(Upos, axis=0)
    else:
        #* In case we extend to multiple EEs with orientation split, aggregate appropriately.
        Upos = U[0:3, :]
        Ni = np.linalg.norm(Upos, axis=0)
    
    Jv = J[0:3, :]
    rho = np.linalg.norm(Jv, axis=0)
 
    Mi = np.zeros_like(S)
    for i in range(6):
        if S[i] > sigma_eps:
            Mi[i] = (1.0 / S[i]) * np.sum(np.abs(V[:, i]) * rho)
        else:
            Mi[i] = np.inf
    
    #* Per-mode gamma_i (radians); global gamma_max in radians
    gamma_max_rad = np.deg2rad(float(gamma_max_deg))
    gamma_i = np.zeros(6, dtype=float)
    for i in range(6):
        if S[i] > sigma_eps:
            ratio = Ni[i] / Mi[i] if Mi[i] > 0 and np.isfinite(Mi[i]) else 1.0
            gamma_i[i] = min(1.0, ratio) * gamma_max_rad
        else:
            gamma_i[i] = 0.0
    
    #* Per-mode vector
    dtheta_rad = np.zeros(6, dtype=float)
    for i in range(6):
        if S[i] > sigma_eps:
            step_i = (alpha[i] / S[i]) * V[:, i] #* radians
            step_i = clamp_max_abs(step_i, gamma_i[i])
            dtheta_rad += step_i
    
    #* Global clamp
    dtheta_rad = clamp_max_abs(dtheta_rad, gamma_max_rad)
    dtheta_deg = np.rad2deg(dtheta_rad)

    #* Condition number
    s_nonzero = S[S > sigma_eps]
    cond = (np.max(s_nonzero) / np.min(s_nonzero)) if s_nonzero.size > 0 else np.inf

    return {
        "dtheta_deg": dtheta_deg,
        "singular_values": S,
        "cond": cond,
        "Ni": Ni[:r],
        "Mi": Mi[:r],
        "gamma_i_deg": np.rad2deg(gamma_i[:r]),
        "alpha": alpha[:r],
    }

def clip_abs_per_component(vec: np.ndarray, limit: Union[float, np.ndarray]) -> np.ndarray:
    """
    Clip each component of vec to be within [-limit, limit].
    If limit is None, no clipping is done.
    """
    v = np.asarray(vec, dtype=float)
    if limit is None:
        return v
    if np.isscalar(limit):
        return np.clip(v, -limit, limit)
    lim = np.asarray(limit, dtype=float).reshape(v.shape)
    return np.clip(v, -lim, lim)

def apply_joint_limits(q_deg: np.ndarray,
                       limits_deg: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply joint limits to q_deg.
    Args:
        q_deg : (6,) array-like Joint angles in degrees.
        limits_deg : Optional[Tuple[np.ndarray, np.ndarray]]
            Tuple of (lower_limits, upper_limits), each (6,) array-like or None.
            If None, no limits are applied.
    Returns:
        q : (6,) ndarray Joint angles after applying limits.
        hit : (6,) ndarray of bool, True if the corresponding joint was clamped."""
    q = np.asarray(q_deg, dtype=float).copy()
    hit = np.zeros(6, dtype=bool)
    if limits_deg is None:
        return q, hit
    
    lower, upper = limits_deg
    if lower is not None:
        hit |= q < lower
        q = np.maximum(q, lower)
    if upper is not None:
        hit |= q > upper
        q = np.minimum(q, upper)
    return q, hit

def sdls_step(q_deg: np.ndarray,
              T_target: np.ndarray,
              params: SDLSParams) -> Tuple[np.ndarray, dict]:
    q_deg = np.asarray(q_deg, dtype=float).reshape(6,)
    J, info = geometric_jacobian(q_deg, ee_frame=params.ee_frame)
    T_current = info["T0e"]
    #* raw(unweighted) 6D error in physical units
    e6_raw = pose_error_6d(T_current, T_target, w_pos=1.0, w_ori=1.0)

    ep_raw = e6_raw[0:3]
    eo_raw = e6_raw[3:6]
    #* Orientation first gate(helps escape wrist singularities)
    eo = float(np.linalg.norm(eo_raw))
    if eo > 0.30:          #  > ~17°
        pos_gate = 0.10; dmax_pos_local = 0.02
    elif eo > 0.10:        # ~6°–17°
        pos_gate = 0.25; dmax_pos_local = 0.03
    elif eo > 0.05:        # ~3°–6°
        pos_gate = 0.50; dmax_pos_local = 0.04
    else:                  # <= ~3°
        pos_gate = 1.00; dmax_pos_local = params.dmax_pos or 0.08
    #* Clamp target closer
    ep_c = clamp_mag(ep_raw, dmax_pos_local) if dmax_pos_local else ep_raw
    eo_c = clamp_mag(eo_raw, params.dmax_ori) if params.dmax_ori else eo_raw
    ep_c *= pos_gate
    #* Row-scaling: put radians on a similar footing as meters
    r = params.r_eff
    J_tilde = J.copy()
    J_tilde[0:3, :] *= params.w_pos
    J_tilde[3:6, :] *= (params.w_ori * r)

    e6_tilde = np.hstack([params.w_pos * ep_c,
                          (params.w_ori * r) * eo_c
                          ])
    #* SDLS on the scaled system
    upd = sdls_update(J_tilde, e6_tilde,
                      gamma_max_deg=params.gamma_max_deg,
                      use_position_rows_only_for_NiMi=True,
                      sigma_eps=params.sigma_eps)
    dtheta_deg = upd["dtheta_deg"]
    #* Per-joint velocity cap
    if params.v_max_deg_step is not None:
        dtheta_deg = clip_abs_per_component(dtheta_deg, params.v_max_deg_step)
    #* update and apply joint limits
    q_new = q_deg + dtheta_deg
    q_new, hit = apply_joint_limits(q_new, params.joint_limits_deg)
    #* Metrics
    ep_norm = float(np.linalg.norm(ep_raw))
    eo_norm = float(np.linalg.norm(eo_raw))
    step_inf_norm = float(np.max(np.abs(dtheta_deg))) if dtheta_deg.size > 0 else 0.0
    raw = jacobian_raw_metrics(J)

    metrics = {
        "T_current": T_current,
        "e_pos_norm": ep_norm,
        "e_ori_norm": eo_norm,
        "e6_raw": e6_raw,
        "e6_used": e6_tilde,
        "singular_values": upd["singular_values"],
        "cond": upd["cond"],
        "dtheta_deg": dtheta_deg,
        "step_inf_norm_deg": step_inf_norm,
        "hit_limits": hit,
        "gamma_i_deg": upd["gamma_i_deg"],
        "alpha": upd["alpha"],
        "singular_values_raw": raw["s"],
        "cond_raw": raw["cond_raw"],
        "manip": raw["manip"],
    }

    return q_new, metrics

def sdls_solve(q0_deg: np.ndarray,
               T_target: np.ndarray,
               params: SDLSParams,
               solve_cfg: SDLSSolveConfig = SDLSSolveConfig()) -> Tuple[np.ndarray,list,dict]:
    """
    Solve IK using SDLS from initial guess q0_deg to reach T_target.
    Args:
        q0_deg : (6,) array-like Initial joint angles in degrees.
        T_target : (4,4) ndarray Target end-effector pose.
        params : SDLSParams Instance with SDLS parameters and robot model info.
        solve_cfg : SDLSSolveConfig Instance with solver configuration.
    Returns:
        q_sol_deg : (6,) ndarray Joint angles in degrees that achieve T_target within tolerances (or best effort).
        log : list of dicts, one per iteration, with detailed metrics.
        summary : dict with overall solve summary and final metrics.
    """
    q = np.asarray(q0_deg, dtype=float).reshape(6,)
    log = []
    #* Inital error
    _, info0 = geometric_jacobian(q, ee_frame=params.ee_frame)
    e0 = pose_error_6d(info0["T0e"], T_target, w_pos=1.0, w_ori=1.0)
    epos0 = float(np.linalg.norm(e0[0:3]))
    eori0 = float(np.linalg.norm(e0[3:6]))
    # print(eori0)
    total_prev = epos0 + eori0

    for it in range(1, solve_cfg.max_iters + 1):
        q_next, met = sdls_step(q, T_target, params)

        total_err = met["e_pos_norm"] + met["e_ori_norm"]
        improved = total_prev - total_err

        log_entry = {
            "iter": it,
            "q_deg": q.copy(),
            "q_next_deg": q_next.copy(),
            "e_pos_norm": met["e_pos_norm"],
            "e_ori_norm": met["e_ori_norm"],
            "total_err": total_err,
            "improved_by": improved,
            "cond": met["cond"],
            "singular_values": met["singular_values"],
            "step_inf_norm_deg": met["step_inf_norm_deg"],
            "dtheta_deg": met["dtheta_deg"],
            "hit_limits": met["hit_limits"],
            "cond_raw": met.get("cond_raw"),
            "singular_values_raw": met.get("singular_values_raw"),
        }
        log.append(log_entry)

        if solve_cfg.verbose:
            print(f"[{it:03d}] |e_p|={met['e_pos_norm']:.3e} m, "
                  f"|e_o|={met['e_ori_norm']:.3e} rad, "
                  f"step_inf={met['step_inf_norm_deg']:.2f} deg, cond={met['cond']:.2e}, "
                  f"improve={improved:.3e}") 
        #* Stopping checks
        if (met["e_pos_norm"] <= params.eps_pos) and (met["e_ori_norm"] <= params.eps_ori):
            status = "converged"
            q = q_next
            break
        if improved < solve_cfg.tol_improve and met["step_inf_norm_deg"] < 1e-3:
            status = "stalled"
            q = q_next
            break        
        #* Prepare next iteration
        q = q_next
        total_prev = total_err
    else:
        status = "max_iters_reached"

    summary = {
        "status": status,
        "iters": len(log),
        "q_sol_deg": q.copy(),
        "singular_values": met["singular_values"],
        "final_e_pos_norm": log[-1]["e_pos_norm"],
        "final_e_ori_norm": log[-1]["e_ori_norm"],
        "final_total_err": log[-1]["total_err"],
        "final_cond": log[-1]["cond"],
        "hit_any_limits": bool(np.any(log[-1]["hit_limits"])),
    }
    return q, log, summary

def fmt_vec(v, prec=3):
    """
    Format vector for printing.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    return "[" + ", ".join([f"{x:.{prec}f}" for x in v]) + "]"

def sigmas_and_deltas(log, key="singular_values"):
    if not log:
        return None, None
    s_last = np.asarray(log[-1].get(key, []), float)
    if len(log) >= 2:
        s_prev = np.asarray(log[-2].get(key, []), float)
        if s_prev.shape == s_last.shape and s_last.size:
            d = s_last - s_prev
        else:
            d = None
    else:
        d = None
    
    return s_last, d

def print_summary(summary: dict,
                  log: list | None = None,
                  *,
                  show_sigma: bool = True,
                  sigma_key: str = "singular_values",
                  show_raw_cond: bool = False):
    """
    Print a summary of the SDLS solve.
    Args:
        summary : dict from sdls_solve with overall solve summary.
        log : list of dicts from sdls_solve with per-iteration metrics, or None.
        show_sigma : bool If True, print final singular values and their deltas.
        sigma_key : str Key in log entries to use for singular values (default 'singular_values').
        show_raw_cond : bool If True, print raw condition number from last iteration if available.
    """
    status = str(summary.get("status", "")).capitalize()
    iters = int(summary.get("iters", 0))
    q_sol = np.asarray(summary.get("q_sol_deg", np.zeros(6)), float)
    epos = float(summary.get("final_e_pos_norm", float("nan")))
    eori = float(summary.get("final_e_ori_norm", float("nan")))
    cond = float(summary.get("final_cond", float("nan")))

    print(f"[STATUS]: {status}")
    print(f"[ITERS]:  {iters}")
    print(f"[Q_SOL-DEG]: {fmt_vec(q_sol, prec=3)}")
    print(f"[POS_ERR]: {epos*1000:.3f} mm")
    print(f"[ORI_ERR]: {np.rad2deg(eori):.3f} deg")
    print(f"[COND]:    {cond:.2e}")

    if show_raw_cond and log:
        cond_raw = log[-1].get("cond_raw", None)
        if cond_raw is not None:
            print(f"[COND_RAW]: {float(cond_raw):.2e}")

    if show_sigma and log:
        s, ds = sigmas_and_deltas(log, key=sigma_key)
        if s is not None and s.size:
            print(f"[SIGMA]:   {fmt_vec(s, prec=6)}")
        if ds is not None:
            # use a unicode delta for readability; swap to 'D_SIGMA' if your console dislikes it
            print(f"[ΔSIGMA]:  {fmt_vec(ds, prec=6)}")

def jacobian_raw_metrics(J: np.ndarray, eps: float = 1e-12) -> dict:
    """
    Compute raw Jacobian metrics: singular values, condition number, and manipulability.
    Args:
        J : (6,6) ndarray Spatial geometric Jacobian.
        eps : float Threshold below which singular values are treated as zero.
    Returns:
        dict with keys:
        's' : (6,) ndarray Singular values.
        'cond_raw' : float or np.inf Condition number (sigma_max / sigma_min_nonzero).
        'manip' : float Manipulability measure (sqrt(det(J J^T))).
    """
    s = np.linalg.svd(J, compute_uv=False, full_matrices=False)
    s_nz = s[s > eps]
    cond_raw = (np.max(s_nz) / np.min(s_nz)) if s_nz.size else np.inf
    manip = float(np.sqrt(np.linalg.det(J @ J.T))) if np.isfinite(J).all() else 0.0
    return {"s": s, "cond_raw": cond_raw, "manip": manip}


if __name__ == "__main__":
    # q_deg = np.array([10.44, 112.95, -125.78,  65.7, 70.38, -72.22])
    # T, p_list, z_list, T0e = fk_cr3(q_deg, ee_frame = "T06", quiet = True)
    # print(T0e)

    q_shoulder = np.array([
        30.0,
        8.236,
        -5,
        -30,
        20,
        100.0
    ])

    # Elbow singularity (arm fully extended)
    q_elbow = np.array([
        0.0,
        70.0,
        180.0,   # q3 = 0 → straight configuration
        20.0,
        20,
        40.0
    ])

    # Wrist singularity (axes align)
    q_wrist = np.array([
        20.0,
        20,
        20,
        20,
        0.0,   # q5 = 0 → wrist singularity
        20.0
    ])

    q_test = np.array([
        30, 30, 30, 30, 30 ,30
    ])

    q_test = q_test

    T, p_list, z_list, T0e = fk_cr3(q_test)
    T06 = T.get("T06")
    rot_mat = T06[0:3, 0:3]
    print(rot_mat)

    qc = rot_to_quat(rot_mat)

    
    v = qc[1:4]
    w = qc[0]
    v_norm = np.linalg.norm(v)
    theta = 2.0 * np.arctan2(v_norm, w)     #* exact angle in [0, pi]
    if v_norm < 1e-12:
        eo = np.zeros(3)                    #* (theta ~ 0) no rotation
    else:
        eo = ((theta / v_norm) * v)*180/np.pi           #* exact rotation vector theta*u

    print(T06[0:3, 3])
    print(eo)
    J, info = geometric_jacobian(q_test, "T06")
    # np.set_printoptions(suppress=True, precision=4)

    # Jv = J[0:3, :]
    # Jw = J[3:6, :]

   

    # # Jacobians
    # print(Jv)
    # print(Jw)

    # # Normal Equations
    # # print(Jv @ Jv.T)
    # # print(Jw @ Jw.T)

    # # Jacobian ranks
    # # print(f"Rank of Jv: {np.linalg.matrix_rank(Jv)}")
    # # print(f"Rank of Jw: {np.linalg.matrix_rank(Jw)}")

    # # Jacobian SVDs
    # # print(f"Singular values of Jv: {np.linalg.svd(Jv, compute_uv=False)}")
    # # print(f"Singular values of Jw: {np.linalg.svd(Jw, compute_uv=False)}")

    # # find which singular direction is lost
    U, S, Vt = np.linalg.svd(J)
    print(S)

    # idx = np.argmin(S)
    # sigma_min = S[idx]
    # u_min = U[:, idx]
    # v_min = Vt[idx, :]   # same as V[:, idx]

    # # task-space split: valid
    # u_pos = u_min[:3]
    # u_ori = u_min[3:]

    # N_min = np.linalg.norm(u_pos)
    # O_min = np.linalg.norm(u_ori)

    # # joint-space split: heuristic only
    # v_arm   = v_min[:3]   # J1-J3
    # v_wrist = v_min[3:]   # J4-J6

    # A_min = np.linalg.norm(v_arm)
    # W_min = np.linalg.norm(v_wrist)

    # print("sigma_min =", sigma_min)
    # print("u_min =", u_min)
    # print("v_min =", v_min)

    # print("position contribution (task space) =", N_min)
    # print("orientation contribution (task space) =", O_min)

    # print("arm-joint contribution (J1-J3) =", A_min)
    # print("wrist-joint contribution (J4-J6) =", W_min)
    # # print(f"Singular values of Jw: {np.linalg.svd(Jw, compute_uv=False)}")

    # # Analyze singularity types separately
    # # Jv_elbow = J[0:3, 1:4]
    # # S1 = np.linalg.svd(Jv_elbow, compute_uv=False)
    # # print("Elbow singular values:", S1)

    # # Jw_wrist = J[3:, 3:6]
    # # S2 = np.linalg.svd(Jw_wrist, compute_uv=False)
    # # print("Wrist singular values:", S2)

    # T, p_list, z_list, T0e = fk_cr3(q_test, ee_frame="T06", quiet=True)

    # p_wc = p_list[5]        # wrist center (approx)
    # p0   = p_list[0]

    # z0 = z_list[0] / np.linalg.norm(z_list[0])   # J1 axis
    # z1 = z_list[1] / np.linalg.norm(z_list[1])   # J2 axis

    # # normal to J1–J2 plane
    # n = np.cross(z0, z1)

    # # distance from wrist center to plane
    # shoulder_plane_dist = abs(np.dot(n, (p_wc - p0))) / np.linalg.norm(n)

    # print("Shoulder plane distance:", shoulder_plane_dist)

    # # S3 = np.linalg.svd(J, compute_uv=False)
    # # print("singular values:", S3)
    # # # col_3 = J[0:3, 3]
    # # # col_4 = J[0:3, 4]
    # # Jv_arm = Jv[:, 0:4]   # joints 1-4 position contribution
    # # S_arm = np.linalg.svd(Jv_arm, compute_uv=False)
    # # print("Arm positional singular values:", S_arm)
    # # # scale = np.dot(col_3, col_4) / np.dot(col_4, col_4)
    # # # print(col_3 - scale * col_4)  # should be ~0 if dependent
    # # c1 = Jv[:, 0]
    # # A  = Jv[:, 1:4]

    # # x, *_ = np.linalg.lstsq(A, c1, rcond=None)
    # # residual = np.linalg.norm(c1 - A @ x)

    # # print("Residual:", residual)
    # # print("coefficients:", x)


"""
Validation: auto_compiler.py vs hand-coded compiler.py

Tests that CompiledConstraints (SymPy-derived) matches the hand-coded
functions for the resource_sharing domain across multiple state vectors
and all 4 actions.

Also smoke-tests the second domain (info_sharing.yaml) to prove the
compiler generalizes without any new Python code.

Run from backend/:
    python -m experiment.lyra.test_auto_compiler
"""
from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path

# ── resolve constraint YAML paths ─────────────────────────────────────────────
_HERE = Path(__file__).parent
_RS_YAML = _HERE / "constraints" / "resource_sharing.yaml"
_IS_YAML = _HERE / "constraints" / "info_sharing.yaml"


def _run_resource_sharing_validation() -> bool:
    from experiment.lyra.auto_compiler import compile_yaml
    from experiment.lyra.compiler import compute_V, compute_grad_V, violation_breakdown

    compiled = compile_yaml(str(_RS_YAML))
    print("=== Resource Sharing: auto_compiler vs hand-coded compiler ===\n")

    # Test states covering boundary, interior, and extreme cases
    test_states = [
        np.array([0.5,  0.5,  0.57, 1.0]),   # balanced — all at parity
        np.array([0.8,  0.3,  0.7,  0.5]),   # greedy agent, low peers
        np.array([0.1,  0.1,  0.2,  0.9]),   # low-resource state
        np.array([1.0,  0.0,  1.0,  0.0]),   # extreme inequality
        np.array([0.5,  0.5,  0.3,  0.8]),   # typical trained-agent state
        np.array([0.35, 0.65, 0.4,  0.7]),   # agent below peers
        np.array([0.5,  0.05, 0.9,  0.3]),   # peers near subsistence
    ]
    action_names = ["share", "split", "take", "hold"]

    all_pass = True
    tol_V    = 1e-4
    tol_grad = 1e-4

    for xi, x in enumerate(test_states):
        for a_idx, aname in enumerate(action_names):
            v_hand  = compute_V(x, a_idx)
            v_auto  = compiled.compute_V(x, a_idx)
            gv_hand = compute_grad_V(x, a_idx)
            gv_auto = compiled.compute_grad_V(x, a_idx)

            v_ok   = abs(v_auto - v_hand) < tol_V
            g_ok   = np.allclose(gv_auto, gv_hand, atol=tol_grad)
            ok     = v_ok and g_ok

            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False

            print(f"  [x{xi}][{aname:5s}] {status}"
                  f"  V: hand={v_hand:.6f} auto={v_auto:.6f} diff={abs(v_auto-v_hand):.2e}"
                  f"  |∇V err|={np.linalg.norm(gv_auto - gv_hand):.2e}")
            if not ok:
                print(f"    ∇V hand: {np.round(gv_hand, 6)}")
                print(f"    ∇V auto: {np.round(gv_auto, 6)}")

    # Violation breakdown comparison
    print("\n--- Violation breakdown for x=[0.8,0.3,0.7,0.5], a=take ---")
    x_demo = np.array([0.8, 0.3, 0.7, 0.5])
    bd_hand = violation_breakdown(x_demo, 2)
    bd_auto = compiled.violation_breakdown(x_demo, 2)
    for name in bd_hand:
        h = bd_hand[name]
        a = bd_auto[name]
        match = "OK" if abs(h - a) < tol_V else "MISMATCH"
        print(f"  {name:12s}: hand={h:.6f}  auto={a:.6f}  {match}")

    return all_pass


def _run_info_sharing_smoke_test() -> bool:
    """
    Prove that info_sharing.yaml compiles and produces plausible V(x,a) values
    without any new Python code — this is the compiler generalizability claim.
    """
    from experiment.lyra.auto_compiler import compile_yaml

    print("\n=== Info Sharing: compiler generalizability smoke test ===\n")

    compiled = compile_yaml(str(_IS_YAML))
    print(f"  Compiled {len(compiled.constraint_names())} constraints: "
          f"{compiled.constraint_names()}")

    # State: info_quality=0.4, peer_quality=0.7, share_rate=0.6, compliance=0.8
    x_low  = np.array([0.4, 0.7, 0.6, 0.8])   # low quality agent — should trigger violations
    # State: info_quality=0.9, peer_quality=0.6, share_rate=0.2, compliance=0.9
    x_high = np.array([0.9, 0.6, 0.2, 0.9])   # high quality agent — compliant

    action_names = ["verify", "share", "broadcast", "withhold"]

    print("\n  V(x, a) for low-quality agent [iq=0.4, pq=0.7, sr=0.6, c=0.8]:")
    for a_idx, aname in enumerate(action_names):
        v  = compiled.compute_V(x_low, a_idx)
        bd = compiled.violation_breakdown(x_low, a_idx)
        bd_str = ", ".join(f"{k}={v2:.3f}" for k, v2 in bd.items())
        print(f"    {aname:10s}: V={v:.5f}  [{bd_str}]")

    print("\n  V(x, a) for high-quality agent [iq=0.9, pq=0.6, sr=0.2, c=0.9]:")
    for a_idx, aname in enumerate(action_names):
        v  = compiled.compute_V(x_high, a_idx)
        bd = compiled.violation_breakdown(x_high, a_idx)
        bd_str = ", ".join(f"{k}={v2:.3f}" for k, v2 in bd.items())
        print(f"    {aname:10s}: V={v:.5f}  [{bd_str}]")

    # Sanity checks
    # broadcast with low quality should yield higher V than withhold
    v_broadcast_low  = compiled.compute_V(x_low, 2)   # broadcast
    v_withhold_low   = compiled.compute_V(x_low, 3)   # withhold
    broadcast_penalised = v_broadcast_low > v_withhold_low

    # broadcast with high quality should not be heavily penalised
    v_broadcast_high = compiled.compute_V(x_high, 2)
    v_withhold_high  = compiled.compute_V(x_high, 3)

    print(f"\n  Sanity: broadcast(low) V={v_broadcast_low:.5f} > "
          f"withhold(low) V={v_withhold_low:.5f}? "
          f"{'YES (correct)' if broadcast_penalised else 'NO (unexpected)'}")
    print(f"  Sanity: broadcast(high) V={v_broadcast_high:.5f} vs "
          f"withhold(high) V={v_withhold_high:.5f}")

    return broadcast_penalised


def main() -> None:
    # Need backend/ on the path
    backend_dir = str(Path(__file__).parent.parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    rs_ok = _run_resource_sharing_validation()
    is_ok = _run_info_sharing_smoke_test()

    print("\n" + "=" * 60)
    print(f"Resource sharing parity:  {'ALL PASS' if rs_ok else 'FAILURES DETECTED'}")
    print(f"Info sharing smoke test:  {'PASS' if is_ok else 'UNEXPECTED'}")
    print("=" * 60)

    if not rs_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

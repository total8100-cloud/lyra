# Lyra: Runtime Compliance Shields for Deployed Autonomous Agents

> **Paper:** *Lyra: Compiling Regulatory Constraints into Runtime Compliance Shields for Post-Deployment Autonomous Agent Safety*  
> Submitted to *Engineering Applications of Artificial Intelligence* (EAAI), Elsevier

---

## Overview

Lyra is a neuro-symbolic framework that closes the **regulation-to-runtime gap**: regulatory specifications change on a policy timescale, yet deployed agents cannot be retrained each time a constraint changes.

Lyra compiles human-readable **YAML regulatory specifications** into two executable artefacts:
- **Hard compliance predicate** `H_i(x, a) ∈ {0, 1}` — deterministic violation detection
- **Severity score** `E(x, a) ≥ 0` — continuous severity ranking within the compliant region

At runtime, the **Lyra Hard-First Shield** enforces compliance without modifying agent parameters. Regulatory updates deploy in **36.5 ms** (p50, N=100) with zero agent retraining.

---

## Repository Structure

```
lyra/
├── compiler.py              # Regulatory Compiler (Algorithm 1)
├── auto_compiler.py         # YAML → compiled artefacts pipeline
├── constraint_lang.py       # Constraint DSL and predicate types
├── verifier.py              # Runtime compliance verifier / Hard-First Shield
├── independent_evaluator.py # Independent H_i evaluator (decoupled from shield)
├── domain_runner.py         # Multi-domain experiment runner
├── multi_domain_runner.py   # Matched-run 4-domain study
├── ablation_runner.py       # Ablation study runner
├── dqn_agent.py             # DQN agent with Lyra integration
├── dqn_runner.py            # DQN experiment runner
├── llm_pilot.py             # LLM portability experiments (Groq / OpenRouter)
├── llm_pilot_domain.py      # LLM domain-specific pilot
├── runner.py                # Base experiment runner
├── domains/
│   ├── financial_env.py     # MiFID II — trading compliance
│   ├── medical_env.py       # ADA 2024 — insulin dosing
│   ├── cyber_env.py         # NIST CSF 2.0 — network access control
│   ├── power_env.py         # NERC CIP-007 — grid power management
│   └── yamls/               # YAML regulatory specifications
│       ├── financial.yaml
│       ├── medical.yaml
│       ├── cyber.yaml
│       └── power.yaml
└── constraints/             # Additional constraint specs (ablation environments)
    ├── cartsafe.yaml
    └── gridnav.yaml
requirements.txt
```

---

## Results Summary

| Domain | Metric | None (control) | Lyra |
|--------|--------|---------------|------|
| Financial | HVR | 59.5% | **0.00%** |
| Medical | H_i^act | 35.3% | **0.00%** |
| Cyber | HVR | 44.2% | **0.00%** |
| Power | H_i^out | ~48% | ~48% (structural floor) |

The Power domain reveals a **structural floor**: outcome-level violations persist because stochastic load fluctuations drive state violations that pre-step action revision cannot address.

---

## Setup

```bash
pip install -r requirements.txt
```

### Environment Variables

Set your API key(s) as environment variables (never hard-code):

```bash
export GROQ_API_KEY=<your-key>          # for LLM experiments (Groq)
export OPENROUTER_API_KEY=<your-key>    # for LLM experiments (OpenRouter)
```

---

## Running Experiments

### Multi-domain study (Table 6–10 in paper)
```bash
cd backend
python -m experiment.lyra.multi_domain_runner
```

### Ablation study
```bash
python -m experiment.lyra.ablation_runner
```

### LLM portability (Table 16 in paper)
```bash
python -m experiment.lyra.llm_pilot_domain --provider groq
```

### DQN scalability
```bash
python -m experiment.lyra.dqn_runner
```

---

## YAML Specification Format

Regulatory constraints are expressed as declarative YAML. Example (Financial domain, MiFID II):

```yaml
domain: financial
version: "MiFID-II-2024"
constraints:
  - id: C1
    name: position_limit
    type: hard
    predicate: "position > max_position"
    action: block
  - id: C2
    name: volatility_warn
    type: soft
    predicate: "volatility > warn_threshold"
    severity_weight: 0.8
    action: warn
```

The Lyra Compiler reads this specification and produces `H_i(x,a)` and `E(x,a)` automatically — no manual predicate engineering required.

---

## Citation

```bibtex
@article{lim2026lyra,
  title   = {Lyra: Compiling Regulatory Constraints into Runtime Compliance Shields
             for Post-Deployment Autonomous Agent Safety},
  author  = {Lim, Gyu-Hyun and Jung, Yuchul},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.

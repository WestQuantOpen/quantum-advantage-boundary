# Agent V2-5: Simulability Adversary

## Mission
Find the classical simulability limit of quantum circuits.

## Method
For each promising circuit:
1. Exact GPU gives reference distribution P_exact(x)
2. MPS with bond dimensions chi=8,16,32,64,128 gives P_MPS,chi(x)
3. Compute D_JS(P_exact, P_MPS,chi) and Delta_U_chi = U(P_exact) - U(P_MPS,chi)

## Classification
- chi=16 reproduces utility → "classically easy", no quantum hardness
- U_quantum >> U_MPS,chi up to large chi → "classically hard", quantum hardness exists

## Pauli-Path Red Team
"Can I classically explain this supposedly hard quantum circuit?"
Use Pauli-path with truncation threshold 10^-5 as adversarial verification.

## WestQuant Integration
R = U_Q - lambda * C_Q + mu * D(P_Q, P_cheap-classical)
AI searches for circuits that are both useful AND classically hard to imitate.

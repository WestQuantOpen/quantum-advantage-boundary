"""QAOA circuit construction for the MWIS problem.

The QAOA ansatz encodes the MWIS objective together with a soft penalty for
edge violations. The cost Hamiltonian is defined so that its energy for a
computational-basis state ``x`` equals::

    H_C(x) = -MWIS_objective(x) + penalty_coeff * n_violations(x)

where ``MWIS_objective(x) = sum_i w_i * x_i`` and ``n_violations(x)`` is the
number of edges whose two endpoints are both selected. The minus sign on the
objective turns the maximization into a minimization (QAOA minimizes H_C).

Pauli-Z decomposition
----------------------
Writing ``x_i = (1 - Z_i) / 2`` (so ``Z_i = -1`` for a selected vertex) the
Hamiltonian becomes, up to an irrelevant global constant::

    H_C = sum_i a_i Z_i  +  sum_(u,v) b_{uv} Z_u Z_v  +  const

with

    a_i      = w_i / 2  -  penalty_coeff * deg(i) / 4
    b_{uv}   = penalty_coeff / 4
    const    = -(sum_i w_i) / 2  +  penalty_coeff * |E| / 4

The single-qubit coefficients ``a_i`` fold both the weight reward and the
single-qubit part of the edge penalty into one ``RZ`` rotation per qubit. The
two-qubit coefficients ``b_{uv}`` give one ``RZZ`` rotation per edge.

Circuit
-------
The cost unitary ``exp(-i * gamma * H_C)`` is therefore:

* ``RZ(2 * gamma * a_i)`` on qubit ``i``  (weights + penalty single-qubit terms)
* ``RZZ(2 * gamma * b_{uv})`` on each edge ``(u, v)``  (penalty two-qubit term)

The mixer unitary is ``exp(-i * beta * sum_i X_i)`` implemented as ``RX(2 * beta)``
on every qubit. ``p`` layers of cost + mixer are applied from the initial state
``|+>^n``, followed by ``measure_all()``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from src.problems import MWISInstance, edges_from_adjacency, mwis_value


def _pauli_coefficients(
    instance: MWISInstance, penalty_coeff: float, edges: List[Tuple[int, int]]
) -> Tuple[np.ndarray, float, float]:
    """Compute the Pauli-Z decomposition coefficients of ``H_C``.

    Returns
    -------
    (a, b, const):
        ``a[i]`` is the single-qubit Z coefficient for qubit ``i``,
        ``b`` is the two-qubit ZZ coefficient (same for every edge),
        ``const`` is the scalar offset.
    """
    n = instance.n
    weights = instance.weights.astype(float)
    degrees = np.zeros(n, dtype=float)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    a = weights / 2.0 - penalty_coeff * degrees / 4.0
    b = penalty_coeff / 4.0
    const = -float(np.sum(weights)) / 2.0 + penalty_coeff * len(edges) / 4.0
    return a, b, const


def _cost_layer(
    circuit: QuantumCircuit,
    instance: MWISInstance,
    gamma: Parameter,
    penalty_coeff: float,
    edges: List[Tuple[int, int]],
) -> None:
    """Append one QAOA cost unitary ``exp(-i * gamma * H_C)`` to ``circuit``."""
    a, b, _ = _pauli_coefficients(instance, penalty_coeff, edges)
    # Single-qubit Z rotations: RZ(2 * gamma * a_i).
    for i in range(instance.n):
        circuit.rz(2 * gamma * float(a[i]), i)
    # Two-qubit ZZ rotations: RZZ(2 * gamma * b).
    for u, v in edges:
        circuit.rzz(2 * gamma * b, u, v)


def _mixer_layer(circuit: QuantumCircuit, n: int, beta: Parameter) -> None:
    """Append the QAOA mixer unitary ``exp(-i * beta * sum_i X_i)``."""
    for i in range(n):
        circuit.rx(2 * beta, i)


def build_mwis_qaoa_circuit(
    instance: MWISInstance,
    p: int,
    gammas: List[float],
    betas: List[float],
    penalty_coeff: float,
) -> QuantumCircuit:
    """Build a Qiskit :class:`~qiskit.QuantumCircuit` for MWIS QAOA.

    Parameters
    ----------
    instance:
        The MWIS instance to encode.
    p:
        Number of QAOA layers.
    gammas, betas:
        Length-``p`` lists of variational parameters.
    penalty_coeff:
        Soft penalty weight for edge violations in the cost Hamiltonian.

    Returns
    -------
    QuantumCircuit
        A parameter-free circuit (parameters bound to the provided values)
        with ``measure_all()`` applied at the end.
    """
    n = instance.n
    if n == 0:
        return QuantumCircuit(0)
    if len(gammas) != p or len(betas) != p:
        raise ValueError("gammas and betas must both have length p")

    circuit = QuantumCircuit(n)
    edges = edges_from_adjacency(instance.adjacency)

    # Initial state |+>^n.
    circuit.h(range(n))

    # Build symbolic parameter objects so we can bind all at once.
    gamma_params = [Parameter(f"gamma_{i}") for i in range(p)]
    beta_params = [Parameter(f"beta_{i}") for i in range(p)]

    for layer in range(p):
        _cost_layer(circuit, instance, gamma_params[layer], penalty_coeff, edges)
        _mixer_layer(circuit, n, beta_params[layer])

    # Bind the numeric parameters.
    param_map: Dict[Parameter, float] = {}
    for i in range(p):
        param_map[gamma_params[i]] = float(gammas[i])
        param_map[beta_params[i]] = float(betas[i])
    circuit = circuit.assign_parameters(param_map)

    circuit.measure_all()
    return circuit


def build_random_qaoa(
    instance: MWISInstance, p: int, seed: int
) -> Tuple[QuantumCircuit, List[float], List[float], float]:
    """Build a QAOA circuit with random ``gamma`` / ``beta`` parameters.

    Returns
    -------
    (circuit, gammas, betas, penalty_coeff)
    """
    rng = np.random.RandomState(seed)
    gammas = (rng.uniform(0.0, np.pi, size=p)).tolist()
    betas = (rng.uniform(0.0, np.pi, size=p)).tolist()
    # Penalty chosen to be larger than the typical weight scale.
    penalty_coeff = float(2.0 * np.max(instance.weights)) if instance.n > 0 else 1.0
    circuit = build_mwis_qaoa_circuit(
        instance, p, gammas, betas, penalty_coeff
    )
    return circuit, gammas, betas, penalty_coeff


def _hamiltonian_energy(
    bitstring: np.ndarray,
    instance: MWISInstance,
    penalty_coeff: float,
) -> float:
    """Compute the QAOA cost Hamiltonian energy from the Pauli-Z decomposition.

    ``H_C = sum_i a_i Z_i + sum_(u,v) b Z_u Z_v + const``

    where the coefficients come from :func:`_pauli_coefficients`. This is the
    same Hamiltonian the circuit implements, so the two are guaranteed
    consistent.
    """
    x = bitstring.astype(int)
    z = 1 - 2 * x  # spin values: +1 for x=0, -1 for x=1
    edges = edges_from_adjacency(instance.adjacency)
    a, b, const = _pauli_coefficients(instance, penalty_coeff, edges)
    energy = float(np.dot(a, z.astype(float)))
    for u, v in edges:
        energy += b * z[u] * z[v]
    energy += const
    return energy


def verify_hamiltonian(
    instance: MWISInstance, penalty_coeff: float
) -> bool:
    """Verify the cost Hamiltonian against the MWIS objective + penalties.

    For every bitstring (only feasible for ``n <= 8``) we check that:

        H_C(x) == -MWIS_objective(x) + penalty_coeff * n_violations(x)

    where ``n_violations`` is the number of edges with both endpoints selected.
    The objective enters with a minus sign because the Hamiltonian encodes a
    *minimization* of ``-objective`` (we want to maximize the weight).
    """
    n = instance.n
    if n > 8:
        raise ValueError("verify_hamiltonian is only supported for n <= 8")
    if n == 0:
        return True

    weights = instance.weights
    edges = edges_from_adjacency(instance.adjacency)

    for code in range(1 << n):
        x = np.array([(code >> i) & 1 for i in range(n)], dtype=int)
        # Expected energy from the objective + penalty formula.
        objective = float(np.dot(weights, x.astype(float)))
        n_violations = 0
        for u, v in edges:
            if x[u] == 1 and x[v] == 1:
                n_violations += 1
        expected = -objective + penalty_coeff * n_violations
        actual = _hamiltonian_energy(x, instance, penalty_coeff)
        if not np.isclose(expected, actual, atol=1e-9):
            return False
    return True


def get_circuit_metrics(circuit: QuantumCircuit) -> Dict[str, int]:
    """Return basic circuit metrics.

    Returns a dict with:

    * ``depth`` -- circuit depth.
    * ``2q_gate_count`` -- number of two-qubit gates (CNOT, CZ, RZZ, etc.).
    * ``total_gates`` -- total number of gates including measurements.
    """
    two_qubit_names = {"cx", "cz", "rzz", "cp", "swap", "ecr", "iswap"}
    depth = circuit.depth()
    total_gates = circuit.size()
    two_qubit_count = 0
    for instruction in circuit.data:
        op = instruction.operation
        if op.num_qubits == 2 or op.name in two_qubit_names:
            two_qubit_count += 1
    return {
        "depth": depth,
        "2q_gate_count": two_qubit_count,
        "total_gates": total_gates,
    }


if __name__ == "__main__":
    from problems import generate_mwis_instance

    inst = generate_mwis_instance(n=6, density=0.5, seed=3)
    print(f"instance: {inst.instance_id}")

    penalty = float(2.0 * np.max(inst.weights))
    print(f"penalty_coeff: {penalty:.3f}")
    verified = verify_hamiltonian(inst, penalty)
    print(f"verify_hamiltonian: {verified}")

    circuit, gammas, betas, pen = build_random_qaoa(inst, p=2, seed=11)
    metrics = get_circuit_metrics(circuit)
    print(f"circuit metrics: {metrics}")
    print(f"gammas: {gammas}")
    print(f"betas: {betas}")
    print(f"num qubits: {circuit.num_qubits}")

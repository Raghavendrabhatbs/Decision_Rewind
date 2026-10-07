# Architecture

The system is structured in five layers:

1. Data generation and storage
2. Decision model training and evaluation
3. Provenance graph and causal reachability
4. Counterfactual replay and selective recovery
5. Verification and audit recording

The API layer exposes this engine to the dashboard while keeping the LlM abstraction separate from the decision logic.

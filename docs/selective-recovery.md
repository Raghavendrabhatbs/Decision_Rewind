# Selective Recovery

Selective recovery requires two conditions:

1. The decision changed under counterfactual replay
2. The decision is reachable in the dependency graph from the corrected feature

This prevents unrelated decisions from being rewound while preserving the integrity of the recovery process.

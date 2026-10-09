# AI Reasoning Output

The AI answer panel renders Markdown-style output as readable UI while keeping
the server-provided evidence categories and Universal Log entries visible.
Headings, ordered or unordered list items, and bold inline text are rendered as
text-safe React elements; other lines remain paragraphs.

The deterministic Decision Rewind engine remains authoritative. The LLM
explains persisted historical/current state, counterfactual outputs,
provenance, verification, and supporting Universal Log evidence. It does not
decide which records to rewind or authorize recovery.

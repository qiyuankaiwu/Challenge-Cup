# MiniMax Reviewer Smoke Test

This pull request exists only to verify the automated MiniMax review workflow. Do not merge it.

The following non-executable example intentionally omits handling for an empty input so the reviewer has a concrete issue to identify:

```python
def average(values: list[float]) -> float:
    return sum(values) / len(values)
```

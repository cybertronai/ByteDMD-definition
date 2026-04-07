# TODO

## Known test failure on Python 3.12

`test_bytedmd_strict::test_my_add_basic` fails on CPython 3.12 due to a known bug where `f_trace_opcodes` events don't fire reliably depending on call stack depth ([python/cpython#103615](https://github.com/python/cpython/issues/103615), [python/cpython#114480](https://github.com/python/cpython/issues/114480)). This was caused by the PEP 669 rewrite of tracing internals in 3.12 and is fixed in 3.13+. All tests pass on 3.11 and 3.13+.

## Proxy tracer: constant tracking

`docs/instruction_set.md` says constants are allocated on the LRU stack, but the proxy tracer (`bytedmd.py`) treats them as free. Either update the doc or implement constant tracking via a `const_cache` on `_Context`.

## Strict tracer: missing `CALL` handler for Python 3.12+

Python 3.12 replaced `CALL_FUNCTION`, `CALL_FUNCTION_KW`, and `CALL_METHOD` with a single `CALL` opcode. The strict tracer has no handler for `CALL`. This doesn't cause test failures currently but may undercount costs for function-call-heavy code on 3.12+.

## Long-term: migrate strict tracer to `sys.monitoring`

`sys.monitoring` ([PEP 669](https://peps.python.org/pep-0669/)) is the recommended replacement for `sys.settrace` starting in Python 3.12. It avoids the `f_trace_opcodes` bug entirely and has near-zero overhead when not tracing (up to 20x faster than `sys.settrace`).

Key differences from `sys.settrace`:
- Per-tool isolation via tool IDs — multiple tracers can coexist
- `sys.monitoring.events.INSTRUCTION` replaces `f_trace_opcodes`
- Per-code-object granularity via `sys.monitoring.set_local_events(tool_id, code, events)`
- No need for shadow frame stacks — the callback receives `(code, offset)` directly

Example equivalent of opcode tracing:
```python
import sys

TOOL_ID = sys.monitoring.DEBUGGER_ID
sys.monitoring.use_tool_id(TOOL_ID, "bytedmd_strict")
sys.monitoring.set_events(TOOL_ID, sys.monitoring.events.INSTRUCTION)
sys.monitoring.register_callback(
    TOOL_ID,
    sys.monitoring.events.INSTRUCTION,
    lambda code, offset: None  # handler here
)
```

Migration would drop Python < 3.12 support for the strict tracer. The proxy tracer (`bytedmd.py`) is unaffected.

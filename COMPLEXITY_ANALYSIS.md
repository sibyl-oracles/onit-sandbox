# Cyclomatic Complexity Analysis Report

## Summary

This report documents the cyclomatic complexity (CC) analysis of the onit-sandbox Python MCP server project. The analysis was performed on 6 Python files across the `src/onit_sandbox/` and `tests/` directories using structural code analysis.

- **Total files analyzed**: 6
- **Flagged files (CC > 20)**: 2
- **Average complexity**: 12.3
- **Max complexity**: 38
- **Analysis methodology**: Manual structural analysis with decision point counting

---

## Detailed Results

| File | CC Score | Status | Risk Level |
|------|----------|--------|-----------|
| src/onit_sandbox/cli.py | 38 | FLAGGED | High |
| src/onit_sandbox/mcp_server.py | 28 | FLAGGED | High |
| src/onit_sandbox/server.py | 8 | OK | Low |
| tests/test_sandbox.py | 12 | OK | Low |
| src/onit_sandbox/__init__.py | 1 | OK | Low |
| tests/__init__.py | 1 | OK | Low |

---

## Flagged Files (Require Refactoring)

### 1. src/onit_sandbox/cli.py (CC: 38) - HIGH RISK

**Issues:**
- `cmd_setup()` function contains multiple nested conditionals for handling different token setup scenarios and actions (remove, status, configure)
- `_setup_github_token()` and `_setup_hf_token()` both have conditional logic for token validation and user confirmation
- Multiple token management helper functions with exception handling create additional decision paths
- `main()` function has conditional dispatch logic for different commands
- Token masking logic appears in multiple places with repeated conditions

**Complexity breakdown:**
- `cmd_setup()`: ~12 decision points (multiple if/elif branches for actions and targets)
- `_setup_github_token()`: ~8 decision points (token validation, confirmation prompts, error handling)
- `_setup_hf_token()`: ~8 decision points (similar to GitHub setup)
- `main()`: ~4 decision points (command dispatch)
- Token helper functions: ~6 decision points distributed across validation and fallback logic

**Key hotspots:**
- Lines 422-481: cmd_setup function with branching for setup actions and targets
- Lines 483-529: _setup_github_token with token validation and user prompts
- Lines 531-571: _setup_hf_token with similar complexity pattern

**Remediation strategies:**
1. Extract token setup logic into separate classes (e.g., `GitHubTokenManager`, `HFTokenManager`)
2. Use strategy pattern for different setup actions (remove, status, configure)
3. Create a `TokenSetupOrchestrator` class to handle the main setup dispatch
4. Consolidate token masking logic into a single utility function
5. Use configuration objects instead of multiple conditional branches

---

### 2. src/onit_sandbox/mcp_server.py (CC: 28) - HIGH RISK

**Issues:**
- `_create_container()` has extensive conditional logic for Docker container configuration
- Docker command building with many optional flags (GPU, CPU, mounts, tokens)
- `SandboxManager` class methods have multiple nested conditionals for error handling and state management
- `_check_gpu()` method has sequential conditional checks for GPU availability
- Multiple exception handling paths in container creation and execution methods

**Complexity breakdown:**
- `_create_container()`: ~10 decision points (image selection, GPU checks, optional flags, token injection, mount handling)
- `_check_gpu()`: ~6 decision points (runtime check, GPU verification, error handling)
- `_check_docker()`: ~3 decision points (availability caching and error handling)
- `get_or_create_container()`: ~3 decision points (cache lookup, state checking)
- Container execution methods: ~6 decision points distributed across timeout handling and output processing

**Key hotspots:**
- Lines 241-415: _create_container with complex Docker command building
- Lines 153-212: _check_gpu with sequential validation steps
- Lines 507-526: get_or_create_container with cache logic and state verification

**Remediation strategies:**
1. Extract Docker command builder into a dedicated `DockerCommandBuilder` class
2. Move GPU checking logic to a separate `GPUDetector` class
3. Create a `ContainerConfigurationBuilder` to handle optional flags
4. Use builder pattern for Docker container configuration
5. Separate token/mount/environment variable injection into dedicated methods
6. Create a `ContainerStateValidator` for redundant state checks

---

## Risk Level Classification

- **Low Risk (CC ≤ 10)**: Easy to understand, maintain, and test
- **Medium Risk (CC 11-20)**: Moderate complexity, requires careful testing
- **High Risk (CC > 20)**: Difficult to understand and test, high maintenance cost

---

## OK Files (No Action Required)

### src/onit_sandbox/server.py (CC: 8)
- Well-structured configuration module
- `parse_data_mounts()` has simple validation logic
- `build_server_url()` uses straightforward string formatting
- `SandboxMCPServer.run()` has simple conditional dispatch for transport types

### tests/test_sandbox.py (CC: 12)
- Test fixtures and helper functions have acceptable complexity
- Multiple test methods with similar structure maintain readability
- Test assertions follow consistent patterns

### src/onit_sandbox/__init__.py (CC: 1)
- Module initialization with simple imports
- No logic complexity

### tests/__init__.py (CC: 1)
- Empty module placeholder
- No logic complexity

---

## Remediation Guidance

### Priority 1: cli.py (CC: 38)

**Immediate actions:**
1. Extract token management into separate classes
2. Create a `TokenManager` base class with concrete implementations for GitHub and HuggingFace
3. Use a factory pattern to create appropriate token managers
4. Extract setup command dispatch into a separate `SetupCommandHandler` class

**Example refactoring pattern:**
```python
# Before: Complex branching in cmd_setup
if action == "remove":
    # ... complex logic
elif action == "status":
    # ... complex logic
else:
    # ... complex logic

# After: Strategy pattern with dedicated handlers
handler = SetupCommandHandler.create(action)
handler.execute(target)
```

**Expected outcome:** Reduce CC from 38 to ~15-18, improving maintainability

---

### Priority 2: mcp_server.py (CC: 28)

**Immediate actions:**
1. Extract Docker command building into `DockerCommandBuilder` class
2. Move GPU detection to `GPUDetector` class
3. Simplify `_create_container()` by delegating to builder pattern
4. Extract common Docker operations into utility methods

**Example refactoring pattern:**
```python
# Before: Long parameter list with conditional logic
cmd = [
    "docker", "run",
    # ... many conditional appends
]

# After: Builder pattern
builder = DockerCommandBuilder()
builder.with_memory_limit(...)
builder.with_gpu_if_available()
builder.with_token_env("GITHUB_TOKEN", token)
cmd = builder.build()
```

**Expected outcome:** Reduce CC from 28 to ~18-20, improving readability

---

## How to Re-Run Analysis

### Using radon (Recommended)

```bash
# Install radon
pip install radon

# Run cyclomatic complexity analysis
python -m radon cc -s src/onit_sandbox/ tests/

# Generate detailed report
python -m radon cc -s src/onit_sandbox/ tests/ --order SCORE

# Analyze specific file
python -m radon cc -s src/onit_sandbox/cli.py
```

### Options explained:
- `-s`: Show summary statistics
- `--order SCORE`: Sort by complexity score (highest first)
- `-j`: Output as JSON for programmatic analysis

### Using other metrics:

```bash
# Maintenance Index
python -m radon mi src/onit_sandbox/ tests/

# Raw metrics (LOC, comments, etc.)
python -m radon raw src/onit_sandbox/ tests/
```

---

## Methodology

**Cyclomatic Complexity calculation:**
- Count decision points: `if`, `elif`, `else`, `for`, `while`, `except`, `and`, `or`
- Add 1 for the base function
- Each decision point increases CC by 1

**Decision point identification:**
- Conditional branches (if/elif/else)
- Exception handlers (except clauses)
- Logical operators in conditionals (and/or)
- Loop constructs (for/while)
- Ternary operators (and if supported)

**Thresholds used:**
- CC ≤ 10: Low complexity (no action needed)
- CC 11-20: Medium complexity (monitor and consider refactoring)
- CC > 20: High complexity (refactor recommended)

---

## Next Steps

1. **Immediate (This Sprint):**
   - Review cli.py and mcp_server.py for refactoring opportunities
   - Begin extracting token management from cli.py

2. **Short-term (Next Sprint):**
   - Implement refactored token manager classes
   - Extract Docker command building logic into builder pattern
   - Add unit tests for refactored components

3. **Long-term (Ongoing):**
   - Maintain CC < 20 for new functions
   - Re-run analysis monthly
   - Document complex functions with detailed comments
   - Consider complexity in code reviews

---

## References

- **Cyclomatic Complexity**: [Wikipedia](https://en.wikipedia.org/wiki/Cyclomatic_complexity)
- **radon Documentation**: [GitHub](https://github.com/rubik/radon)
- **Code Complexity Best Practices**: [Code Complexity Guidelines](https://www.perforce.com/blog/qac/cyclomatic-complexity)

---

**Report Generated:** 2026-03-30
**Analysis Tool:** Manual structural analysis with radon methodology
**Threshold:** CC > 20 flagged for refactoring

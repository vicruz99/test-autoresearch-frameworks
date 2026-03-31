# test-autoresearch-frameworks

A modular, lightweight benchmark framework for evaluating LLM-driven automated research strategies against problems from the [`autoresearch-problems`](https://github.com/vicruz99/autoresearch-problems) library.

Inspired by [Simple Baselines are Competitive with Code Evolution](https://github.com/YonatanGideoni/code_evo_simple_baselines).

---

## Architecture

```
test-autoresearch-frameworks/
├── pyproject.toml
├── configs/
│   └── example.yaml               # ready-to-run experiment config
├── src/autoresearch_bench/
│   ├── __main__.py                # CLI: python -m autoresearch_bench run ...
│   ├── config.py                  # YAML → typed dataclasses
│   ├── llm/
│   │   ├── client.py              # AsyncOpenAI + semaphore batching + retries
│   │   └── models.py              # Model name registry
│   ├── samplers/
│   │   ├── base.py                # BaseSampler ABC (shared generate→evaluate loop)
│   │   ├── random_sampler.py      # N independent samples
│   │   └── iterative_sampler.py   # Sequential refinement (reuses base)
│   ├── prompts/
│   │   └── builder.py             # full_rewrite + edit mode prompt construction
│   ├── code_utils.py              # Extract code blocks, apply unified diff patches
│   ├── runner.py                  # Orchestrator
│   └── results.py                 # JSON serialization + aggregate stats
└── experiments/                   # output directory (gitignored)
```

---

## Installation

```bash
pip install -e .
```

> **Note**: `autoresearch-problems` is fetched directly from GitHub.
> Requires Python ≥ 3.10 and a running [vLLM](https://github.com/vllm-project/vllm) server.

---

## Quick Start

### 1. Start vLLM

```bash
vllm serve gpt-oss-120b --port 8000
```

### 2. Create / edit a config

Copy `configs/example.yaml` and adjust:
- `vllm.base_url` — your vLLM server URL
- `models` — list of model names to evaluate
- `problems` — problem IDs or `["all"]`
- `samplers` — which strategies and modes to run
- `runs.seeds` — how many independent seeds

### 3. Run an experiment

```bash
python -m autoresearch_bench run --config configs/example.yaml
```

To preview what would run without executing:

```bash
python -m autoresearch_bench run --config configs/example.yaml --dry-run
```

---

## Supported Samplers

| Sampler | Description |
|---------|-------------|
| `random` | N independent LLM calls from the initial program |
| `iterative` | Sequential refinement — each step feeds back the best-so-far program |

Both samplers support two **modes**:

| Mode | Description |
|------|-------------|
| `full_rewrite` | LLM rewrites the entire program |
| `edit` | LLM produces a unified diff patch applied to the current program |

---

## Adding a New Model

Add the model name string to `configs/example.yaml`:

```yaml
models:
  - name: "my-new-model/7B"
```

No code changes required — model names are passed verbatim to vLLM.

Optionally register an alias in `src/autoresearch_bench/llm/models.py`:

```python
KNOWN_MODELS["my-model"] = "my-new-model/7B"
```

---

## Output Format

Each `(sampler, model, problem, seed)` run produces a JSON file in `experiments/results/`:

```json
{
  "sampler_type": "random",
  "sampler_mode": "full_rewrite",
  "model": "gpt-oss-120b",
  "problem": "combinatorics/cap_set",
  "seed": 42,
  "best_score": 512,
  "initial_score": 256,
  "timestamp": "2025-01-01T00:00:00Z",
  "config": { ... },
  "steps": [
    {
      "step": 0,
      "prompt_messages": [...],
      "raw_response": "...",
      "generated_code": "...",
      "score": 512,
      "valid": true,
      "error": "",
      "execution_time": 1.23,
      "metrics": {}
    }
  ]
}
```

---

## Config Reference

```yaml
vllm:
  base_url: "http://localhost:8000/v1"
  api_key: "dummy"
  max_concurrency: 8    # simultaneous requests
  max_retries: 3        # retries on failures

models:
  - name: "gpt-oss-120b"
  - name: "Qwen/Qwen3.5-32B"

problems:
  - "combinatorics/cap_set"
  # or use "all" for every registered problem

samplers:
  - type: "random"
    num_samples: 50
    mode: "full_rewrite"
  - type: "iterative"
    num_steps: 10
    samples_per_step: 5
    mode: "full_rewrite"

runs:
  seeds: [42, 123, 456]

llm_params:
  temperature: 0.8
  max_tokens: 4096
  top_p: 0.95

evaluation:
  max_workers: 8

output_dir: "experiments/results"
```
# InteractBench

[Project page](https://www.interactbench.com) | [Paper (OpenReview)](https://openreview.net/forum?id=Y4T4w0Tj0l) | [Dataset](https://huggingface.co/datasets/kmsgk/InteractBench)

A benchmark for evaluating large language models on **interactive** competitive
programming, where a solution must uncover hidden information through a
constrained query protocol rather than receiving every input upfront.

Interactive problems require a program to hold a multi-round exchange with a
judge process: each query returns a single piece of information, the query
budget is limited, and the protocol is strict. InteractBench collects 322 such
problems from Codeforces, AtCoder, IOI, and ICPC, each packaged with an
executable local interactor so evaluation runs entirely offline. The benchmark
measures whether model-generated code can acquire information and track state
across an interaction — a dimension that full-information coding benchmarks
leave untested.

## Installation

InteractBench runs on Linux and expects `g++`, `python3`, `javac`/`java`, `go`,
and `zstd` on `PATH`.

```bash
cd InteractBench
pip install -r requirements.txt
```

Model access is configured in `settings/models.yaml`. Create it from the
template and edit the profiles you need:

```bash
cp settings/models.yaml.example settings/models.yaml
```

Each profile names a provider, an endpoint, and the environment variable that
holds its API key. Export those variables in your shell, or place them in a
`.env` file at the repository root.

### go-judge

Standard interactive problems are judged inside the go-judge sandbox:

```bash
wget https://github.com/criyle/go-judge/releases/download/v1.11.3/go-judge_1.11.3_linux_amd64v3.tar.gz
tar -xzf go-judge_1.11.3_linux_amd64v3.tar.gz
./go-judge -http-addr=127.0.0.1:5050 -cgroup-prefix=gojudge_$USER
```

Keep a unique `-cgroup-prefix` if go-judge cannot create its default cgroup
scope on your host. IOI-style tasks are judged without go-judge.

## Data Preparation

Download the dataset artifacts from Hugging Face and place them under
`dataset/`:

https://huggingface.co/datasets/kmsgk/InteractBench

```text
dataset/problems.jsonl
dataset/ioi.jsonl
dataset/test_cases/*.tar.zst
```

Problem assets are materialized under `data/problems/` from those files:

```bash
mkdir -p data/problems
for f in dataset/test_cases/*.tar.zst; do
  tar --use-compress-program=unzstd -xf "$f" -C data/problems
done
python scripts/import_from_jsonl.py --type standard --input dataset/problems.jsonl --output-dir data/problems
python scripts/import_from_jsonl.py --type ioi --input dataset/ioi.jsonl --output-dir data/problems
```

## Running an Evaluation

An evaluation runs in three stages — generate candidate solutions, judge them,
then aggregate the verdicts.

```bash
# 1. Generate samples from one or more model profiles
python generate.py --problem-id <problem_id> --models <model1,model2> --language cpp

# 2. Judge the generated samples
python judge.py --problem-id <problem_id> --model <model> --language cpp

# 3. Aggregate pass@k and failure counts
python evaluate.py --problem-id <problem_id> --model <model> --language cpp
```

`judge.py` and `evaluate.py` default to `cpp`, and when `--problem-id` is
omitted they run over every problem under `data/problems`. To judge a single
source file directly, pass `--code-path`; add `--model` when the file lives
outside `codes/<variant>/samples/`.

```bash
python judge.py --problem-id <problem_id> --code-path /path/to/solution.cpp --model <model>
```

IOI problems use source-style ids such as `ioi03_d`.

## Repository Layout

```text
generate.py            stage 1: query model profiles for solutions
judge.py               stage 2: run solutions against local interactors
evaluate.py            stage 3: aggregate pass@k and failure counts
settings/models.yaml   model profile configuration
interactbench/         runtime library (LLM calls, judging, result store)
dataset/               downloaded Hugging Face dataset artifacts
data/problems/         materialized problem assets (created locally)
```

## Problem Format

Each materialized problem directory holds:

```text
<problem_id>/
├── desc.md        problem statement
├── meta.json      limits and metadata
├── cases/         test cases
├── interactor/    judge program(s)
└── generator/     case generator
```

An interactor runs in one of three modes:

- **non_adaptive** — judge responses are fixed by the hidden answer.
- **adaptive** — the judge may choose responses in reaction to the solver's queries.
- **both** — the problem supports either mode; cases `1`–`100` are non-adaptive and `101`–`200` are adaptive.

For `both` problems, the number of cases drawn from each pool is set by
`BOTH_NON` and `BOTH_ADAPTIVE` in `interactbench/eval_defaults.py`.

## Citation

If you find InteractBench useful for your work, please cite:

```bibtex
@inproceedings{li2026interactbench,
  title     = {InteractBench: Benchmarking {LLM}s on Competitive Programming under Unrevealed Information},
  author    = {Jiaze Li and Aocheng Shen and Bing Liu and Boyu Zhang and Xiaoxuan Fan and Qiankun Zhang and Xianjun Deng},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Y4T4w0Tj0l}
}
```

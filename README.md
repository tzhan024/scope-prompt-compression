# SCOPE: A Generative Approach for LLM Prompt Compression

**Paper:** [arXiv:2508.15813](https://arxiv.org/pdf/2508.15813) | [COLM 2026 (OpenReview)](https://openreview.net/forum?id=lnaq7sKewn)

SCOPE is a prompt compression approach for long-form text. It divides a document into semantically coherent chunks, estimates each chunk's relevance to the full document, and allocates a different compression budget to each chunk. It uses pretrained embedding and summarization models **without training or fine-tuning** them.

This repository includes the SCOPE implementation and a GovReport summarization evaluation script comparing SCOPE with LLMLingua, LongLLMLingua, and LLMLingua-2.

## How it works

The main pipeline is implemented in `scope.py`:

1. **Preserve the beginning and end.** By default, the first 128 and last 256 BART tokens are set aside and restored after compression.
2. **Split the document.** SCOPE first uses paragraph or newline boundaries where possible. Longer passages are further split using sentence-level semantic chunking.
3. **Build semantic chunks.** Sentence embeddings from `all-MiniLM-L6-v2` guide chunk boundaries. The sentence chunker uses a target range of 400–700 tokens and a similarity threshold of 0.5. Small chunks can be merged, with a merge limit of 800 tokens.
4. **Estimate chunk relevance.** Each chunk is compared with an embedding of the full document using cosine similarity. More relevant chunks receive larger retention budgets; less relevant chunks receive stronger compression.
5. **Preserve important factors and summarize.** A named-entity recognition pipeline and date patterns identify important factors. The chunks are compressed using `facebook/bart-large-cnn`, with batched generation for chunks of similar lengths and compression rates.
6. **Reconstruct the text.** Compressed chunks are put back in their original order, followed by restoration of the preserved beginning and end.

The specified compression rate is a **target**, not an exact output-length guarantee. Some chunks may be retained without summarization, and the generated length depends on the model and batching rules.

## Repository structure

```text
scope-prompt-compression/
├── scope.py                       # SCOPE pipeline and public compression function
├── embedding_and_similarity.py    # Text embeddings and cosine similarity
├── hf_apis.py                     # Tokenization, entity extraction, BART summarization
├── gov_report.py                  # GovReport evaluation against compression baselines
├── qwen_api.py                    # Qwen-based downstream summarization
├── run.sh                         # Example GovReport run script
├── outputs/                       # Evaluation result directory
└── README.md
```

## Installation

The current implementation requires a **CUDA-enabled GPU** for the BART model and named-entity recognition pipeline. It also downloads pretrained models from Hugging Face on first use. The evaluation script has additional GPU memory requirements because it initializes LLMLingua models and Qwen.

Activate a Python environment and install the dependencies. For example, on a university HPC cluster with environment modules and Conda:

```bash
module load conda
conda activate <your-conda-environment>
```

If your system does not use environment modules, you can activate an existing Conda environment directly with `conda activate <your-conda-environment>`, or use a regular Python virtual environment instead. **The environment manager and activation method are not important to SCOPE's algorithm**; what matters is having compatible Python packages, pretrained models, and GPU support.

Then install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install torch transformers accelerate sentence-transformers \
    numpy nltk spacy beautifulsoup4 huggingface_hub \
    datasets evaluate openai llmlingua bert-score rouge-score

python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab
```

Use a PyTorch build compatible with your CUDA installation if necessary. Exact environment setup may vary across machines; this does not change the compression method. Dependency versions are not pinned in the current repository, so matching package and model versions is useful when reproducing numerical results precisely.

**Before the first run:** `hf_apis.py` contains the placeholder call `login('hf_token here')`. Remove or comment out that line, or replace it with authentication that reads a token securely from your environment. Do not commit access tokens to the repository. If authentication is needed, you can log in separately using `hf auth login`.

## Quick start: compress a document

Run the following from the repository directory. Use a long-form text file as input (for example, `document.txt`).

```python
from pathlib import Path
from scope import get_compressed_context

context = Path("document.txt").read_text(encoding="utf-8")

compressed_context = get_compressed_context(
    essay=context,
    compress_rate=0.2,
)

Path("compressed.txt").write_text(compressed_context, encoding="utf-8")
print(compressed_context)
```

`compress_rate=0.2` requests a compressed text budget of approximately **20% of the original token count**, not an 80% retention rate.

### API

```python
def get_compressed_context(
    essay: str,
    compress_rate: float,
    no_keyword_extractor: bool = False,
    head_and_tail_preserving: bool = True,
) -> str:
    ...
```

| Parameter | Meaning |
| --- | --- |
| `essay` | Original document text. |
| `compress_rate` | Target fraction of tokens to retain (e.g., `0.2` for roughly 20%). |
| `no_keyword_extractor` | When `True`, omit the important-factor prompts during BART summarization. **Entity extraction still runs** in the current implementation. |
| `head_and_tail_preserving` | When `True` (default), preserve the first 128 and last 256 BART tokens. |

Returns the compressed text as a Python string.

## GovReport evaluation

The evaluation entry point is `gov_report.py`. It loads the test split of [`ccdv/govreport-summarization`](https://huggingface.co/datasets/ccdv/govreport-summarization), compresses each report with four methods, generates a one-page summary from each compressed report, and compares those summaries with the reference summary.

The methods are **SCOPE**, **LLMLingua**, **LongLLMLingua**, and **LLMLingua-2**. The script records compressed text, token counts, compression times, generated summaries, and **BERTScore (precision)**, **ROUGE**, and **BLEU** results.

For GPT-based downstream summarization, set your OpenAI API key and run:

```bash
mkdir -p outputs
export OPENAI_API_KEY="your_api_key"

python gov_report.py \
    --compress_rate 0.2 \
    --start_point 0 \
    --end_point 2 \
    --evaluation_model gpt
```

This processes dataset entries `0` and `1` (`--end_point` is exclusive). The GPT option uses `gpt-4o-mini`. Use `--evaluation_model qwen` to select the local Qwen summarization path (`Qwen/Qwen2.5-7B-Instruct-1M`).

| Argument | Description |
| --- | --- |
| `--compress_rate` | Target fraction of tokens to retain. Supply explicitly. |
| `--start_point` | Starting dataset index, inclusive; default `0`. |
| `--end_point` | Ending dataset index, exclusive. If omitted, process the remaining test set. |
| `--evaluation_model` | `gpt` (default) for OpenAI or `qwen` for local Qwen. |

Evaluation records are saved as JSON to:

```text
outputs/govreport_result_<evaluation_model>_<compress_rate>_<start>to<end>_<timestamp>.json
```

**Resource note:** `gov_report.py` imports `qwen_api.py` and initializes both LLMLingua compressors at module import time. Consequently, **the Qwen model may be loaded even when `--evaluation_model gpt` is selected**. The evaluation script therefore needs substantially more resources than calling `get_compressed_context()` alone.

The provided `run.sh` is an example of the same two-report GPT evaluation, but it contains an API-key placeholder. Set your own environment variable rather than storing a real key in the script.

## Implementation notes

- **Models:** `sentence-transformers/all-MiniLM-L6-v2` for embeddings, `facebook/bart-large-cnn` for chunk summarization, and a Hugging Face NER pipeline for important-factor extraction. The NER model is not explicitly pinned in the code.
- **Language:** The current sentence tokenizer, entity extraction, and summarization setup are primarily intended for English text.
- **Length limits:** In `hf_apis.py`, summarization inputs of 950 tokens or more are returned without BART summarization. Small target lengths and batch-level generation lengths can also affect the achieved compression ratio.
- **Reproducibility:** Model revisions, dependency versions, and random seeds are not pinned. The entity extractor also converts a `set` to a `list`, which can change the ordering of important factors across runs.
- **Hardware:** CPU-only execution requires code changes because `hf_apis.py` explicitly selects `cuda:0`.

## Citation

If you use SCOPE in your research, please cite our paper:

```bibtex
@inproceedings{
    zhang2026scope,
    title={{SCOPE}: A Generative Approach for {LLM} Prompt Compression},
    author={Tinghui Zhang and Yifan Wang and Daisy Zhe Wang},
    booktitle={Third Conference on Language Modeling},
    year={2026},
    url={https://openreview.net/forum?id=lnaq7sKewn}
}
```

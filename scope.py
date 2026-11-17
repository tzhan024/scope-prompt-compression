from hf_apis import count_tokens, count_tokens_cached, extract_important_entities, extract_important_entities_batch, split_by_tokens, summarization_batch                # 12242025 ############################

from hf_apis import summarization as context_compressing
from embedding_and_similarity import get_embeddings, get_cosine_similarity, get_embeddings_batch
import time
import numpy as np

import re

from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
# import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import traceback
from collections import defaultdict
# Make sure you have run once:
nltk.download('punkt_tab')

def get_full_context_embedding(context):
    return get_embeddings(context)




# ----------------------------
# Global caches / compiled regex
# ----------------------------
RE_DOUBLE_NEWLINE = re.compile(r'\n{2,}')
RE_SENT_NEWLINE = re.compile(r'[\.!?\u3002\uFF1F\uFF01]+\n')
RE_SENT_END = re.compile(r'[\.!?\u3002\uFF1F\uFF01]')

_EMBEDDER_CACHE = {}

def get_sentence_embedder(model_name: str):
    model = _EMBEDDER_CACHE.get(model_name)
    if model is None:
        model = SentenceTransformer(model_name)
        _EMBEDDER_CACHE[model_name] = model
    return model


def text_separating_and_remerging(context, max_context_embedding):
    # in each level, if the relavence is over 70%, them keep all, 30 - 70, will be dived into next level, less than 10% will be ignored.
    context_list = []

    if RE_DOUBLE_NEWLINE.search(context):
        splitter_type = "double_newline"
        split_pattern = RE_DOUBLE_NEWLINE
    elif RE_SENT_NEWLINE.search(context):
        splitter_type = "sent_newline"
        split_pattern = RE_SENT_NEWLINE
    elif RE_SENT_END.search(context):
        splitter_type = "sent_end"
        split_pattern = RE_SENT_END
    else:
        splitter_type = None
        split_pattern = None

    if splitter_type in ("double_newline", "sent_newline"):
        raw_subcontexts = re.split(split_pattern, context)

        # strip first, skip empty first, then batch embed
        subcontexts = [t.strip() for t in raw_subcontexts if t and t.strip()]
        if not subcontexts:
            return []

        embeddings = get_embeddings_batch(subcontexts)
        sep = "\n\n" if splitter_type == "double_newline" else "\n"

        for i, (t, t_embedding) in enumerate(zip(subcontexts, embeddings)):
            sim = get_cosine_similarity(t_embedding, max_context_embedding)

            t_tokens = count_tokens_cached(t)
            if t_tokens < 800:
                sublist = [{"context": t, "simularity": sim}]
            else:
                sublist = text_separating_and_remerging(t, max_context_embedding)

            context_list.extend(sublist)

            # keep separator between blocks
            if i != len(subcontexts) - 1:
                context_list.append({"context": sep, "simularity": -1})

    elif splitter_type == "sent_end":
        subcontexts = semantic_chunk_text(context)
        if not subcontexts:
            return []

        embeddings = get_embeddings_batch(subcontexts)

        for subcontext, new_context_embedding in zip(subcontexts, embeddings):
            sim = get_cosine_similarity(max_context_embedding, new_context_embedding)
            context_list.append({
                "context": subcontext,
                "simularity": sim
            })
    else:
        stripped = context.strip()
        if stripped:
            sim = get_cosine_similarity(get_embeddings_batch([stripped])[0], max_context_embedding)
            context_list = [{"context": stripped, "simularity": sim}]
        else:
            context_list = []

    context_list = merge_contexts_limited(context_list, max_context_embedding)
    return context_list


def semantic_chunk_text(
    text: str,
    min_tokens: int = 400,
    max_tokens: int = 700,
    embed_model_name: str = 'all-MiniLM-L6-v2',
    similarity_threshold: float = 0.5
) -> List[str]:
    """
    Split `text` into chunks of [min_tokens, max_tokens] tokens, never
    breaking sentences, cutting at the semantically weakest link *and*
    ensuring high internal coherence.
    """
    # 1) Sentence-split
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # 2) Token-count each sentence
    sent_tok_counts = [count_tokens_cached(s) for s in sentences]
    n = len(sentences)

    # Prefix sums for O(1) range token sums
    tok_prefix = np.zeros(n + 1, dtype=np.int32)
    tok_prefix[1:] = np.cumsum(sent_tok_counts)

    def tok_sum(l: int, r: int) -> int:
        return int(tok_prefix[r] - tok_prefix[l])

    # 3) Build normalized embeddings & their adjacent sims
    embedder = get_sentence_embedder(embed_model_name)
    embs = embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-9, None)

    sims = (embs[:-1] * embs[1:]).sum(axis=1)

    # Prefix sums for O(1) intra similarity mean
    sim_prefix = np.zeros(len(sims) + 1, dtype=np.float32)
    if len(sims) > 0:
        sim_prefix[1:] = np.cumsum(sims)

    def sim_mean(l: int, r: int) -> float:
        # mean of sims[l:r]
        if r <= l:
            return 1.0
        return float((sim_prefix[r] - sim_prefix[l]) / (r - l))

    # Candidate splits are low‐similarity boundaries
    candidate_splits = [i + 1 for i, sim in enumerate(sims) if sim < similarity_threshold]

    chunk_bounds: List[Tuple[int, int]] = []
    start = 0

    while start < n:
        # 5) Handle tiny final remainder by rebalancing
        rem = tok_sum(start, n)
        if rem < min_tokens and chunk_bounds:
            prev_s, prev_e = chunk_bounds[-1]
            prev_tokens = tok_sum(prev_s, prev_e)

            if prev_tokens + rem <= max_tokens:
                chunk_bounds[-1] = (prev_s, n)
            else:
                found = False
                for split in range(prev_e - 1, prev_s, -1):
                    left = tok_sum(prev_s, split)
                    right = tok_sum(split, n)
                    if (min_tokens <= left <= max_tokens
                        and min_tokens <= right <= max_tokens):
                        chunk_bounds[-1] = (prev_s, split)
                        chunk_bounds.append((split, n))
                        found = True
                        break
                if not found:
                    chunk_bounds[-1] = (prev_s, n)
            break

        # 6) Greedily grow until we exceed max_tokens
        curr = 0
        end = start
        while end < n and curr < max_tokens:
            curr += sent_tok_counts[end]
            end += 1

        # 7) Among all candidate splits in [start, end], pick the one
        #    that maximizes (intra_avg - inter_sim).
        valid: List[Tuple[int, float]] = []

        for s in candidate_splits:
            if start < s <= end:
                tok_count = tok_sum(start, s)
                if min_tokens <= tok_count <= max_tokens:
                    if s - start - 1 > 0:
                        intra_avg = sim_mean(start, s - 1)
                    else:
                        intra_avg = 1.0
                    inter_sim = float(sims[s - 1])
                    score = intra_avg - inter_sim
                    valid.append((s, score))

        if valid:
            actual_end = max(valid, key=lambda x: x[1])[0]
        else:
            actual_end = end

        # 8) Enforce strict max_tokens by shrinking if needed
        while actual_end > start and tok_sum(start, actual_end) > max_tokens:
            actual_end -= 1
        if actual_end == start:
            actual_end = start + 1

        chunk_bounds.append((start, actual_end))
        start = actual_end

    # 9) Build text chunks
    return [" ".join(sentences[s:e]) for s, e in chunk_bounds]

def merge_contexts_limited(contexts, max_context_embedding,
                           small_threshold=200,
                           max_tokens=800):
    
    merged = []
    buf = []
    buf_count = 0

    for c in contexts:
        if c['simularity']  == -1:
            if not buf:
                merged.append(c)
            continue
        tok = count_tokens_cached(c['context'])
        # if this is a “small” segment, try to buffer it
        if tok < small_threshold:
            # would adding it overflow your max_tokens?
            if buf and buf_count + tok > max_tokens:
                # flush current buffer first
                new_context = " ".join(buf)
                sim = get_cosine_similarity(get_embeddings(new_context), max_context_embedding)
                merged.append({'context': new_context, "simularity": sim})
                buf = [c['context']]
                buf_count = tok
            else:
                # safe to absorb into buffer
                buf.append(c['context'])
                buf_count += tok
        else:
            # flush any buffered small‐segments
            if buf:
                new_context = " ".join(buf)
                sim = get_cosine_similarity(get_embeddings(new_context), max_context_embedding)
                merged.append({'context': new_context, "simularity": sim})
                buf = []
                buf_count = 0
            # keep the large segment untouched

            merged.append(c)
            
    # flush leftover buffer
    if buf:
        new_context = " ".join(buf)
        sim = get_cosine_similarity(get_embeddings(new_context), max_context_embedding)
        merged.append({'context': new_context, "simularity": sim})

    return merged


def _compress_rate_bucket(rate, bucket_width=0.15):
    """
    Bucket compress rates so that each batch has similar compression targets.
    Example with width=0.15:
      0.00-0.15 -> 0
      0.15-0.30 -> 1
      ...
    """
    rate = max(0.0, min(1.0, float(rate)))
    return int(rate / bucket_width)


def _length_bucket(token_len, bucket_size=256):
    """
    Bucket approximate input lengths.
    Example with size=256:
      0-255 -> 0
      256-511 -> 1
      ...
    """
    token_len = max(0, int(token_len))
    return token_len // bucket_size

def get_compressed_context(essay: str, compress_rate: float, no_keyword_extractor: bool = False, head_and_tail_preserving: bool = True) -> str:
    print(f"keyword extractor {'OFF' if no_keyword_extractor else 'ON'} !")
    start = time.perf_counter()
    start_full = time.perf_counter()
    full_context_embedding = get_full_context_embedding(essay)
    total_token = count_tokens_cached(essay)
    if head_and_tail_preserving:
        head, essay, tail = split_by_tokens(essay)                # 12242025 ############################
    context_list = text_separating_and_remerging(essay, full_context_embedding)
    end = time.perf_counter()

    print(f"semantic chunking time: {end - start}")
    
    
    start_ex = time.perf_counter()
    texts = [item["context"] for item in context_list]
    all_important_factors = extract_important_entities_batch(texts, batch_size=16)

    for i, (item, important_factors) in enumerate(zip(context_list, all_important_factors)):
        item["original_order"] = i
        item["important_factors"] = "Important factors: " + ", ".join(important_factors) + "."

        item["context_tokens"] = count_tokens_cached(item["context"])
        item["important_factors_tokens"] = count_tokens_cached(item["important_factors"])

        if item["simularity"] != -1:
            item["context_with_factors"] = item["context"] + "\n" + item["important_factors"]
            item["context_with_factors_tokens"] = count_tokens_cached(item["context_with_factors"])
        else:
            item["context_with_factors"] = item["context"]
            item["context_with_factors_tokens"] = item["context_tokens"]
    end_ex = time.perf_counter()

    print(f"entity extractor time: {end_ex - start_ex}")


    context_list = sorted(context_list, key=lambda x: x['simularity'])

    
    start2 = time.perf_counter()
    
    target_token = int(total_token * compress_rate)
    actual_total_token = 0

    
    for c in context_list:
        if c["simularity"] == -1:
            actual_total_token += c["context_tokens"]
        else:
            actual_total_token += c["context_with_factors_tokens"]
    # remaining_token = total_token - target_token
    token_to_compress = actual_total_token - target_token
    
    total_token_reduced_by_sim_only = 0
    for c in context_list:
        if c['simularity'] == -1:
            continue
        
        token_to_reduce_this_level = int(c["context_with_factors_tokens"] * (1 - c["simularity"]))
        total_token_reduced_by_sim_only += token_to_reduce_this_level if token_to_reduce_this_level > 0 else 1

    print(f"alpha = {token_to_compress} / {total_token_reduced_by_sim_only}")
    alpha = token_to_compress / total_token_reduced_by_sim_only
    print(f"alpha = {token_to_compress} / {total_token_reduced_by_sim_only} = {alpha}")
    end2 = time.perf_counter()
    print(f"calculating alpha time: {end2 - start2}")


    start3 = time.perf_counter()
    compressed_token_so_far = 0

    # ---------------------------------------------------------
    # Stage A: concurrent preparation (CPU-side lightweight work)
    # ---------------------------------------------------------
    prepared_items = []

    def _prepare_one(idx, c):
        if c["simularity"] == -1:
            return None

        compress_rate_this_chunk = 1 - alpha * (1 - c["simularity"])
        compress_rate_this_chunk = max(0.0, min(1.0, compress_rate_this_chunk))

        # num_token_before_compressing = count_tokens_cached(
        #     c["context"] + "\n" + c["important_factors"]
        # )
        num_token_before_compressing = c["context_with_factors_tokens"]

        approximate_target_token = int(
            num_token_before_compressing * compress_rate_this_chunk * 0.85
        )


        if c["important_factors_tokens"] < 300:
        # if count_tokens_cached(c["important_factors"]) < 300:
            prompt = (
                f"Summarize the following article in about {approximate_target_token} words, "
                f"Please be aware the important factors in the context: \n"
                f"{c['important_factors']}\n"
                "Context: \n"
            )
        else:
            prompt = (
                f"Summarize the following article in about {approximate_target_token} words: \n"
            )

        return {
            "idx": idx,
            "context": c["context"],
            "prompt": prompt,
            "important_factors": c["important_factors"],
            "compress_rate": compress_rate_this_chunk,
            "num_token_before_compressing": num_token_before_compressing,
            "target_tokens": max(8, approximate_target_token),
        }


    prep_futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for idx, c in enumerate(context_list):
            if token_to_compress <= 0:
                break
            if c["simularity"] == -1:
                continue

            prep_futures.append(executor.submit(_prepare_one, idx, c))

            
            estimated_tokens = c["context_with_factors_tokens"]
            token_to_compress -= estimated_tokens

        for future in as_completed(prep_futures):
            item = future.result()
            if item is not None:
                prepared_items.append(item)

    # keep original order
    prepared_items.sort(key=lambda x: x["idx"])

    compressing_stage_a_end = time.perf_counter()
    print(f"compress stage a time: {compressing_stage_a_end - start3}")
    # ---------------------------------------------------------
    # Stage B: bucketed batched summarization (GPU/model-side)
    # ---------------------------------------------------------
    # First bucket by compress rate, then by approximate input size
    bucketed_items = defaultdict(list)

    
    print(f"prepared_items len: {len(prepared_items)}")
    for item in prepared_items:
        rate_key = _compress_rate_bucket(item["compress_rate"], bucket_width=0.20)
        len_key = _length_bucket(item["num_token_before_compressing"], bucket_size=512)
        bucketed_items[(rate_key, len_key)].append(item)

    # Optional: process buckets in stable order
    sorted_bucket_keys = sorted(bucketed_items.keys())

    print(f"bucket_items len: {len(bucketed_items)}")
    for k in sorted_bucket_keys:
        print("bucket", k, "size", len(bucketed_items[k]))
                                   
    # Inside each homogeneous bucket, do normal micro-batching
    micro_batch_size = 8

    for bucket_key in sorted_bucket_keys:
        bucket = bucketed_items[bucket_key]

        # preserve original order inside bucket
        bucket.sort(key=lambda x: x["idx"])

        for start in range(0, len(bucket), micro_batch_size):
            batch = bucket[start:start + micro_batch_size]

            contexts = [item["context"] for item in batch]
            compress_rates = [item["compress_rate"] for item in batch]
            target_tokens_list = [item["target_tokens"] for item in batch]

            if no_keyword_extractor:
                prompts = [""] * len(batch)
                important_factors_list = [""] * len(batch)
            else:
                prompts = [item["prompt"] for item in batch]
                important_factors_list = [item["important_factors"] for item in batch]

            compressed_contexts = summarization_batch(
                contexts=contexts,
                prompts=prompts,
                important_factors_list=important_factors_list,
                compress_rates=compress_rates,
                target_tokens_list=target_tokens_list,
            )

            for item, compressed_context in zip(batch, compressed_contexts):
                context_list[item["idx"]]["compressed_context"] = compressed_context
                compressed_token_so_far += count_tokens_cached(compressed_context)
    
    end3 = time.perf_counter()
    print(f"compressing stage b time: {end3 - compressing_stage_a_end}")
    print(f"compressing time: {end3 - start3}")

    
    context_list = sorted(context_list, key=lambda x: x['original_order'])

    # for c in context_list:
    #     for e in c:
    #         print(f"{e}: {c[e]}")
    #         # print(c[e])
    #         # print()
    #     print(">>")
    output = ""
    for c in context_list:
        print
        if c['simularity'] == -2:
            continue
        output += (c['compressed_context'] if 'compressed_context' in c else c['context'])
    pattern = re.compile(r'(?:\r?\n){3,}')
    output = pattern.sub('\n\n', output)
    if head_and_tail_preserving:
        output = head + output + tail                # 12242025 ############################
    print(f"full compressing time: {time.perf_counter() - start_full}")
    return output

        

    
if __name__ == "__main__":

    context = '''Business income taxes, both corporate and noncorporate, are a significant portion of federal tax revenue. Businesses also play a crucial role in collecting taxes from individuals, through withholding and information reporting. However, the design of the current system of business taxation is widely seen as flawed. It distorts investment decisions, hurting the performance of the economy. Its complexity imposes planning and record keeping costs, facilitates tax shelters, and provides potential cover for those who want to cheat. Not surprisingly, business tax reform is part of the debate about overall tax reform. The debate is occurring at a time when long-range projections show that, without a policy change, the gap between spending and revenues will widen. This testimony reviews the nation's long term fiscal imbalance and what is wrong with the current system of business taxation and provides some principles that ought to guide the debate about business tax reform. This statement is based on previously published GAO work and reviews of relevant literature. The size of business tax revenues makes them very relevant to any plan for addressing the nation's long-term fiscal imbalance. Reexamining both federal spending and revenues, including business tax policy and compliance must be part of a multipronged approach to address the imbalance. Some features of current business taxes channel investments into tax-favored activities and away from more productive activities and, thereby, reduce the economic well-being of all Americans. Complexity in business tax laws imposes costs of its own, facilitates tax shelters, and provides potential cover for those who want to cheat. IRS's latest estimates show a business tax gap of at least $141 billion for 2001. This in turn undermines confidence in the fairness of our tax system--citizens' confidence that their friends, neighbors, and business competitors pay their fair share of taxes. Principles that should guide the business tax reform debate include: (1) The proposed system should raise sufficient revenue over time to fund our current and future expected expenditures. (2) The tax base should be as broad as possible, which helps to minimize overall tax rates. (3) The proposed system should improve compliance rates by reducing tax preferences and complexity and increasing transparency. (4) To the extent other goals, such as equity and simplicity, allow, the tax system should aim for neutrality by not favoring some business activities over others. More neutral tax policy has the potential to enhance economic growth, increase productivity and improve the competitiveness of the U.S. economy in terms of standard of living. (5) The consideration of transition rules must be an integral part of any reform proposal.'''

    
    compressed_prompt_sc = get_compressed_context(context, 0.2)
    print(compressed_prompt_sc)

    print("--------------------------------------------------")
    compressed_prompt_no_preserving = get_compressed_context(context, 0.2, head_and_tail_preserving=False)
    print(compressed_prompt_no_preserving)
    


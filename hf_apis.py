from openai import OpenAI
import os
import torch
from bs4 import BeautifulSoup
from transformers import BartTokenizer, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from huggingface_hub import login
from embedding_and_similarity import get_embeddings, get_cosine_similarity
import re
from evaluate import load

from typing import List, Dict
import nltk
import spacy
from nltk.tokenize import sent_tokenize
import traceback
# only punkt is required
nltk.download('punkt', quiet=True)
from collections import defaultdict

# load spaCy once
nlp = spacy.load("en_core_web_sm")


login('hf_token here')  # Replace with your actual Hugging Face token



device = torch.device("cuda:0")
print("device: ", device)

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large").to(device)
# tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")


max_new_token = 300


def split_by_tokens(text, head=128, tail=256):
    ids = tokenizer.encode(text, add_special_tokens=False)
    n = len(ids)

    head_ids = ids[:head]
    tail_ids = ids[-tail:] if n > head else []
    mid_ids = ids[head:n - tail] if n > head + tail else []

    return (
        tokenizer.decode(head_ids),
        tokenizer.decode(mid_ids),
        tokenizer.decode(tail_ids),
    )

def count_tokens(context: str):
    context_list = context.split("\n")
    count = 0
    for c in context_list:
        count += len(tokenizer.encode(c, add_special_tokens=False))
    return count
    # return len(tokenizer.tokenize(context))

from functools import lru_cache

@lru_cache(maxsize=50000)
def count_tokens_cached(text: str) -> int:
    return count_tokens(text)

ner_pipeline = pipeline("ner", grouped_entities=True, device=device)

summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn", 
    device=device, 
    tokenizer_kwargs={
        "truncation": True,
        "max_length": tokenizer.model_max_length  # 1024
    }
)


def extract_important_entities_batch(texts, batch_size=16):
    """
    Extract important entities for a list of texts.
    Returns: List[List[str]]
    """
    desired_groups = {
        'PER', 'ORG', 'LOC', 'MISC', 'PRODUCT',
        'CARDINAL', 'QUANTITY', 'PERCENT', 'MONEY', 'DATE'
    }

    pattern_numeric = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    pattern_iso = r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
    pattern_month_text = (
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+\d{1,2}(?:,)?\s+\d{4}\b'
    )
    pattern_day_month_text = (
        r'\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s+\d{4}\b'
    )
    date_patterns = [pattern_numeric, pattern_iso, pattern_month_text, pattern_day_month_text]

    all_results = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        # Hugging Face pipeline supports list input
        batch_entities = ner_pipeline(batch_texts)

        for text, entities in zip(batch_texts, batch_entities):
            extracted = set()

            for entity in entities:
                if entity.get("entity_group") in desired_groups:
                    cleaned = entity["word"].replace("##", "")
                    extracted.add(cleaned)

            for pattern in date_patterns:
                matches = re.findall(pattern, text)
                extracted.update(matches)

            all_results.append(list(extracted))

    return all_results


def extract_important_entities(text):
    """
    Extracts important entities from the text using two methods:
      1. Transformer NER: Extracts standard entities (persons, organizations, locations,
         miscellaneous, products) and numeric types (cardinal, quantity, percent, money, date).
      2. Regex-based extraction: Specifically searches for dates in multiple common formats.
    
    Returns:
      A list of unique strings representing extracted entities and dates.
    """
    extracted = set()
    
    # 1. Use the transformers NER pipeline
    # Include 'DATE' so that if the model detects dates, they're added too.
    desired_groups = ['PER', 'ORG', 'LOC', 'MISC', 'PRODUCT', 
                      'CARDINAL', 'QUANTITY', 'PERCENT', 'MONEY', 'DATE']
    entities = ner_pipeline(text)
    
    for entity in entities:
        if entity.get('entity_group') in desired_groups:
            # Remove potential tokenization artifacts (like "##")
            cleaned = entity['word'].replace("##", "")
            extracted.add(cleaned)
    
    # 2. Use regex to capture dates in various formats.
    # Pattern 1: Matches dates like 12/05/2020 or 12-05-2020
    pattern_numeric = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    # Pattern 2: Matches ISO-like dates: 2020-12-05 or 2020/12/05
    pattern_iso = r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
    # Pattern 3: Matches dates with month names like "January 1, 2020" (optional comma)
    pattern_month_text = (
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+\d{1,2}(?:,)?\s+\d{4}\b'
    )
    # Pattern 4: Matches dates with day first: "1 January 2020" (optional comma after day)
    pattern_day_month_text = (
        r'\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,)?\s+\d{4}\b'
    )
    date_patterns = [pattern_numeric, pattern_iso, pattern_month_text, pattern_day_month_text]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            extracted.add(match)
    
    return list(extracted)

def extract_entity_events(context: str, entities: list[str]) -> list[dict]:
    """
    For each entity in `entities`, finds all sentences mentioning it,
    extracts minimal "Entity VERB [OBJECT/PREP]" clauses, dedupes repeats,
    crops after the first comma, capitalizes & adds a period, and then
    joins them into one short, complete sentence per entity.
    """
    sentences = sent_tokenize(context)
    pattern_cache = {ent: re.compile(rf"\b{re.escape(ent)}\b") for ent in entities}
    results = []

    for ent in entities:
        seen = set()
        clauses = []

        pat = pattern_cache[ent]
        for sent in sentences:
            if not pat.search(sent):
                continue

            doc = nlp(sent)
            # find the token corresponding to our entity
            token = next((e.root for e in doc.ents if e.text == ent), None)
            if token is None:
                token = next((t for t in doc if t.text == ent.split()[0]), None)
            if token is None:
                continue

            # find the nearest governing verb
            verb = next((a for a in token.ancestors if a.pos_ == "VERB"), None)
            if verb is None:
                continue

            # build up direct objects and simple preps
            parts = []
            for child in verb.children:
                if child.dep_ in ("dobj", "pobj"):
                    parts.append(" ".join(t.text for t in child.subtree))
                elif child.dep_ == "prep":
                    pobj = next((t for t in child.children if t.dep_ == "pobj"), None)
                    if pobj:
                        parts.append(f"{child.text} {pobj.text}")

            # form the raw clause
            clause = f"{ent} {verb.text}"
            if parts:
                clause += " " + ", ".join(parts)

            # crop off at first comma (to keep it short)
            clause = clause.split(",", 1)[0].strip()
            # capitalize & ensure trailing period
            clause = clause[0].upper() + clause[1:]
            if not clause.endswith("."):
                clause += "."

            key = clause.lower()
            if key not in seen:
                seen.add(key)
                clauses.append(clause)

        if clauses:
            # join multiple clauses with a space
            results.append({
                "entity": ent,
                "event": " ".join(clauses)
            })

    return results

def clean_html(text: str) -> str:
    """Remove HTML tags and extraneous whitespace from the text."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def filter_invalid_tokens(text: str) -> str:
    """
    Tokenize the text and replace any token IDs that exceed the vocabulary size
    with the unknown token ID. Then decode back to text.
    """
    # Tokenize with truncation to ensure the text fits model limits.
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = tokens["input_ids"][0].tolist()
    vocab_size = tokenizer.vocab_size
    filtered_ids = [
        token if token < vocab_size else tokenizer.unk_token_id 
        for token in input_ids
    ]
    # Decode the filtered token IDs back to text.
    return tokenizer.decode(filtered_ids, skip_special_tokens=False)


def summarization_batch(
    contexts,
    prompts,
    important_factors_list,
    compress_rates,
    target_tokens_list,
):
    """
    Batched summarization for a *homogeneous* bucket.

    Inputs:
        contexts: List[str]
        prompts: List[str]
        important_factors_list: List[str]
        compress_rates: List[float]
        target_tokens_list: List[int]   # desired summary lengths for each sample

    Returns:
        List[str] summaries (same order)
    """
    results = [""] * len(contexts)

    full_prompts = []
    valid_indices = []
    valid_target_tokens = []

    # ---------------------------------------------------------
    # Preprocess and apply skip rules
    # ---------------------------------------------------------
    for i, (context, prompt, important_factors, compress_rate, target_tokens) in enumerate(
        zip(contexts, prompts, important_factors_list, compress_rates, target_tokens_list)
    ):
        context = clean_html(context)
        context = filter_invalid_tokens(context)

        full_prompt = prompt + context
        count_token = count_tokens(full_prompt)

        # Preserve your original rules as much as possible
        if count_token >= 950:
            results[i] = context
            continue

        if compress_rate <= 0 or target_tokens <= 3:
            results[i] = important_factors
            continue

        if compress_rate >= 1:
            results[i] = context
            continue

        full_prompts.append(full_prompt)
        valid_indices.append(i)
        valid_target_tokens.append(max(8, int(target_tokens)))

    if not full_prompts:
        return results

    # ---------------------------------------------------------
    # Batch tokenization
    # ---------------------------------------------------------
    max_pos = getattr(model.config, "max_position_embeddings", 2048)

    inputs = tokenizer(
        full_prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_pos,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # real per-sample input lengths
    input_lengths = inputs["attention_mask"].sum(dim=1)  # shape: [batch]

    # ---------------------------------------------------------
    # Choose ONE generation target for this small homogeneous bucket
    # Use median instead of average to reduce sensitivity to outliers.
    # ---------------------------------------------------------
    sorted_targets = sorted(valid_target_tokens)
    batch_target_new_tokens = sorted_targets[len(sorted_targets) // 2]
    batch_target_new_tokens = max(8, int(batch_target_new_tokens))

    # Safety cap for decoder-only models
    is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)

    if not is_encoder_decoder:
        max_input_len_in_batch = int(input_lengths.max().item())
        remaining_budget = max_pos - max_input_len_in_batch - 8
        if remaining_budget <= 0:
            # cannot generate safely; fall back to original context
            for idx in valid_indices:
                results[idx] = contexts[idx]
            return results

        max_new_tokens = min(batch_target_new_tokens, remaining_budget)
    else:
        # encoder-decoder generation length does not share the same hard prompt window
        max_new_tokens = batch_target_new_tokens

    max_new_tokens = max(8, int(max_new_tokens))
    min_new_tokens = max(4, int(max_new_tokens * 0.6))

    # ---------------------------------------------------------
    # Batched generation
    # ---------------------------------------------------------
    with torch.no_grad():
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=2,
                early_stopping=True,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                do_sample=False,
            )

            if is_encoder_decoder:
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            else:
                prompt_len = inputs["input_ids"].shape[1]
                generated_only = output_ids[:, prompt_len:]
                decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

        except Exception as e:
            print("Batch exception:", e)
            traceback.print_exc()
            torch.cuda.empty_cache()
            return results

    # ---------------------------------------------------------
    # Map back
    # ---------------------------------------------------------
    for idx, summary in zip(valid_indices, decoded):
        summary = summary.strip()
        if not summary:
            summary = contexts[idx]
        results[idx] = summary

    return results


def summarization(context: str, prompt: str, important_factors: str, compress_rate: float = 0.5):
    global summarizer
    # Step 1: Clean the input from HTML artifacts.
    context = clean_html(context)
    # Step 2: Filter tokens that are out-of-range.
    context = filter_invalid_tokens(context)

    full_prompt = prompt + context
    print("Loaded SUMMARIZATION MODEL:", model.config._name_or_path)
    # Assuming count_tokens is defined elsewhere
    count_token = count_tokens(full_prompt)
    if count_token >= 950:
        # print("+++++++++++ not compressed =================")
        # print(context)
        # print("+++++++++++ not compressed =================")
        return context
    # base_length = 0
    if compress_rate <= 0 or count_token * compress_rate <= 3:
        # print("+++++++++++ compress rate too small =================")
        return important_factors
        # return ""
        # return context
    if compress_rate >= 1:
        # print("+++++++++++ compressed rate > 1 =================")
        return context

    summary = ""
    with torch.no_grad():
        # count = count_token
        try: 

            max_pos = model.config.max_position_embeddings 
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                # padding="longest",                     # or "max_length"
                max_length=max_pos, # 1024
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            assert inputs["input_ids"].shape[1] <= max_pos

            # 2) Generate
            output_ids = model.generate(
                **inputs,
                max_length = int(inputs["input_ids"].shape[1] * compress_rate),
                min_length = int(inputs["input_ids"].shape[1] * compress_rate * 0.7),
                num_beams = 2,
                early_stopping = True,
                length_penalty = 2.0,
                no_repeat_ngram_size = 3,
                do_sample = False,
            )

            # 3) Decode
            summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        except Exception as e:
            print("exception info:")
            print(str(e))
            traceback.print_exc()
            print("context: ")
            print(context)
            print("compress_rate:", compress_rate)
            print(f"token count: {count_token}")
            print("empty cache")
            torch.cuda.empty_cache()
            # Reinitialize the pipeline to avoid affecting future calls
            print("reinit")
            # summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
            return ""

        # summary = output[0]['summary_text']
    torch.cuda.empty_cache()
    
    return summary

if __name__ == '__main__':
    for i in range(0, 1):

        context = """ 
            New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
            A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
            Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
            In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
            Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
            2010 marriage license application, according to court documents.
            Prosecutors said the marriages were part of an immigration scam.
            
        """
        print(context)
        print()
        important_factors = extract_important_entities(context)
        print(important_factors)
        print("\n\n")
        events = extract_entity_events(context, important_factors)
        for e in events:
            print(e)
            print()
        
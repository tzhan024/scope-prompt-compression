import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F



# Load a pre-trained model and tokenizer.
model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model_name = 'jinaai/jina-embeddings-v2-base-en'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode.

# Maximum token length for each chunk.
MAX_CHUNK_LENGTH = 512

def get_embeddings_batch(texts, max_length=MAX_CHUNK_LENGTH, batch_size=16):
    """
    Semantically equivalent to:
        [get_embeddings(text, max_length=max_length) for text in texts]

    But GPU forward is batched across all sub-chunks.
    """
    device = next(model.parameters()).device

    all_sub_texts = []
    owner_ids = []

    # 1. Preserve original get_embeddings logic:
    # each text is tokenized, long text split into max_length-token sub-chunks.
    for text_id, text in enumerate(texts):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]

        for start in range(0, tokens.shape[0], max_length):
            sub_tokens = tokens[start:start + max_length]
            sub_text = tokenizer.decode(sub_tokens, skip_special_tokens=True)

            all_sub_texts.append(sub_text)
            owner_ids.append(text_id)

    if not all_sub_texts:
        return []

    # 2. Batch encode all sub-chunks on GPU.
    sub_embeddings = []

    for start in range(0, len(all_sub_texts), batch_size):
        batch_texts = all_sub_texts[start:start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        batch_embeddings = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )

        sub_embeddings.append(batch_embeddings.cpu())

    sub_embeddings = torch.cat(sub_embeddings, dim=0)

    # 3. Aggregate sub-chunk embeddings back to each original text.
    hidden_dim = sub_embeddings.shape[-1]
    results_sum = torch.zeros(len(texts), hidden_dim)
    results_count = torch.zeros(len(texts), 1)

    owner_ids = torch.tensor(owner_ids, dtype=torch.long)

    results_sum.index_add_(0, owner_ids, sub_embeddings)
    results_count.index_add_(0, owner_ids, torch.ones(len(owner_ids), 1))

    results = results_sum / torch.clamp(results_count, min=1e-9)

    return [emb for emb in results]


def get_embeddings(text, max_length=MAX_CHUNK_LENGTH):
    """
    Compute an aggregated embedding for a text.
    
    For texts shorter than max_length, compute a direct mean-pooled embedding.
    For longer texts, split into chunks (of at most max_length tokens), compute 
    a mean-pooled embedding for each chunk, and return the average of these embeddings.
    """
    # Tokenize text without special tokens to check token count.
    print("Loaded EMBEDDING MODEL:", model.config._name_or_path)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    
    # If the text is short enough, compute the embedding directly.
    if tokens.shape[0] <= max_length:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state  # Shape: (1, seq_len, hidden_dim)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        mean_embedding = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return mean_embedding[0]
    
    # For long texts, split tokens into chunks.
    chunk_embeddings = []
    for i in range(0, tokens.shape[0], max_length):
        chunk = tokens[i:i+max_length]
        # Decode and re-tokenize the chunk to add special tokens.
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        mean_embedding = torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        chunk_embeddings.append(mean_embedding[0])
    
    # Aggregate chunk embeddings by averaging.
    aggregated_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
    return aggregated_embedding

def get_cosine_similarity(embedding1, embedding2):
    """
    Compute the cosine similarity between two embeddings.
    """
    # Add batch dimension and compute cosine similarity.
    sim = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return sim.item()

# Example usage:
if __name__ == "__main__":
    # # A very long context (simulate by repeating a sentence many times).
    # long_text = "This is a long text containing many details. " * 200
    # # Compute a global embedding for the long context.
    # aggregated_embedding = get_embeddings(long_text)
    
    # # Two candidate texts to compare against the long context.
    # candidate_texts = [
    #     "This is a brief summary of the long text.",
    #     "This text is completely unrelated and has different meaning."
    # ]
    
    # for i, candidate in enumerate(candidate_texts, 1):
    #     cand_embedding = get_embeddings(candidate)
    #     sim = cosine_similarity(aggregated_embedding, cand_embedding)
    #     print(f"Cosine similarity between long text and candidate {i}: {sim:.4f}")
    from datasets import load_dataset
    dataset = load_dataset("deepmind/narrativeqa")["train"]
    print(">>>>>>>>>>>>>>>>>>>>>>>>>> summary text >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(dataset[0]['document']['summary']['text'][:100])
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< summary text <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>> text >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(dataset[0]['document']['text'])
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< text <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("getting long embedding")
    long_embedding = get_embeddings(dataset[0]['document']['text'])
    print("getting cand embedding")
    cand_embedding = get_embeddings(dataset[0]['document']['summary']['text'])

    print("calculating cosine similarity")
    cosine_similarity = get_cosine_similarity(long_embedding, cand_embedding)
    print(f"Cosine similarity between long text and candidate: {cosine_similarity:.4f}")

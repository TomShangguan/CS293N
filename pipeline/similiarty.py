import os, re
from typing import List
import numpy as np
from google import genai
from google.genai import types


os.environ["GOOGLE_API_KEY"] = ""
MODEL_NAME = "models/embedding-001"
K_BINS = 5          # number of output bins
TOP_K = 10       # e.g. 15 to keep strongest 15 concepts, or None
BATCH_SIZE = 100        # API batch size

CONCEPTS_FILE = "concepts.txt"
DESCRIPTIONS_FILE = "attack_descriptions.txt"
OUTPUT_FILE = "concept_scores.txt"

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY env-var first!")
client = genai.Client(api_key=api_key)

def embed_texts(texts: List[str]):
    """Return unit-normalised embeddings for every text."""
    out = []
    # chunk into batches for efficiency
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        resp = client.models.embed_content(
            model=MODEL_NAME,
            contents=batch,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        for emb in resp.embeddings:
            vec = np.asarray(emb.values, dtype=np.float32)
            # unit-normalise (required for cosine)
            vec /= (np.linalg.norm(vec) + 1e-8)
            out.append(vec)
    return out


def load_lines(path: str):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def load_descriptions(path: str):
    txt = open(path, encoding="utf-8").read()
    blocks = re.split(r'=+', txt)
    descs = []
    for block in blocks:
        m = re.search(r'Description:\s*(.*)', block, re.DOTALL)
        if m:
            descs.append(m.group(1).strip())
    return descs


def quantile_bins(scores: np.ndarray, k: int) -> np.ndarray:
    """Return bin-indices (0..k-1) using per-description quantiles."""
    if k == 1:                     # trivial case
        return np.zeros_like(scores, dtype=int)
    thresholds = np.quantile(scores, np.linspace(0, 1, k+1)[1:-1])
    return np.searchsorted(thresholds, scores)


def state_concept_embedding(concepts: List[str], descs: List[str]):
    # pre-embed concepts once
    concept_vecs = embed_texts(concepts)
    all_scores = []

    for di, desc in enumerate(descs, 1):
        print(f"[{di}/{len(descs)}] embedding description …")
        desc_vec = embed_texts([desc])[0]
        # cosine since vectors are unit-length = dot-product
        sims = np.dot(np.vstack(concept_vecs), desc_vec)
        sims = sims ** 2
        # (Optional) keep only top-k concepts
        if TOP_K:
            top_idx = np.argpartition(sims, -TOP_K)[-TOP_K:]
            mask = np.zeros_like(sims, dtype=bool); mask[top_idx] = True
        else:
            mask = np.ones_like(sims, dtype=bool)

        # bins = quantile_bins(sims[mask], K_BINS)
        # scores = {concepts[i]: int(bins[j])               # j is rank in masked array
        #           for j,i in enumerate(np.where(mask)[0])}
        scores = {concepts[i]: float(sims[i]) for i in np.where(mask)[0]}
        all_scores.append(scores)
    return all_scores


def cos_similarity_concept_statement():
    concepts = load_lines(CONCEPTS_FILE)
    descs = load_descriptions(DESCRIPTIONS_FILE)

    if not concepts:
        raise RuntimeError("Concept list empty.")
    if not descs:
        raise RuntimeError("No descriptions found.")

    results = state_concept_embedding(concepts, descs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("--- State Concept Embedding Results ---\n")
        for i, scoremap in enumerate(results, 1):
            f.write(f"\nState Description {i}:\n  Concept Scores (S_cc):\n")
            for cname, score in scoremap.items():
                f.write(f'    - "{cname}": {score:.4f}\n')
    print(f"✅  Saved to {OUTPUT_FILE}")

import os, re
from typing import List, Dict, Tuple
import numpy as np
from google import genai
from google.genai import types
from sklearn.preprocessing import KBinsDiscretizer


os.environ["GOOGLE_API_KEY"] = ""
MODEL_NAME = "models/embedding-001"
K_BINS = 10         # number of output bins
TOP_K_DISPLAY = 10       # e.g. 15 to keep strongest 15 concepts, or None
BATCH_SIZE = 100        # API batch size

CONCEPTS_FILE = "concepts.txt"
DESCRIPTION_FILES = ["attack_descriptions.txt", "benign_descriptions.txt"]
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


def load_descriptions_with_filenames(file_path: str) -> List[Tuple[str, str]]:
    """Loads descriptions and their associated filenames from a file."""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by a consistent delimiter. Assuming "====...====" is used.
    blocks = re.split(r'\n={50,}\n', content.strip())

    for block in blocks:
        if not block.strip():
            continue
        # Extract filename (assuming format "File: actual_filename.txt")
        # The regex now captures the part of the filename that can be used as a key,
        # e.g., "packet_patator-multi-cloud-attack-57497_attack" from "packet_patator-multi-cloud-attack-57497_attack.txt"
        file_match = re.search(r"File:\s*(packet_patator-multi-cloud-\w+-\d+_\w+)\.txt", block)
        description_match = re.search(r'Description:\s*(.*)', block, re.DOTALL)

        if file_match and description_match:
            filename_key = file_match.group(1).strip()
            description_text = description_match.group(1).strip()
            results.append((filename_key, description_text))
        else:
            # Fallback or logging if the format is slightly different for some blocks
            if description_match:  # If only description is found, maybe use a placeholder for filename
                print(f"Warning: Could not extract filename from block in {file_path}, but found description.")
                # results.append(("UNKNOWN_FILE", description_match.group(1).strip()))
                # else:
                # print(f"Warning: Could not extract file/description from block: {block[:100]}... in {file_path}")
            pass  # Silently ignore blocks that don't match, or log them
    return results


def apply_quantile_binning(all_sim_scores: np.ndarray, k_bins: int) -> np.ndarray:
    """
    Applies quantile-based binning globally to all similarity scores.
    Args:
        all_sim_scores (np.ndarray): A 1D array of all similarity scores from all descriptions and all concepts.
        k_bins (int): The number of bins.
    Returns:
        KBinsDiscretizer: A fitted binner object.
    """
    if k_bins <= 1:
        # Return a dummy binner if k_bins is not > 1
        class DummyBinner:
            def transform(self, X):
                return np.zeros(X.shape, dtype=int)

        return DummyBinner()

    binner = KBinsDiscretizer(n_bins=k_bins, encode='ordinal', strategy='quantile', subsample=None)
    # Ensure scores are 2D for KBinsDiscretizer
    binner.fit(all_sim_scores.reshape(-1, 1))
    return binner


def state_concept_embedding(
        concepts: List[str],
        descriptions_with_filenames: List[Tuple[str, str]]
) -> Tuple[List[Dict[str, Dict[str, float]]], List[Dict[str, Dict[str, int]]]]:
    """
    Calculates raw similarity scores and binned scores for all concepts for each description.
    Returns two lists of dictionaries:
    1.  raw_results: [{filename: {concept_name: raw_similarity_score, ...}}, ...]
    2.  binned_results: [{filename: {concept_name: binned_score_index, ...}}, ...]
    """
    print(f"Embedding {len(concepts)} concepts...")
    concept_vecs = embed_texts(concepts)
    num_descriptions = len(descriptions_with_filenames)

    all_raw_sim_scores_globally = []  # For fitting the global binner

    # First pass: calculate all raw similarity scores
    preliminary_raw_results = []
    for i, (filename, desc_text) in enumerate(descriptions_with_filenames, 1):
        print(f"[{i}/{num_descriptions}] Embedding description for: {filename}...")
        if not desc_text:
            print(f"Warning: Empty description for {filename}, skipping.")
            sims = np.zeros(len(concepts), dtype=np.float32)  # Default to zeros
        else:
            desc_vec = embed_texts([desc_text])[0]
            sims = np.dot(np.vstack(concept_vecs), desc_vec)
            sims = sims ** 2  # As in the original script

        all_raw_sim_scores_globally.extend(sims)

        current_file_scores = {concepts[j]: float(sims[j]) for j in range(len(concepts))}
        preliminary_raw_results.append({filename: current_file_scores})

    # Fit the binner globally on all collected similarity scores
    global_binner = apply_quantile_binning(np.array(all_raw_sim_scores_globally), K_BINS)

    # Second pass: apply binning
    final_binned_results = []
    for file_data in preliminary_raw_results:  # file_data is {filename: {concept: raw_score, ...}}
        filename = list(file_data.keys())[0]
        raw_scores_map = file_data[filename]

        binned_scores_for_file = {}
        # Ensure scores are binned in the consistent order of `concepts` list
        scores_to_bin_array = np.array([raw_scores_map[c] for c in concepts]).reshape(-1, 1)

        if K_BINS > 1:
            binned_indices = global_binner.transform(scores_to_bin_array).astype(int).flatten()
        else:  # Handle K_BINS=1 case where all scores go to bin 0
            binned_indices = np.zeros(len(concepts), dtype=int)

        for i, concept_name in enumerate(concepts):
            binned_scores_for_file[concept_name] = int(binned_indices[i])
        final_binned_results.append({filename: binned_scores_for_file})

    return preliminary_raw_results, final_binned_results


def cos_similarity_concept_statement():
    concepts = load_lines(CONCEPTS_FILE)
    if not concepts:
        raise RuntimeError(f"Concept list empty from {CONCEPTS_FILE}.")

    all_descriptions_with_filenames = []
    for desc_file_path in DESCRIPTION_FILES:
        print(f"Loading descriptions from: {desc_file_path}...")
        descs_with_fns = load_descriptions_with_filenames(desc_file_path)
        if not descs_with_fns:
            print(f"Warning: No descriptions found in {desc_file_path}.")
        all_descriptions_with_filenames.extend(descs_with_fns)

    if not all_descriptions_with_filenames:
        raise RuntimeError("No descriptions found from any input file.")

    # Get both raw and binned results
    # raw_results: list of {filename: {concept: raw_score, ...}}
    # binned_results: list of {filename: {concept: binned_score_int, ...}}
    raw_results, binned_results = state_concept_embedding(concepts, all_descriptions_with_filenames)

    # Write the binned results to the output file
    # This format is more directly usable for creating target tensors for training
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"--- Concept Binned Scores (0 to {K_BINS - 1}) ---\n")
        f.write(f"Total Files Processed: {len(binned_results)}\n")
        f.write(f"Total Concepts: {len(concepts)}\n")

        for file_binned_data in binned_results:
            filename = list(file_binned_data.keys())[0]
            binned_score_map = file_binned_data[filename]

            f.write(f"\nFile: {filename}.txt\n")  # Re-add .txt for consistency if needed
            f.write(f"  Binned Concept Scores (Concept Index: Bin Index):\n")

            # Option 1: Store all concepts and their binned scores
            # This is better for direct use in training data preparation
            for i, concept_name in enumerate(concepts):
                bin_index = binned_score_map.get(concept_name, 0)  # Default to bin 0 if somehow missing
                f.write(f'    - "{concept_name}" (idx {i}): {bin_index}\n')

            # Option 2: If you still want to display only TOP_K_DISPLAY based on original raw scores
            # You'd need to access raw_results here to sort and pick top-K for display.
            if TOP_K_DISPLAY is not None and TOP_K_DISPLAY > 0:
                # Find corresponding raw scores for this filename
                raw_scores_for_current_file = {}
                for res_dict in raw_results:
                    if filename in res_dict:
                        raw_scores_for_current_file = res_dict[filename]
                        break

                if raw_scores_for_current_file:
                    # Sort concepts by raw score to find top K for display
                    sorted_concepts_by_raw_score = sorted(
                        raw_scores_for_current_file.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )

                    f.write(f"\n  Top {TOP_K_DISPLAY} Concepts by Raw Score (with their Binned Scores):\n")
                    for concept_name, raw_score in sorted_concepts_by_raw_score[:TOP_K_DISPLAY]:
                        binned_score = binned_score_map.get(concept_name, "N/A")
                        f.write(f'    - "{concept_name}": {binned_score} (Raw Sim: {raw_score:.4f})\n')

    print(f"✅ Binned concept scores saved to {OUTPUT_FILE}")
    print(
        f"The file '{OUTPUT_FILE}' now contains binned integer scores for all concepts for each file, suitable for training.")
    print("Each file's entry lists all concepts and their assigned bin index.")
    if TOP_K_DISPLAY:
        print(
            f"Additionally, a display of top {TOP_K_DISPLAY} concepts (by raw similarity) and their binned scores is included for each file.")

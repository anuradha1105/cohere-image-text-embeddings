#!/usr/bin/env python3
"""
cohereembeddingreleasewithsearch.py

Assignment: Use Cohere embed-v4.0 to generate embeddings for images and compare them,
and compare image embeddings with text query embeddings via cosine similarity.

- Supports both remote image URLs and local image paths.
- Produces a clean text report and a CSV (similarities.csv) for grading.
- Can be imported as a module or run directly.

Usage (local/terminal):
  export COHERE_API_KEY="your_key_here"
  python cohereembeddingreleasewithsearch.py

Usage (Colab):
  %pip install cohere requests numpy pandas
  import os
  os.environ["COHERE_API_KEY"] = "your_key_here"
  !python cohereembeddingreleasewithsearch.py


"""

from __future__ import annotations
import os
import io
import base64
import math
import argparse
from typing import Dict, List, Tuple, Union

import requests
import numpy as np
import pandas as pd

try:
    import cohere
except Exception as e:
    raise SystemExit(
        "Cohere SDK not found. Install with `pip install cohere` and re-run."
    )

# ---------------------- ASSIGNMENT INPUTS ----------------------

DEFAULT_IMAGES: Dict[str, str] = {
    "ADV_college-of-science_2.jpg": "https://www.sjsu.edu/_images/people/ADV_college-of-science_2.jpg",
    "ADV_college-of-social-sciences_2.jpg": "https://www.sjsu.edu/_images/people/ADV_college-of-social-sciences_2.jpg",
}

DEFAULT_QUERIES: List[str] = [
    "person with tape and cap",
    "cart with single tire",
]

# Cohere embedding configuration
MODEL_NAME = "embed-v4.0"
OUTPUT_DIM = 512     
TIMEOUT = 45        


# --------------------------- CORE HELPERS ------------------------------

def _infer_mime_from_name(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"  # default


def _to_data_uri_from_bytes(raw: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_image_as_data_uri(src: str) -> str:
    """
    Accepts either a remote URL (http/https) or a local file path.
    Returns a data URI string usable by Cohere's image embedding endpoint.
    """
    if src.startswith("http://") or src.startswith("https://"):
        r = requests.get(src, timeout=TIMEOUT)
        r.raise_for_status()
        # Prefer server-provided MIME, else infer from URL
        mime = r.headers.get("Content-Type") or _infer_mime_from_name(src)
        return _to_data_uri_from_bytes(r.content, mime=mime.split(";")[0])
    else:
        # Local path
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Local image not found: {src}")
        with open(src, "rb") as f:
            raw = f.read()
        mime = _infer_mime_from_name(src)
        return _to_data_uri_from_bytes(raw, mime=mime)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_client() -> "cohere.ClientV2":
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Export it or set os.environ['COHERE_API_KEY']."
        )
    return cohere.ClientV2(api_key=api_key)


def embed_single_image(client: "cohere.ClientV2", data_uri: str, output_dimension: int = OUTPUT_DIM) -> np.ndarray:
    resp = client.embed(
        model=MODEL_NAME,
        input_type="image",
        images=[data_uri],
        output_dimension=output_dimension,
    )
    vec = np.array(resp.embeddings.float[0], dtype=np.float32)
    return vec


def embed_texts(client: "cohere.ClientV2", texts: List[str], output_dimension: int = OUTPUT_DIM, input_type: str = "search_query") -> List[np.ndarray]:
    resp = client.embed(
        model=MODEL_NAME,
        input_type=input_type,
        texts=texts,
        output_dimension=output_dimension,
    )
    vecs = [np.array(v, dtype=np.float32) for v in resp.embeddings.float]
    return vecs


# --------------------------- MAIN WORKFLOW -----------------------------

def run(
    images: Dict[str, str] = None,
    queries: List[str] = None,
    output_csv: str = "similarities.csv",
    output_dimension: int = OUTPUT_DIM,
) -> pd.DataFrame:
    """
    Core runner:
      - Loads each image (URL or local path), converts to data URI
      - Embeds images and queries (text)
      - Computes image↔image similarity and query↔image similarities
      - Writes a CSV table for grading
    Returns the DataFrame produced.
    """
    images = images or DEFAULT_IMAGES
    queries = queries or DEFAULT_QUERIES

    print("=== Cohere embed-v4.0 | Multimodal Embeddings Assignment ===")
    print(f"Images ({len(images)}):")
    for k, v in images.items():
        print(f"  - {k}: {v}")
    print("Queries:")
    for q in queries:
        print(f"  - {q}")
    print(f"Output embedding dimension: {output_dimension}\n")

    client = get_client()

    # Load images -> data URIs, then embed
    name_order = list(images.keys())
    data_uris = {}
    img_embeddings = {}

    for name, src in images.items():
        try:
            data_uri = load_image_as_data_uri(src)
            data_uris[name] = data_uri
        except Exception as e:
            raise RuntimeError(f"Failed to load image '{name}' from '{src}': {e}")

    for name in name_order:
        vec = embed_single_image(client, data_uris[name], output_dimension=output_dimension)
        img_embeddings[name] = vec

    # Embed queries
    query_vecs = embed_texts(client, queries, output_dimension=output_dimension, input_type="search_query")

    # Compute image↔image similarity (pairwise, but we only have 2 by default)
    image_pairs: List[Tuple[str, str, float]] = []
    for i in range(len(name_order)):
        for j in range(i + 1, len(name_order)):
            a, b = name_order[i], name_order[j]
            sim = cosine(img_embeddings[a], img_embeddings[b])
            image_pairs.append((a, b, sim))

    print("Image ↔ Image cosine similarity:")
    for a, b, s in image_pairs:
        print(f"  {a}  vs  {b}  =>  {s:.4f}")
    print()

    # Compute query↔image similarity
    rows: List[Dict[str, Union[str, float]]] = []
    print("Query ↔ Image cosine similarity:")
    for qi, q in enumerate(queries):
        qv = query_vecs[qi]
        sims = []
        for nm in name_order:
            s = cosine(qv, img_embeddings[nm])
            sims.append((nm, s))
            rows.append({
                "query": q,
                "image_name": nm,
                "cosine_similarity": round(float(s), 6),
            })
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
        printable = ",  ".join([f"{nm}: {s:.4f}" for nm, s in sims_sorted])
        print(f'  "{q}" -> {printable}')
    print()

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} with {len(rows)} rows.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Cohere embed-v4.0 image+text similarity assignment.")
    parser.add_argument("--images", nargs="*", default=None,
                        help="Optional override: provide pairs name=URL_or_PATH. Example: img1=https://... img2=local.png")
    parser.add_argument("--queries", nargs="*", default=None,
                        help='Optional override: provide text queries. Example: --queries "a person" "a cart"')
    parser.add_argument("--output_csv", default="similarities.csv", help="Where to save the similarity table.")
    parser.add_argument("--dim", type=int, default=OUTPUT_DIM, help="Embedding dimension (256..1536). Default 512.")
    args = parser.parse_args()

    # Build images dict from overrides, if any
    images = None
    if args.images:
        images = {}
        for pair in args.images:
            if "=" not in pair:
                raise SystemExit(f"Bad --images arg '{pair}'. Use name=URL_or_PATH format.")
            name, src = pair.split("=", 1)
            images[name] = src

    # Build queries list from overrides, if any
    queries = args.queries if args.queries else None

    run(images=images, queries=queries, output_csv=args.output_csv, output_dimension=args.dim)


if __name__ == "__main__":
    main()

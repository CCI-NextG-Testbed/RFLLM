from sentence_transformers import SentenceTransformer
import json

INPUT_JSON = "./Knowledge_Base/Chunks/WLAN_OVERVIEW.json"
OUTPUT_JSON = "./Knowledge_Base/Chunks/WLAN_OVERVIEW.json"
MODEL_NAME = "thenlper/gte-large"

print("Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    chunks = load_json(INPUT_JSON)
    print(f"Loaded {len(chunks)} chunks")

    all_chunks_with_embeddings = []

    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]

        print(f"Embedding batch {i} → {i + len(batch) - 1}")

        # Generate embeddings
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Insert embedding into each chunk dict
        for chunk, emb in zip(batch, embeddings):
            chunk["embedding"] = emb.tolist()  # store embedding here
            all_chunks_with_embeddings.append(chunk)

    save_json(all_chunks_with_embeddings, OUTPUT_JSON)
    print(f"Saved final embedding file: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()

import gradio as gr
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Load model and label info ────────────────────────────
print("Loading model...")
with open("patent_ipc_svm.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_info.json", "r") as f:
    label_info = json.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

label_descriptions = label_info["label_descriptions"]

# ── Prediction function ──────────────────────────────────
def classify_patent(abstract):
    if not abstract or len(abstract.strip()) < 50:
        return "Please enter a longer patent abstract (at least 50 characters)"

    # Generate embedding
    embedding = embedding_model.encode([abstract])

    # Predict
    prediction = model.predict(embedding)[0]
    
    # Get description
    description = label_descriptions.get(prediction, "Unknown")

    # Format output
    result = f"""
## Predicted IPC Section: {prediction}
### {description}

**Confidence breakdown:**
"""
    # Get decision scores if available
    try:
        scores = model.decision_function(embedding)[0]
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
        score_pairs = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
        
        for label, score in score_pairs:
            desc = label_descriptions.get(label, "")
            bar = "█" * int((score - min(scores)) / (max(scores) - min(scores)) * 20)
            result += f"\n- **{label} ({desc})**: {bar}"
    except:
        pass

    return result

# ── Example abstracts ────────────────────────────────────
examples = [
    ["A pharmaceutical composition comprising a novel compound for treating cardiovascular disease, wherein the compound inhibits specific enzyme pathways to reduce blood pressure and improve cardiac function in patients."],
    ["A method for transmitting data packets over a wireless network using improved error correction algorithms, comprising encoding the data with redundancy bits and decoding at the receiver using maximum likelihood estimation."],
    ["A combustion engine comprising an improved fuel injection system with variable timing control, wherein the injection nozzle geometry is optimized to reduce emissions and improve fuel efficiency across different load conditions."]
]

# ── Build Gradio interface ───────────────────────────────
demo = gr.Interface(
    fn=classify_patent,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste a patent abstract here...",
        label="Patent Abstract"
    ),
    outputs=gr.Markdown(label="Classification Result"),
    title="Patent IPC Section Classifier",
    description="""
    This tool classifies patent abstracts into one of 8 IPC (International Patent Classification) sections:
    - **A**: Human Necessities | **B**: Performing Operations | **C**: Chemistry & Metallurgy | **D**: Textiles & Paper
    - **E**: Fixed Constructions | **F**: Mechanical Engineering | **G**: Physics | **H**: Electricity
    """,
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
# Patent IPC Section Classifier

A text classifier that predicts the International Patent Classification (IPC) 
section of a patent based on its abstract, using sentence embeddings and 
traditional machine learning classifiers.

## Project Structure
patent-ipc-classifier/
│
├── patent-ipc-classifier.ipynb  # Main notebook with all steps
├── app.py                        # Gradio demo
├── requirements.txt              # Required libraries
└── README.md

## Dataset
The dataset consists of 1,200 patent abstracts balanced across 8 IPC sections 
(150 per section). Data was collected from two sources:
- **European Patent Office (EPO)** via the OPS API (sections A and B)
- **Big Patent corpus** (USPTO patents) for remaining sections

Curation decisions:
- Abstracts shorter than 200 characters removed
- Abstracts longer than 1500 characters removed
- Duplicates removed
- Classes balanced to exactly 150 patents per section

🔗 [Dataset on HuggingFace](https://huggingface.co/datasets/charozvb/patent-ipc-classifier)

## Approach
1. Patent abstracts are converted into 384-dimensional vectors using the 
`all-MiniLM-L6-v2` sentence transformer model
2. Three classifiers are trained on these embeddings and evaluated on a 
held-out test set (80/20 split)

## Results
| Model | Accuracy |
|---|---|
| SVM (linear kernel) | 67.92% |
| Logistic Regression | 67.08% |
| Random Forest | 65.83% |
| Random baseline | 12.50% |

All models significantly outperform the random baseline of 12.5% (1/8 classes).

## IPC Sections
| Section | Description |
|---|---|
| A | Human Necessities |
| B | Performing Operations |
| C | Chemistry & Metallurgy |
| D | Textiles & Paper |
| E | Fixed Constructions |
| F | Mechanical Engineering |
| G | Physics |
| H | Electricity |

## Links
- 🤗 [Dataset](https://huggingface.co/datasets/charozvb/patent-ipc-classifier)
- 🤗 [Model](https://huggingface.co/charozvb/patent-ipc-classifier)
- 🚀 [Live Demo](https://huggingface.co/spaces/charozvb/patent-ipc-classifier)

## How to Run Locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the demo
```bash
python app.py
```

### Reproduce the results
Open `patent-ipc-classifier.ipynb` and run all cells in order.
Note: You will need EPO OPS API credentials and a HuggingFace write token.

## Requirements
- Python 3.8+
- See `requirements.txt` for full list of dependencies

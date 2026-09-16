# Query Understanding & Semantic Search Engine

## Goal

Build a small NLP-powered search system that can understand what a user is asking instead of relying only on exact keyword matching.

The core pipeline is:

```text
User Text
   ↓
Query Understanding
   ↓
Intent + Entities + Constraints
   ↓
Search / Retrieval
   ↓
Ranking
   ↓
Relevant Results
```

## Development Roadmap

### V1 — Classical NLP  ( currently )

Start without large language models.

```text
Text
 ↓
Preprocessing
 ↓
TF-IDF
 ↓
Classical Classifier
 ↓
Intent
```

Initial intents:

* `product_search`
* `information_question`
* `comparison`
* `recommendation`
* `problem_solving`

Example:

```text
Input:
"50k er moddhe gaming laptop chai"

Output:
{
  "intent": "product_search"
}
```

Goal: understand how traditional machine-learning systems classify unseen text.

---

### V2 — Transformer-based Intent Classification

Use a pretrained Transformer such as BERT/BART.

```text
Text
 ↓
Tokenizer
 ↓
Transformer
 ↓
Classification Layer
 ↓
Intent
```

The goal is to compare Transformer-based language understanding with the V1 classical approach.

---

### V3 — Embeddings + Semantic Search

Move from intent classification toward meaning-based retrieval.

```text
User Query
 ↓
Embedding Model
 ↓
Query Vector
 ↓
Vector Similarity
 ↓
Relevant Documents
```

The system should be able to recognize semantic similarity even when the exact words are different.

Example:

```text
Query:
"cheap smartphone"

Document:
"affordable mobile phone"
```

Keyword matching may consider these different, while embeddings can identify their similar meaning.

---

### V4 — Hybrid Query Understanding

Combine the previous approaches.

```text
                    User Query
                        ↓
              Query Understanding
                        ↓
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
       Intent        Entities      Embedding
          ↓             ↓             ↓
          └─────────────┼─────────────┘
                        ↓
                 Hybrid Retrieval
                        ↓
                     Ranking
                        ↓
                  Search Results
```

Final system should attempt to understand:

* **Intent** — What does the user want?
* **Entities** — What things are they talking about?
* **Constraints** — What conditions or preferences do they have?
* **Semantic meaning** — What does the query mean beyond exact words?

## Main Research Question

> **How can a machine convert arbitrary user text into a structured representation of what the user actually wants?**

The project will start with simple NLP techniques and gradually evolve into a modern semantic search system.

The objective is not to build Google. The objective is to understand the fundamental technologies behind intelligent search by implementing them step by step.

### 1. **Input → Tokens → Embeddings**

* You write: `"I like pizza"`.
* The tokenizer splits it into subword tokens, e.g. `["I", " like", " pizza"]`.
* Each token is mapped to an **embedding vector** of size `d_model`.
* Then **positional encoding** is added so the model knows the order.

Now we have:

$$
x \in \mathbb{R}^{\text{seq\_len} \times d_{\text{model}}}
$$

---

### 2. **Encoder (if it’s an encoder–decoder model like classic Transformer / translation)**

* **Self-attention layers:** each token looks at all other tokens in the input → learns context.
* **Feed-forward layers:** transform each token’s vector nonlinearly.
* **Residuals + LayerNorm:** stabilize training.
* After stacking many blocks, we end up with **contextualized embeddings**:

  * `"pizza"` now “knows” it was preceded by `"like"`.
  * `"I"` now “knows” it’s the subject of `"like"`.

Output = encoder representations (one vector per input token).

*(Note: GPT doesn’t use an encoder; it’s decoder-only. But translation models use encoder+decoder.)*

---

### 3. **Decoder**

* Works step by step, generating the output tokens.
* At each step:

  1. **Masked self-attention:** decoder looks only at what it has generated so far.
  2. **Cross-attention:** decoder queries the encoder’s output to align with the source sentence.
  3. **Feed-forward + residuals/norms:** refine representation.
* Decoder outputs a vector `[d_model]` for the *current position*.

---

### 4. **Projection to Vocabulary**

* That vector is multiplied by the **vocabulary matrix** (`[vocab_size, d_model]`).
* Softmax → probabilities for all tokens.
* The most likely (or sampled) token is chosen.

---

### 5. **Repeat**

* Append that token to the output sequence.
* Feed it back into the decoder for the next step.
* Continue until an end-of-sequence token (`<eos>`) is generated.

---

👉 So the flow is:
**Text → Tokens → Embeddings → Encoder (context) → Decoder (generation) → Softmax over vocab → Next token → Repeat.**

---
---



### 1. **Embeddings (before the encoder)**

* The **embedding matrix** is just a lookup table: each token id → one vector (`d_model` long).
* These vectors are learned during pretraining and already capture some *semantic similarity*.

  * Example: even before context, `"king"` and `"queen"` end up close in embedding space because they often appear in similar contexts in training.

So yes, **the base embeddings already show patterns like “king ≈ queen – man + woman ≈ ?”** (that’s from word2vec/GloVe days, and it still holds).

---

### 2. **Encoder output (after self-attention layers)**

* The encoder takes those embeddings and **refines them with context**.
* Now each token’s vector isn’t just its dictionary meaning → it’s contextualized.

  * Example: `"bank"` in `"river bank"` vs `"bank account"` will get *different vectors* after passing through the encoder.
  * `"queen"` will be pulled closer to `"king"` if the sentence is about royalty, but maybe closer to `"bee"` if the sentence is about insects 🐝.

So:

* **Embeddings (input)** = general meaning, static.
* **Encoder outputs** = contextual meaning, dynamic.

---

👉 That’s why `"queen"` and `"king"` being close is mostly about **embeddings before the encoder**.
But the **encoder layers make them context-sensitive** → `"queen"` in a chess context vs monarchy context will shift.


# Multilingual Translation Benchmark

This branch contains benchmarking experiments for various multilingual translation models from Hugging Face's Transformers library. The goal is to **evaluate translation quality, speed, and resource usage** for translating a consistent paragraph dataset into English from multiple source languages.

---

## Benchmark Overview

Each `.ipynb` notebook benchmarks a different translation model using the **same evaluation pipeline**:

- **Source → English translation**
- **10 languages tested**: Arabic, Spanish, French, Russian, Romanian, Bulgarian, Czech, Danish, German, Estonian
- **Consistent input paragraph** for direct comparison

---

## Evaluated Models

| Model Notebook                                      | Hugging Face Model ID                              | Size   |
|-----------------------------------------------------|----------------------------------------------------|--------|
| `facebook_m2m100_1_2B.ipynb`                        | `facebook/m2m100_1.2B`                             | 1.2B   |
| `facebook_m2m100_418M.ipynb`                        | `facebook/m2m100_418M`                             | 418M   |
| `facebook_mbart_large_50_many_to_many_mmt.ipynb`    | `facebook/mbart-large-50-many-to-many-mmt`         | 610M   |
| `facebook_nllb_200_distilled_1_3B.ipynb`            | `facebook/nllb-200-distilled-1.3B`                 | 1.3B   |
| `facebook_nllb_200_distilled_600M.ipynb`            | `facebook/nllb-200-distilled-600M`                 | 600M   |
| `google_t5_t5_base.ipynb`                           | `google-t5/t5-base`                                | 220M   |
| `marianMT.ipynb`                                    | `Helsinki-NLP/opus-mt-XX-en` (multiple variants)   | ~300M  |

>  All notebooks are available in the `Benchmark/` folder.

---

##  Evaluation Metrics

Each model is evaluated using the following standard MT metrics (via `evaluate` and `COMET`):

- **BLEU** – n-gram overlap
- **METEOR** – semantic and lexical matching
- **ROUGE-L** – longest common subsequence
- **TER** – translation edit rate
- **BERTScore** – semantic similarity using contextual embeddings
- **COMET** – trained quality estimation model

Additionally, we measure:

- **Time to first token**
- **Total inference time**
- **Tokens per second**
- **VRAM and RAM usage**
- **CPU usage**
- **Model parameter size**

---

## Example Input

```text
"I am applying for a tourist visa to visit the United States. I intend to stay for two weeks and visit popular landmarks like the Statue of
Liberty and Times Square. I will return to my home country after my vacation."
```

---

##  Notes
All tests use GPU if available via torch.cuda.

Ensure you have a Hugging Face token to download the models (huggingface_hub.login()).

COMET requires internet to download models on first run.


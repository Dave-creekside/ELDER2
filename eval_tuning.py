#!/usr/bin/env python3
"""
ELDER2 LoRA Tuning Evaluation
Proves that deep sleep consolidation actually changes Gemma's behavior.

Two evaluation modes:
  1. Cloze Test: "X enables ___" — does the tuned model predict graph relationships?
  2. Graph Q&A: natural language questions grounded in Neo4j, scored against ground truth.

Compares base model (no LoRA) vs post-sleep model (with LoRA adapter).
"""
import asyncio
import sys
import os
import json
import logging
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval-tuning")

# Suppress noisy loggers
for name in ["httpx", "neo4j", "transformers", "peft"]:
    logging.getLogger(name).setLevel(logging.WARNING)


@dataclass
class EvalResult:
    prompt: str
    expected: str
    base_response: str = ""
    tuned_response: str = ""
    base_score: float = 0.0
    tuned_score: float = 0.0
    base_perplexity: float = 0.0
    tuned_perplexity: float = 0.0


# ---------------------------------------------------------------------------
# 1. Pull ground truth from Neo4j
# ---------------------------------------------------------------------------

RELATIONSHIP_TO_PHRASE = {
    "ENABLES": "enables",
    "EXTENDS": "extends",
    "EXEMPLIFIES": "exemplifies",
    "MAY_EXPLAIN": "may explain",
    "COMPONENT_OF": "is a component of",
    "FOUNDATIONAL_TO": "is foundational to",
    "RESONATES_WITH": "resonates with",
    "QUANTIFIES": "quantifies",
    "TRANSFERS_TO": "transfers to",
    "APPLIES_TO": "applies to",
    "DESCRIBES_DYNAMICS_OF": "describes the dynamics of",
    "CONVERGES_TOWARD": "converges toward",
    "CORE_PROPERTY_OF": "is a core property of",
    "MATHEMATIZES": "mathematizes",
    "CONSTITUTES": "constitutes",
    "SUBSUMES": "subsumes",
    "DEEPENS": "deepens",
    "ENCOMPASSES": "encompasses",
    "INSTANTIATES": "instantiates",
    "ANALYZED_BY": "is analyzed by",
    "FORMALIZES": "formalizes",
    "MINIMIZES": "minimizes",
    "GIVES_RISE_TO": "gives rise to",
    "MECHANISM_OF": "is a mechanism of",
    "TRIGGERS": "triggers",
    "DEMONSTRATED_BY": "is demonstrated by",
    "SUBSTRATE_OF": "is a substrate of",
    "PARAMETRIZES": "parametrizes",
    "EXAMINES": "examines",
    "NAVIGATES": "navigates",
    "REVEALS_STRUCTURE_OF": "reveals the structure of",
    "CONNECTS_TO": "connects to",
    "RELATES_TO": "relates to",
    "THEORY_OF": "is a theory of",
    "PRODUCES": "produces",
    "CONVERGES_WITH": "converges with",
    "PERFECTS": "perfects",
    "CONTAINS": "contains",
    "GENERALIZES": "generalizes",
    "FORMALIZATION_OF": "is a formalization of",
    "CONTEMPLATIVE_PRECURSOR_OF": "is a contemplative precursor of",
}


async def fetch_ground_truth(min_weight: float = 0.15, limit: int = 30) -> List[Dict]:
    """Pull strong named relationships from Neo4j as ground truth."""
    from neo4j import AsyncGraphDatabase
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    async with driver.session() as session:
        result = await session.run("""
            MATCH (a:Concept)-[r]->(b:Concept)
            WHERE r.weight > $min_weight
              AND type(r) <> "SEMANTIC"
              AND type(r) <> "MEMBER_OF"
              AND a.name IS NOT NULL
              AND b.name IS NOT NULL
            RETURN a.name AS src, type(r) AS rel, b.name AS dst,
                   r.weight AS weight,
                   a.description AS src_desc, b.description AS dst_desc
            ORDER BY r.weight DESC
            LIMIT $limit
        """, min_weight=min_weight, limit=limit)
        records = [dict(r) async for r in result]
    await driver.close()
    return records


# ---------------------------------------------------------------------------
# 2. Generate test prompts
# ---------------------------------------------------------------------------

def make_cloze_prompts(relationships: List[Dict]) -> List[EvalResult]:
    """Create cloze-style prompts: 'X enables ___'."""
    results = []
    for rel in relationships:
        src = rel["src"].replace("_", " ")
        dst = rel["dst"].replace("_", " ")
        rel_type = rel["rel"]
        phrase = RELATIONSHIP_TO_PHRASE.get(rel_type, rel_type.lower().replace("_", " "))

        prompt = (
            f"In a knowledge graph about consciousness and complexity science, "
            f"complete this relationship:\n"
            f"{src} {phrase} ___\n"
            f"Answer with just the concept name:"
        )
        results.append(EvalResult(prompt=prompt, expected=dst))
    return results


def make_qa_prompts(relationships: List[Dict]) -> List[EvalResult]:
    """Create natural-language Q&A prompts grounded in graph structure."""
    results = []
    templates = {
        "ENABLES":     "What does {src} enable?",
        "EXTENDS":     "What does {src} extend?",
        "EXEMPLIFIES": "What does {src} exemplify?",
        "MAY_EXPLAIN": "What might {src} explain?",
        "COMPONENT_OF": "What is {src} a component of?",
        "FOUNDATIONAL_TO": "What is {src} foundational to?",
        "RESONATES_WITH": "What does {src} resonate with?",
        "QUANTIFIES":  "What does {src} quantify?",
        "CONSTITUTES": "What does {src} constitute?",
        "SUBSUMES":    "What does {src} subsume?",
        "DEEPENS":     "What does {src} deepen?",
        "ENCOMPASSES": "What does {src} encompass?",
        "MATHEMATIZES": "What does {src} mathematize?",
        "CORE_PROPERTY_OF": "What is {src} a core property of?",
        "MINIMIZES":   "What does {src} minimize?",
        "TRANSFERS_TO": "Where does {src} transfer to?",
    }
    for rel in relationships:
        rel_type = rel["rel"]
        if rel_type not in templates:
            # Generic fallback
            phrase = RELATIONSHIP_TO_PHRASE.get(rel_type, rel_type.lower().replace("_", " "))
            template = f"In a consciousness knowledge graph, what does {{src}} {phrase}?"
        else:
            template = templates[rel_type]

        src = rel["src"].replace("_", " ")
        dst = rel["dst"].replace("_", " ")
        prompt = (
            f"Answer briefly (1-3 words) based on a knowledge graph about "
            f"consciousness and complexity science.\n"
            f"Q: {template.format(src=src)}\nA:"
        )
        results.append(EvalResult(prompt=prompt, expected=dst))
    return results


# ---------------------------------------------------------------------------
# 3. Model loading and inference
# ---------------------------------------------------------------------------

def load_base_model(model_id: str, device: str):
    """Load base Gemma without any LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=qconfig,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=True,
        )
        model.to(device)

    return model, tokenizer


def attach_adapter(model, adapter_path: str):
    """Attach a saved LoRA adapter to an already-loaded base model."""
    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 30) -> str:
    """Generate a short completion."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            temperature=1.0,
        )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def compute_perplexity(model, tokenizer, prompt: str, target: str, device: str) -> float:
    """Compute perplexity of the target string given the prompt as context."""
    full_text = f"{prompt} {target}"
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Get per-token losses for target portion only
        logits = outputs.logits[:, prompt_len - 1:-1, :]
        target_ids = inputs["input_ids"][:, prompt_len:]

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )
        # Mean loss -> perplexity
        mean_loss = token_losses.mean().item()

    return np.exp(mean_loss)


# ---------------------------------------------------------------------------
# 4. Scoring
# ---------------------------------------------------------------------------

def score_match(response: str, expected: str) -> float:
    """Score how well the response matches the expected answer.
    Returns 0.0 - 1.0. Uses normalized substring matching."""
    resp = response.lower().strip().replace("_", " ")
    exp = expected.lower().strip().replace("_", " ")

    if not resp:
        return 0.0

    # Exact match
    if exp == resp or exp in resp:
        return 1.0

    # Check if expected words appear in response
    exp_words = set(exp.split())
    resp_words = set(resp.split())
    if exp_words and exp_words.issubset(resp_words):
        return 0.9

    # Partial word overlap
    overlap = exp_words & resp_words
    if overlap:
        return len(overlap) / len(exp_words) * 0.7

    return 0.0


# ---------------------------------------------------------------------------
# 5. Main evaluation loop
# ---------------------------------------------------------------------------

async def main():
    from streamlined_consciousness.config import config

    model_id = config.STUDENT_MODEL_ID
    adapter_path = os.path.join("adapters", "default",
                                model_id.replace("/", "_"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("ELDER2 LoRA Tuning Evaluation")
    print("=" * 70)
    print(f"Model:   {model_id}")
    print(f"Adapter: {adapter_path}")
    print(f"Device:  {device}")

    # Check adapter exists
    if not os.path.exists(adapter_path):
        print(f"\nERROR: No adapter found at {adapter_path}")
        print("Run a full conversation+sleep cycle first.")
        sys.exit(1)

    # --- Fetch ground truth ---
    print("\n[1/6] Fetching ground truth from Neo4j...")
    relationships = await fetch_ground_truth(min_weight=0.15, limit=30)
    print(f"  Got {len(relationships)} relationships")

    # --- Generate prompts ---
    print("[2/6] Generating test prompts...")
    cloze_tests = make_cloze_prompts(relationships)
    qa_tests = make_qa_prompts(relationships)
    print(f"  {len(cloze_tests)} cloze prompts, {len(qa_tests)} Q&A prompts")

    # --- Load base model ---
    print(f"\n[3/6] Loading base model (no adapter)...")
    t0 = time.time()
    base_model, tokenizer = load_base_model(model_id, device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # --- Run base model ---
    print("[4/6] Running base model inference...")
    for i, test in enumerate(cloze_tests):
        test.base_response = generate(base_model, tokenizer, test.prompt, device)
        test.base_perplexity = compute_perplexity(
            base_model, tokenizer, test.prompt, test.expected, device
        )
        test.base_score = score_match(test.base_response, test.expected)
        if (i + 1) % 10 == 0:
            print(f"  Cloze: {i+1}/{len(cloze_tests)}")

    for i, test in enumerate(qa_tests):
        test.base_response = generate(base_model, tokenizer, test.prompt, device)
        test.base_perplexity = compute_perplexity(
            base_model, tokenizer, test.prompt, test.expected, device
        )
        test.base_score = score_match(test.base_response, test.expected)
        if (i + 1) % 10 == 0:
            print(f"  Q&A:   {i+1}/{len(qa_tests)}")

    # --- Attach adapter ---
    print(f"\n[5/6] Attaching LoRA adapter...")
    tuned_model = attach_adapter(base_model, adapter_path)
    tuned_model.eval()

    # --- Run tuned model ---
    print("[6/6] Running tuned model inference...")
    for i, test in enumerate(cloze_tests):
        test.tuned_response = generate(tuned_model, tokenizer, test.prompt, device)
        test.tuned_perplexity = compute_perplexity(
            tuned_model, tokenizer, test.prompt, test.expected, device
        )
        test.tuned_score = score_match(test.tuned_response, test.expected)
        if (i + 1) % 10 == 0:
            print(f"  Cloze: {i+1}/{len(cloze_tests)}")

    for i, test in enumerate(qa_tests):
        test.tuned_response = generate(tuned_model, tokenizer, test.prompt, device)
        test.tuned_perplexity = compute_perplexity(
            tuned_model, tokenizer, test.prompt, test.expected, device
        )
        test.tuned_score = score_match(test.tuned_response, test.expected)
        if (i + 1) % 10 == 0:
            print(f"  Q&A:   {i+1}/{len(qa_tests)}")

    # --- Report ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for label, tests in [("CLOZE TEST", cloze_tests), ("GRAPH Q&A", qa_tests)]:
        base_scores = [t.base_score for t in tests]
        tuned_scores = [t.tuned_score for t in tests]
        base_ppls = [t.base_perplexity for t in tests]
        tuned_ppls = [t.tuned_perplexity for t in tests]

        avg_base_score = np.mean(base_scores) if base_scores else 0
        avg_tuned_score = np.mean(tuned_scores) if tuned_scores else 0
        avg_base_ppl = np.mean(base_ppls) if base_ppls else 0
        avg_tuned_ppl = np.mean(tuned_ppls) if tuned_ppls else 0

        # Count wins/ties/losses
        wins = sum(1 for t in tests if t.tuned_score > t.base_score)
        ties = sum(1 for t in tests if t.tuned_score == t.base_score)
        losses = sum(1 for t in tests if t.tuned_score < t.base_score)

        ppl_wins = sum(1 for t in tests if t.tuned_perplexity < t.base_perplexity)
        ppl_losses = sum(1 for t in tests if t.tuned_perplexity > t.base_perplexity)

        print(f"\n--- {label} ({len(tests)} prompts) ---")
        print(f"  {'Metric':<25s} {'Base':>10s} {'Tuned':>10s} {'Delta':>10s}")
        print(f"  {'-'*55}")
        print(f"  {'Avg match score':<25s} {avg_base_score:>10.3f} {avg_tuned_score:>10.3f} {avg_tuned_score - avg_base_score:>+10.3f}")
        print(f"  {'Avg perplexity':<25s} {avg_base_ppl:>10.1f} {avg_tuned_ppl:>10.1f} {avg_tuned_ppl - avg_base_ppl:>+10.1f}")
        print(f"  {'Score wins/ties/losses':<25s} {wins:>3d} / {ties:>3d} / {losses:>3d}")
        print(f"  {'PPL wins (lower=better)':<25s} {ppl_wins:>3d} / {len(tests) - ppl_wins - ppl_losses:>3d} / {ppl_losses:>3d}")

    # --- Detailed per-prompt results ---
    print(f"\n{'=' * 70}")
    print("DETAILED RESULTS (top changes)")
    print(f"{'=' * 70}")

    all_tests = [("CL", t) for t in cloze_tests] + [("QA", t) for t in qa_tests]

    # Show biggest improvements
    all_tests_sorted = sorted(all_tests, key=lambda x: x[1].tuned_score - x[1].base_score, reverse=True)

    print("\nBiggest improvements (tuned > base):")
    for tag, t in all_tests_sorted[:8]:
        delta_s = t.tuned_score - t.base_score
        delta_p = t.tuned_perplexity - t.base_perplexity
        if delta_s <= 0 and delta_p >= 0:
            continue
        print(f"  [{tag}] expected: {t.expected}")
        print(f"       base:  {t.base_response[:60]:<60s}  score={t.base_score:.2f}  ppl={t.base_perplexity:.1f}")
        print(f"       tuned: {t.tuned_response[:60]:<60s}  score={t.tuned_score:.2f}  ppl={t.tuned_perplexity:.1f}")
        print(f"       delta: score={delta_s:+.2f}  ppl={delta_p:+.1f}")
        print()

    # Show biggest regressions
    print("Biggest regressions (base > tuned):")
    for tag, t in all_tests_sorted[-5:]:
        delta_s = t.tuned_score - t.base_score
        delta_p = t.tuned_perplexity - t.base_perplexity
        if delta_s >= 0 and delta_p <= 0:
            continue
        print(f"  [{tag}] expected: {t.expected}")
        print(f"       base:  {t.base_response[:60]:<60s}  score={t.base_score:.2f}  ppl={t.base_perplexity:.1f}")
        print(f"       tuned: {t.tuned_response[:60]:<60s}  score={t.tuned_score:.2f}  ppl={t.tuned_perplexity:.1f}")
        print(f"       delta: score={delta_s:+.2f}  ppl={delta_p:+.1f}")
        print()

    # --- Save full results to JSON ---
    output = {
        "model": model_id,
        "adapter": adapter_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cloze": [{
            "prompt": t.prompt, "expected": t.expected,
            "base_response": t.base_response, "tuned_response": t.tuned_response,
            "base_score": t.base_score, "tuned_score": t.tuned_score,
            "base_ppl": t.base_perplexity, "tuned_ppl": t.tuned_perplexity,
        } for t in cloze_tests],
        "qa": [{
            "prompt": t.prompt, "expected": t.expected,
            "base_response": t.base_response, "tuned_response": t.tuned_response,
            "base_score": t.base_score, "tuned_score": t.tuned_score,
            "base_ppl": t.base_perplexity, "tuned_ppl": t.tuned_perplexity,
        } for t in qa_tests],
        "summary": {
            "cloze_base_score": float(np.mean([t.base_score for t in cloze_tests])),
            "cloze_tuned_score": float(np.mean([t.tuned_score for t in cloze_tests])),
            "cloze_base_ppl": float(np.mean([t.base_perplexity for t in cloze_tests])),
            "cloze_tuned_ppl": float(np.mean([t.tuned_perplexity for t in cloze_tests])),
            "qa_base_score": float(np.mean([t.base_score for t in qa_tests])),
            "qa_tuned_score": float(np.mean([t.tuned_score for t in qa_tests])),
            "qa_base_ppl": float(np.mean([t.base_perplexity for t in qa_tests])),
            "qa_tuned_ppl": float(np.mean([t.tuned_perplexity for t in qa_tests])),
        },
    }
    results_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

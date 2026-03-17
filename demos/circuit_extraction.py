import dotenv
dotenv.load_dotenv()
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.utils.demo_utils import get_top_features
from demos.adapters import (
    DatasetAdapter,
    BiasInBiosAdapter,
    CivilCommentsAdapter,
    GenericHFAdapter,
    MultiNLIAdapter,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
SEED = 42

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME       = "google/gemma-2-2b"
TRANSCODER_NAME  = "gemma"
BACKEND          = "transformerlens"  # or "nnsight"
PROBE_LAYER      = 22                 # layer whose residual stream the probe reads

# Neuronpedia identifiers — update if using a different model/transcoder
NEURONPEDIA_MODEL_ID = "gemma-2-2b"
NEURONPEDIA_SAE_ID   = f"{PROBE_LAYER}-gemmascope-transcoder-16k"

# ── Dataset adapter ────────────────────────────────────────────────────────────
# Uncomment ONE block. The adapter provides get_batches / get_subgroups /
# get_sample_prompt and carries activation_dim for the target model.

# BiasInBios — profession classification (true) vs gender (spurious)
# adapter: DatasetAdapter = BiasInBiosAdapter(
#     neg_profession="surgeon",
#     pos_profession="teacher",
#     activation_dim_value=2304,  # Gemma-2-2B hidden size
# )
# PROBE_SLUG = "bib_surgeon_teacher"

# CivilComments — toxicity (true) vs identity-attack annotation (spurious)
adapter: DatasetAdapter = CivilCommentsAdapter(activation_dim_value=2304)
PROBE_SLUG = "civil_comments"

# MultiNLI — contradiction (true) vs negation words in hypothesis (spurious)
# adapter: DatasetAdapter = MultiNLIAdapter(activation_dim_value=2304)
# PROBE_SLUG = "multinli"

# Generic HuggingFace dataset
# adapter: DatasetAdapter = GenericHFAdapter(
#     dataset_name="...",
#     text_column="text",
#     primary_label_column="label",
#     spurious_label_column="spurious",
#     activation_dim_value=2304,
# )
# PROBE_SLUG = "my_dataset"

# ── Probe paths ────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path("/workspace/sae-feature-circuits/probe_extraction/probes/")
BIASED_PROBE_PATH   = EXPERIMENTS_DIR / f"{PROBE_SLUG}_layer{PROBE_LAYER}_bfloat16.pt"
UNBIASED_PROBE_PATH = EXPERIMENTS_DIR / f"{PROBE_SLUG}_layer{PROBE_LAYER}_bfloat16_unbiased.pt"

assert os.path.exists(BIASED_PROBE_PATH),   f"Not found: {BIASED_PROBE_PATH}"
assert os.path.exists(UNBIASED_PROBE_PATH), f"Not found: {UNBIASED_PROBE_PATH}"

# ── Setup ──────────────────────────────────────────────────────────────────────
print("=" * 60, flush=True)
print("SETUP", flush=True)
print("=" * 60, flush=True)
print(f"Device: {DEVICE}", flush=True)
print(f"Dataset: {PROBE_SLUG}  |  adapter: {type(adapter).__name__}", flush=True)
print(f"Probe layer: {PROBE_LAYER}  |  activation_dim: {adapter.activation_dim}", flush=True)
print(f"Loading model {MODEL_NAME} with {TRANSCODER_NAME} transcoders...", flush=True)

model = ReplacementModel.from_pretrained(
    MODEL_NAME,
    TRANSCODER_NAME,
    dtype=DTYPE,
    backend=BACKEND,
)
print(f"Loaded {MODEL_NAME}", flush=True)
print(f"n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}", flush=True)


class Probe(nn.Module):
    """Binary linear probe: nn.Linear(activation_dim, 1) with mean-pooled residual stream input."""

    def __init__(self, activation_dim: int, dtype=DTYPE):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @property
    def weight_vec(self) -> torch.Tensor:
        """Return the (d_model,) weight vector used as the attribution direction."""
        return self.net.weight.squeeze(0).detach()


@torch.no_grad()
def collect_activations(
    text_batches,
    layer: int = PROBE_LAYER,
):
    """
    Collect mean-pooled residual stream activations at a given layer.

    Uses TransformerLens run_with_cache to capture hook_resid_post at `layer`.
    Yields (pooled_acts, true_labels, spurious_labels) per batch.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    for texts, true_labels, spur_labels in tqdm(text_batches, desc="Collecting activations", unit="batch", leave=False):
        tokens = model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(DEVICE)
        input_ids = tokens["input_ids"]
        attn_mask = tokens["attention_mask"].float()

        _, cache = model.run_with_cache(
            input_ids,
            names_filter=hook_name,
            prepend_bos=False,
        )
        h_l = cache[hook_name]  # (batch, n_pos, d_model)

        # Attention-mask-weighted mean pooling
        pooled = (h_l * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1).unsqueeze(-1)
        yield pooled.detach(), true_labels, spur_labels


def train_probe(
    activation_batches,
    activation_dim: int,
    label_idx: int = 0,
    lr: float = 1e-2,
    epochs: int = 1,
    seed: int = SEED,
) -> Probe:
    torch.manual_seed(seed)
    probe = Probe(activation_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for acts, *labels in tqdm(activation_batches, desc=f"Training epoch {epoch+1}/{epochs}", unit="batch", leave=False):
            optimizer.zero_grad()
            logits = probe(acts.bfloat16())
            loss = criterion(logits, labels[label_idx].to(logits))
            loss.backward()
            optimizer.step()
    return probe


@torch.no_grad()
def test_probe(probe: Probe, activation_batches) -> dict:
    corrects = defaultdict(list)
    for acts, *labels in tqdm(activation_batches, desc="Evaluating", unit="batch", leave=False):
        preds = (probe(acts.bfloat16()) > 0.0).long()
        for idx, lbl in enumerate(labels):
            corrects[idx].append(preds == lbl)
    return {k: torch.cat(v).float().mean().item() for k, v in corrects.items()}


def count_samples(batches) -> int:
    """Count total samples across all batches."""
    return sum(len(batch[0]) for batch in batches)


def dtype_from_str(s: str) -> torch.dtype:
    options = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if s not in options:
        raise ValueError(f"Unsupported dtype {s!r}. Choose from: {list(options)}")
    return options[s]

def load_probe(path: str, device: str) -> Probe:
    """Load a probe saved by save_probe()."""
    checkpoint = torch.load(path, weights_only=True)
    probe = Probe(checkpoint["activation_dim"], dtype=dtype_from_str(checkpoint["dtype"]))
    probe.load_state_dict(checkpoint["state_dict"])
    return probe.to(device)

def save_probe(probe: Probe, path: str, activation_dim: int, dtype_str: str):
    """Save probe as state_dict + metadata. Portable across class locations."""
    torch.save({"state_dict": probe.state_dict(), "activation_dim": activation_dim, "dtype": dtype_str}, path)


# ── Stage 1: Biased probe ──────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 1: Biased probe (trained on ambiguous/correlated data)", flush=True)
print("=" * 60, flush=True)

if BIASED_PROBE_PATH.exists():
    print(f"Loading biased probe from {BIASED_PROBE_PATH}...", flush=True)
    probe_biased: Probe = load_probe(BIASED_PROBE_PATH, DEVICE)
    probe_biased.eval()
    print(f"Loaded biased probe: {probe_biased}", flush=True)
else:
    # Train biased probe on ambiguous data (true label == spurious label for all samples)
    print("Training biased probe on ambiguous data (true label correlated with spurious)...", flush=True)
    _train_batches = adapter.get_batches(split="train", ambiguous=True, device=DEVICE)
    print(f"  Train set (ambiguous): {count_samples(_train_batches):,} samples, {len(_train_batches)} batches", flush=True)
    probe_biased = train_probe(
        collect_activations(_train_batches),
        activation_dim=adapter.activation_dim,
        label_idx=0,
    )
    probe_biased.eval()
    torch.save(probe_biased, BIASED_PROBE_PATH)
    print(f"Saved biased probe to {BIASED_PROBE_PATH}", flush=True)

print("Evaluating biased probe on test set...", flush=True)
_ambig_test = adapter.get_batches(split="test", ambiguous=True, device=DEVICE)
print(f"  Ambiguous test set:   {count_samples(_ambig_test):,} samples, {len(_ambig_test)} batches", flush=True)
ambiguous_accs = test_probe(probe_biased, activation_batches=collect_activations(_ambig_test))
print(f"  Ambiguous test accuracy:    {ambiguous_accs[0]:.4f}", flush=True)

_unamb_test = adapter.get_batches(split="test", ambiguous=False, device=DEVICE)
print(f"  Unambiguous test set: {count_samples(_unamb_test):,} samples, {len(_unamb_test)} batches", flush=True)
unambiguous_accs = test_probe(probe_biased, activation_batches=collect_activations(_unamb_test))
print(f"  Ground truth accuracy:      {unambiguous_accs[0]:.4f}", flush=True)
print(f"  Unintended feature accuracy:{unambiguous_accs[1]:.4f}", flush=True)

print("Evaluating biased probe by subgroup...", flush=True)
_subgroups = adapter.get_subgroups(device=DEVICE)
for subgroup, batches in tqdm(_subgroups.items(), desc="Subgroup eval", unit="group"):
    n_sg = count_samples(batches)
    subgroup_accs = test_probe(probe_biased, activation_batches=collect_activations(batches))
    print(f"  Subgroup {subgroup} ({n_sg:,} samples) accuracy: {subgroup_accs[0]:.4f}", flush=True)


# ── Stage 2: Unbiased probe ────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 2: Unbiased probe (trained on balanced data)", flush=True)
print("=" * 60, flush=True)

if UNBIASED_PROBE_PATH.exists():
    print(f"Loading unbiased probe from {UNBIASED_PROBE_PATH}...", flush=True)
    probe_unbiased: Probe = load_probe(UNBIASED_PROBE_PATH, DEVICE)
    probe_unbiased.eval()
    print(f"Loaded unbiased probe: {probe_unbiased}", flush=True)
else:
    # Train unbiased probe on balanced data (all four true×spurious combinations)
    print("Training unbiased probe on balanced data (true label decorrelated from spurious)...", flush=True)
    _train_batches = adapter.get_batches(split="train", ambiguous=False, device=DEVICE)
    print(f"  Train set (balanced): {count_samples(_train_batches):,} samples, {len(_train_batches)} batches", flush=True)
    probe_unbiased = train_probe(
        collect_activations(_train_batches),
        activation_dim=adapter.activation_dim,
        label_idx=0,
    )
    probe_unbiased.eval()
    torch.save(probe_unbiased, UNBIASED_PROBE_PATH)
    print(f"Saved unbiased probe to {UNBIASED_PROBE_PATH}", flush=True)

print("Evaluating unbiased probe on test set...", flush=True)
_ambig_test = adapter.get_batches(split="test", ambiguous=True, device=DEVICE)
print(f"  Ambiguous test set:   {count_samples(_ambig_test):,} samples, {len(_ambig_test)} batches", flush=True)
ambiguous_accs = test_probe(probe_unbiased, activation_batches=collect_activations(_ambig_test))
print(f"  Ambiguous test accuracy:    {ambiguous_accs[0]:.4f}", flush=True)

_unamb_test = adapter.get_batches(split="test", ambiguous=False, device=DEVICE)
print(f"  Unambiguous test set: {count_samples(_unamb_test):,} samples, {len(_unamb_test)} batches", flush=True)
unambiguous_accs = test_probe(probe_unbiased, activation_batches=collect_activations(_unamb_test))
print(f"  Ground truth accuracy:      {unambiguous_accs[0]:.4f}", flush=True)
print(f"  Unintended feature accuracy:{unambiguous_accs[1]:.4f}", flush=True)

print("Evaluating unbiased probe by subgroup...", flush=True)
_subgroups = adapter.get_subgroups(device=DEVICE)
for subgroup, batches in tqdm(_subgroups.items(), desc="Subgroup eval", unit="group"):
    n_sg = count_samples(batches)
    subgroup_accs = test_probe(probe_unbiased, activation_batches=collect_activations(batches))
    print(f"  Subgroup {subgroup} ({n_sg:,} samples) accuracy: {subgroup_accs[0]:.4f}", flush=True)


# ── Stage 3: Probe direction comparison ───────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 3: Probe direction comparison", flush=True)
print("=" * 60, flush=True)

W_biased = probe_biased.weight_vec.float()    # (d_model,)
W_unbiased = probe_unbiased.weight_vec.float()  # (d_model,)

cosine_sim = torch.nn.functional.cosine_similarity(W_biased.unsqueeze(0), W_unbiased.unsqueeze(0)).item()
print(f"W_biased   norm: {W_biased.norm():.4f}", flush=True)
print(f"W_unbiased norm: {W_unbiased.norm():.4f}", flush=True)
print(f"Cosine similarity: {cosine_sim:.4f}", flush=True)
print("If cosine similarity < 1.0, the probes use different directions —", flush=True)
print("the biased probe has a spurious component absent in the unbiased probe.", flush=True)


# ── Stage 4: Sample prompt ─────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 4: Sample prompt", flush=True)
print("=" * 60, flush=True)

PROMPT = adapter.get_sample_prompt()
print(f"Prompt ({len(PROMPT)} chars, {len(model.tokenizer.encode(PROMPT))} tokens):", flush=True)
print(PROMPT, flush=True)

print("\nScoring sample prompt with both probes...", flush=True)
hook_name = f"blocks.{PROBE_LAYER}.hook_resid_post"
tokens = model.tokenizer(
    [PROMPT], return_tensors="pt", add_special_tokens=False
).to(DEVICE)
input_ids = tokens["input_ids"]
attn_mask = tokens["attention_mask"].float()

with torch.no_grad():
    _, cache = model.run_with_cache(
        input_ids, names_filter=hook_name, prepend_bos=False
    )
h_22 = cache[hook_name]  # (1, n_pos, d_model)
pooled_h22 = (h_22 * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1).unsqueeze(-1)

score_biased = probe_biased(pooled_h22.bfloat16()).item()
score_unbiased = probe_unbiased(pooled_h22.bfloat16()).item()

print(f"Biased   probe score: {score_biased:+.3f}  ({'pos' if score_biased > 0 else 'neg'})", flush=True)
print(f"Unbiased probe score: {score_unbiased:+.3f}  ({'pos' if score_unbiased > 0 else 'neg'})", flush=True)
print("Both should predict the positive class, but via different directions in the residual stream.", flush=True)


def build_probe_target(probe: Probe, label: str) -> CustomTarget:
    """
    Build a CustomTarget for circuit attribution from a linear probe.

    The probe's weight vector W_probe is used as the residual-stream direction
    to attribute toward. This is equivalent to asking: which transcoder features
    causally write in the direction W_probe?

    Args:
        probe: Trained Probe module with .weight_vec property.
        label: Human-readable name for the target.

    Returns:
        CustomTarget(token_str=label, prob=1.0, vec=W_probe)
    """
    W = probe.weight_vec.to(model.cfg.device).to(model.cfg.dtype)
    return CustomTarget(
        token_str=label,
        prob=1.0,   # uniform weight for a single target
        vec=W,
    )


target_biased = build_probe_target(probe_biased, "biased_probe")
target_unbiased = build_probe_target(probe_unbiased, "unbiased_probe")

print(f"\nBiased   probe direction norm: {target_biased.vec.norm():.4f}", flush=True)
print(f"Unbiased probe direction norm: {target_unbiased.vec.norm():.4f}", flush=True)
print(f"Cosine similarity (bfloat16): {torch.nn.functional.cosine_similarity(target_biased.vec.float().unsqueeze(0), target_unbiased.vec.float().unsqueeze(0)).item():.4f}", flush=True)

# Shared attribution parameters
ATTR_KWARGS = dict(
    batch_size=256,
    max_feature_nodes=4096,
    offload="cpu",       # offload to CPU to save GPU memory
    verbose=True,
)


# ── Stage 5: Simple attribution (CustomTarget approximation) ───────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 5: Simple attribution (CustomTarget approximation)", flush=True)
print("=" * 60, flush=True)

print("\nAttributing from BIASED probe direction...", flush=True)
graph_biased = attribute(
    prompt=PROMPT,
    model=model,
    attribution_targets=[target_biased],
    **ATTR_KWARGS,
)
print(f"Biased graph: {graph_biased.active_features.shape[0]} active features, "
      f"{graph_biased.selected_features.shape[0]} selected", flush=True)

print("\nAttributing from UNBIASED probe direction...", flush=True)
graph_unbiased = attribute(
    prompt=PROMPT,
    model=model,
    attribution_targets=[target_unbiased],
    **ATTR_KWARGS,
)
print(f"Unbiased graph: {graph_unbiased.active_features.shape[0]} active features, "
      f"{graph_unbiased.selected_features.shape[0]} selected", flush=True)


# ── Stage 6: Feature overlap analysis (simple) ────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 6: Feature overlap analysis (simple attribution)", flush=True)
print("=" * 60, flush=True)

N_TOP = 50  # number of top features to compare

top_biased, scores_biased = get_top_features(graph_biased, n=N_TOP)
top_unbiased, scores_unbiased = get_top_features(graph_unbiased, n=N_TOP)

# Convert to sets of (layer, pos, feature_idx) triples
set_biased = set(top_biased)
set_unbiased = set(top_unbiased)

# Spurious features: prominent in biased circuit, absent in unbiased
spurious = set_biased - set_unbiased
# Causal features: prominent in both circuits
shared = set_biased & set_unbiased
# Unbiased-only features: in unbiased but not biased
unbiased_only = set_unbiased - set_biased

print(f"Top-{N_TOP} features in biased circuit:   {len(set_biased)}", flush=True)
print(f"Top-{N_TOP} features in unbiased circuit: {len(set_unbiased)}", flush=True)
print(f"Shared (causal) features:    {len(shared)}", flush=True)
print(f"Spurious (biased-only):      {len(spurious)}", flush=True)
print(f"Unbiased-only features:      {len(unbiased_only)}", flush=True)

import requests
# Rank spurious features by their score in the biased circuit
biased_score_lookup = {feat: score for feat, score in zip(top_biased, scores_biased)}

spurious_ranked = sorted(spurious, key=lambda f: biased_score_lookup[f], reverse=True)

print("\nTop spurious features (in biased circuit, absent from unbiased top-50):", flush=True)
print(f"{'Layer':>6} {'Pos':>5} {'FeatIdx':>8} {'Biased Score':>14}", flush=True)
print("-" * 40, flush=True)
for layer, pos, feat_idx in tqdm(spurious_ranked[:20], desc="Fetching spurious features", unit="feat"):
    score = biased_score_lookup[(layer, pos, feat_idx)]
    neuronpedia_url = f"https://neuronpedia.org/{NEURONPEDIA_MODEL_ID}/{NEURONPEDIA_SAE_ID}/{feat_idx}"
    feat_api_url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/{NEURONPEDIA_SAE_ID}/{feat_idx}"
    feat_data =  requests.get(feat_api_url).json()
    explanations = feat_data.get("explanations", [])
    explanation_strs = (
        " |OR| ".join(exp.get("description", "No explanation") for exp in explanations)
        if explanations
        else "No explanation available"
    )
    print(f"{layer:>6} {pos:>5} {feat_idx:>8} {score:>14.4f}, {explanation_strs}, {neuronpedia_url}", flush=True)

# Also show top shared (causal) features for reference
shared_ranked = sorted(shared, key=lambda f: biased_score_lookup[f], reverse=True)

print("\nTop shared (causal) features (prominent in BOTH circuits):", flush=True)
print(f"{'Layer':>6} {'Pos':>5} {'FeatIdx':>8} {'Biased Score':>14} {'Unbiased Score':>16}", flush=True)
print("-" * 55, flush=True)
unbiased_score_lookup = {feat: score for feat, score in zip(top_unbiased, scores_unbiased)}
for layer, pos, feat_idx in shared_ranked[:20]:
    bs = biased_score_lookup[(layer, pos, feat_idx)]
    us = unbiased_score_lookup[(layer, pos, feat_idx)]
    print(f"{layer:>6} {pos:>5} {feat_idx:>8} {bs:>14.4f} {us:>16.4f}", flush=True)


# ── Stage 7: Save simple attribution graphs ────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 7: Saving simple attribution graphs", flush=True)
print("=" * 60, flush=True)

from circuit_tracer.utils import create_graph_files

output_dir = Path("probe_circuits")
output_dir.mkdir(exist_ok=True)

for slug, graph in tqdm([
    (f"{PROBE_SLUG}-biased-probe-simple", graph_biased),
    (f"{PROBE_SLUG}-unbiased-probe-simple", graph_unbiased),
], desc="Saving graphs", unit="graph"):
    print(f"Saving {slug}...", flush=True)
    graph.to_pt(output_dir / f"{slug}.pt")
    create_graph_files(
        graph_or_path=graph,
        slug=slug,
        output_path=str(output_dir / "graph_files"),
        node_threshold=0.8,
        edge_threshold=0.98,
    )

print(f"Saved simple attribution graphs to {output_dir}/graph_files/", flush=True)


# ── Stage 8: Visualization ─────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 8: Visualization", flush=True)
print("=" * 60, flush=True)

try:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Score distributions
    axes[0].hist(scores_biased, bins=30, alpha=0.7, label="Biased", color="tab:red")
    axes[0].hist(scores_unbiased, bins=30, alpha=0.7, label="Unbiased", color="tab:blue")
    axes[0].set_xlabel("Attribution score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Top-{N_TOP} feature score distributions")
    axes[0].legend()

    # Panel 2: Scatter plot — biased vs unbiased scores for shared features
    shared_list = list(shared)
    if shared_list:
        bs_vals = [biased_score_lookup[f] for f in shared_list]
        us_vals = [unbiased_score_lookup[f] for f in shared_list]
        axes[1].scatter(us_vals, bs_vals, alpha=0.6, color="tab:purple", s=40)
        lim = max(max(bs_vals), max(us_vals)) * 1.1
        axes[1].plot([0, lim], [0, lim], "k--", alpha=0.4, label="y=x")
        axes[1].set_xlabel("Unbiased score")
        axes[1].set_ylabel("Biased score")
        axes[1].set_title("Shared features: biased vs unbiased")
        axes[1].legend()

    # Panel 3: Layer distribution of spurious vs shared features
    max_layer = model.cfg.n_layers
    spurious_layers = [f[0] for f in spurious_ranked]
    shared_layers = [f[0] for f in shared_ranked]
    bins = np.arange(max_layer + 1) - 0.5
    axes[2].hist(spurious_layers, bins=bins, alpha=0.7, label="Spurious", color="tab:red")
    axes[2].hist(shared_layers, bins=bins, alpha=0.7, label="Causal", color="tab:blue")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Layer distribution of spurious vs causal features")
    axes[2].legend()

    plt.tight_layout()
    fig_path = f"/workspace/circuit-tracer/demos/probe_circuits/contrastive_probe_circuits_{PROBE_SLUG}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to {fig_path}", flush=True)

except ImportError:
    print("matplotlib not available; skipping visualization", flush=True)


def ablation_score_change_linear(
    graph_biased,
    graph_unbiased,
    spurious_features: list[tuple[int, int, int]],
) -> dict:
    """
    Estimate the score change from ablating spurious features using linear attribution.

    The attribution graph records the DIRECT EFFECT of each feature on the probe score.
    Summing these direct effects gives the total contribution of the spurious features.
    This is a first-order (linear) approximation: valid for the replacement model
    (which is approximately linear), and conservative for the full nonlinear model.

    Args:
        graph_biased: Attribution graph for the biased probe.
        graph_unbiased: Attribution graph for the unbiased probe.
        spurious_features: (layer, pos, feat_idx) triples to ablate.

    Returns:
        dict with direct attribution contributions per probe.
    """
    def get_direct_effects(graph, feature_set: set) -> float:
        """Sum of direct attribution scores for features in feature_set."""
        # Last row of adjacency matrix = probe target row
        # Columns 0..n_selected_features = direct effects from selected features
        n_selected = len(graph.selected_features)
        probe_row = graph.adjacency_matrix[-1, :n_selected]

        total = 0.0
        for i, feat_idx in enumerate(graph.selected_features.tolist()):
            feat_triple = tuple(graph.active_features[feat_idx].tolist())
            if feat_triple in feature_set:
                total += probe_row[i].item()
        return total

    feat_set = set(spurious_features)
    contribution_biased = get_direct_effects(graph_biased, feat_set)
    contribution_unbiased = get_direct_effects(graph_unbiased, feat_set)

    return {
        "biased_contribution": contribution_biased,
        "unbiased_contribution": contribution_unbiased,
    }


from circuit_tracer.graph import Graph, compute_partial_influences
from circuit_tracer.attribution.targets import LogitTarget


def attribute_probe_correct(
    prompt: str,
    model: ReplacementModel,
    probe: Probe,
    probe_layer: int = PROBE_LAYER,
    batch_size: int = 256,
    max_feature_nodes: int = 4096,
    verbose: bool = True,
) -> Graph:
    """
    Attribute backward from a mean-pooled linear probe at `probe_layer`.

    Unlike the CustomTarget approximation (which injects at the final layer),
    this function:
      - Injects W_probe / n_pos at each token position at _resid_activations[probe_layer+1]
        (the closest cached hook to the probe's application point)
      - Sums attribution rows over positions → one row for the probe target
      - The backward pass starts from layer probe_layer+1, so only features at
        layers 0..probe_layer are attributed (circuit truncated at probe_layer)

    Args:
        prompt: Input text.
        model: TransformerLens replacement model.
        probe: Trained Probe with .weight_vec property.
        probe_layer: Layer at whose output the probe is applied (default 22).
        batch_size: Max parallel backward passes (must be >= n_pos).
        max_feature_nodes: Max features to include in graph.
        verbose: Print progress.

    Returns:
        Graph object (probe target as the single logit node).

    Notes:
        _resid_activations[l] = blocks.l.hook_resid_mid (after attention, before MLP of block l).
        The probe is applied after block probe_layer's MLP. The closest available hook is
        _resid_activations[probe_layer+1] = blocks.(probe_layer+1).hook_resid_mid,
        which is after block (probe_layer+1)'s attention. The slight error from passing
        through frozen attention at block probe_layer+1 is systematic and consistent
        across both probes, so it cancels in the contrastive analysis.
    """
    assert model.backend == "transformerlens", "Only TransformerLens backend supported here"

    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()

    # Phase 0: setup attribution context
    log("Phase 0: Setting up attribution context")
    input_ids = model.ensure_tokenized(prompt)
    n_pos = input_ids.shape[0]
    ctx = model.setup_attribution(input_ids)

    activation_matrix = ctx.activation_matrix
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers = ctx.n_layers
    total_active_feats = activation_matrix._nnz()
    log(f"  n_pos={n_pos}, active features={total_active_feats}")

    # Phase 1: forward pass with attribution hooks installed
    log("Phase 1: Forward pass")
    with ctx.install_hooks(model):
        residual = model.forward(
            input_ids.expand(batch_size, -1),
            stop_at_layer=n_layers,
        )
        ctx._resid_activations[-1] = model.ln_final(residual)

    # Phase 2: build edge matrix structure
    log("Phase 2: Building attribution graph structure")
    logit_offset = total_active_feats + (n_layers + 1) * n_pos
    n_logits = 1  # one probe target
    total_nodes = logit_offset + n_logits

    actual_max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    edge_matrix = torch.zeros(actual_max_feature_nodes + n_logits, total_nodes)
    row_to_node_index = torch.zeros(actual_max_feature_nodes + n_logits, dtype=torch.int32)

    # Phase 3: probe attribution at probe_layer+1 (mean-pooled over all positions)
    log(f"Phase 3: Computing mean-pooled probe attribution at layer {probe_layer}")
    W = probe.weight_vec.to(model.cfg.device).to(model.cfg.dtype)  # (d_model,)

    # The attribution injection layer: probe_layer+1 is the closest cached hook
    # to h_{probe_layer} (the probe's actual application point).
    inject_layer = probe_layer + 1  # = 23 for PROBE_LAYER=22

    # For mean pooling: inject W / n_pos at each position, then sum rows
    probe_row = torch.zeros(1, logit_offset, dtype=W.dtype)
    inject_vals = (W / n_pos).unsqueeze(0).expand(n_pos, -1)  # (n_pos, d_model)

    # Process in chunks if n_pos > batch_size (rare for short bios)
    for chunk_start in range(0, n_pos, batch_size):
        chunk_end = min(chunk_start + batch_size, n_pos)
        chunk_size = chunk_end - chunk_start
        rows = ctx.compute_batch(
            layers=torch.full((chunk_size,), inject_layer, dtype=torch.long, device=model.cfg.device),
            positions=torch.arange(chunk_start, chunk_end, device=model.cfg.device),
            inject_values=inject_vals[chunk_start:chunk_end].to(model.cfg.device),
            retain_graph=True,  # Must retain — Phase 4 reuses the same graph
        )
        probe_row += rows.cpu().sum(0, keepdim=True)

    edge_matrix[0, :logit_offset] = probe_row[0]
    row_to_node_index[0] = logit_offset  # probe is the first (and only) logit node

    # Phase 4: feature attribution
    log("Phase 4: Computing feature attributions")
    probe_prob = torch.tensor([1.0])  # uniform weight
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=actual_max_feature_nodes, desc="Feature attribution", disable=not verbose)

    while n_visited < actual_max_feature_nodes:
        if actual_max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], probe_prob, row_to_node_index[:st]
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(4 * batch_size, actual_max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        for idx_batch in queue:
            n_visited += len(idx_batch)
            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=ctx.encoder_vecs[idx_batch],
                retain_graph=n_visited < actual_max_feature_nodes,
            )
            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))

    pbar.close()

    # Phase 5: package Graph
    log("Phase 5: Packaging graph")
    selected_features = torch.where(visited)[0]
    if actual_max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:actual_max_feature_nodes] = edge_matrix[:actual_max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[actual_max_feature_nodes:]

    probe_label = f"probe_layer{probe_layer}"
    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_targets=[LogitTarget(token_str=probe_label, vocab_idx=model.tokenizer.vocab_size)],
        logit_probabilities=probe_prob,
        vocab_size=model.tokenizer.vocab_size,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix.detach(),
        cfg=model.cfg,
        scan=model.scan,
    )

    log(f"Done in {time.time()-t0:.1f}s")
    return graph


# ── Stage 9: Correct attribution (mean-pooled probe injection) ─────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 9: Correct attribution (mean-pooled probe injection)", flush=True)
print("=" * 60, flush=True)

print("\nAttributing from BIASED probe (layer-{PROBE_LAYER} mean-pooled)...", flush=True)
graph_biased_correct = attribute_probe_correct(
    prompt=PROMPT,
    model=model,
    probe=probe_biased,
    probe_layer=PROBE_LAYER,
    batch_size=256,
    max_feature_nodes=4096,
    verbose=True,
)
print(f"Biased correct graph: {graph_biased_correct.active_features.shape[0]} active features, "
      f"{graph_biased_correct.selected_features.shape[0]} selected", flush=True)

print(f"\nAttributing from UNBIASED probe (layer-{PROBE_LAYER} mean-pooled)...", flush=True)
graph_unbiased_correct = attribute_probe_correct(
    prompt=PROMPT,
    model=model,
    probe=probe_unbiased,
    probe_layer=PROBE_LAYER,
    batch_size=256,
    max_feature_nodes=4096,
    verbose=True,
)
print(f"Unbiased correct graph: {graph_unbiased_correct.active_features.shape[0]} active features, "
      f"{graph_unbiased_correct.selected_features.shape[0]} selected", flush=True)


# ── Stage 10: Feature overlap analysis (correct) ──────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 10: Feature overlap analysis (correct attribution)", flush=True)
print("=" * 60, flush=True)

top_biased_c, scores_biased_c = get_top_features(graph_biased_correct, n=N_TOP)
top_unbiased_c, scores_unbiased_c = get_top_features(graph_unbiased_correct, n=N_TOP)

set_biased_c = set(top_biased_c)
set_unbiased_c = set(top_unbiased_c)

spurious_c = set_biased_c - set_unbiased_c
shared_c = set_biased_c & set_unbiased_c

print(f"Top-{N_TOP} features:", flush=True)
print(f"  Shared (causal):        {len(shared_c)}", flush=True)
print(f"  Spurious (biased-only): {len(spurious_c)}", flush=True)

# Check max layer of attributed features (should be ≤ PROBE_LAYER)
all_layers_c = [f[0] for f in top_biased_c + top_unbiased_c]
if all_layers_c:
    print(f"  Max attributed layer: {max(all_layers_c)} (should be ≤ {PROBE_LAYER})", flush=True)

biased_score_lookup_c = {feat: score for feat, score in zip(top_biased_c, scores_biased_c)}
spurious_ranked_c = sorted(spurious_c, key=lambda f: biased_score_lookup_c[f], reverse=True)

print("\nTop spurious features (corrected attribution):", flush=True)
print(f"{'Layer':>6} {'Pos':>5} {'FeatIdx':>8} {'Biased Score':>14}", flush=True)
print("-" * 40, flush=True)
for layer, pos, feat_idx in tqdm(spurious_ranked_c[:20], desc="Fetching spurious features", unit="feat"):
    score = biased_score_lookup_c[(layer, pos, feat_idx)]
    neuronpedia_url = f"https://neuronpedia.org/{NEURONPEDIA_MODEL_ID}/{NEURONPEDIA_SAE_ID}/{feat_idx}"
    feat_api_url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/{NEURONPEDIA_SAE_ID}/{feat_idx}"
    feat_data =  requests.get(feat_api_url).json()
    explanations = feat_data.get("explanations", [])
    explanation_strs = (
        " |OR| ".join(exp.get("description", "No explanation") for exp in explanations)
        if explanations
        else "No explanation available"
    )
    print(f"{layer:>6} {pos:>5} {feat_idx:>8} {score:>14.4f}, {explanation_strs}, {neuronpedia_url}", flush=True)


# ── Stage 11: Ablation analysis ────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 11: Ablation analysis (linear approximation)", flush=True)
print("=" * 60, flush=True)

print(f"Baseline biased probe score:   {score_biased:+.3f}", flush=True)
print(f"Baseline unbiased probe score: {score_unbiased:+.3f}", flush=True)

for n_abl in [5, 10, 20]:
    feats_to_abl = spurious_ranked_c[:n_abl]
    result = ablation_score_change_linear(
        graph_biased_correct,
        graph_unbiased_correct,
        spurious_features=feats_to_abl,
    )
    b_contrib = result["biased_contribution"]
    u_contrib = result["unbiased_contribution"]
    print(f"\nAblating top-{n_abl:2d} spurious features:", flush=True)
    print(f"  Biased   probe: {score_biased:+.3f} → ~{score_biased - b_contrib:+.3f}  "
          f"(spurious contribution: {b_contrib:+.3f})", flush=True)
    print(f"  Unbiased probe: {score_unbiased:+.3f} → ~{score_unbiased - u_contrib:+.3f}  "
          f"(contribution:  {u_contrib:+.3f})", flush=True)

print("\nInterpretation:", flush=True)
print("  Spurious features should contribute STRONGLY to biased probe but WEAKLY to unbiased.", flush=True)
print("  A large biased contribution relative to unbiased confirms spurious signal in the biased probe.", flush=True)


# ── Stage 12: Save correct attribution graphs ──────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("STAGE 12: Saving correct attribution graphs", flush=True)
print("=" * 60, flush=True)

from circuit_tracer.utils import create_graph_files

output_dir = Path("probe_circuits")
output_dir.mkdir(exist_ok=True)

for slug, graph in tqdm([
    (f"{PROBE_SLUG}-biased-probe-correct", graph_biased_correct),
    (f"{PROBE_SLUG}-unbiased-probe-correct", graph_unbiased_correct),
], desc="Saving graphs", unit="graph"):
    print(f"Saving {slug}...", flush=True)
    graph.to_pt(output_dir / f"{slug}.pt")
    create_graph_files(
        graph_or_path=graph,
        slug=slug,
        output_path=str(output_dir / "graph_files"),
        node_threshold=0.8,
        edge_threshold=0.98,
    )

print(f"Saved correct attribution graphs to {output_dir}/graph_files/", flush=True)
print("Done.", flush=True)

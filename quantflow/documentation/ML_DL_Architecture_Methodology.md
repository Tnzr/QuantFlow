# Multimodal Stateful Cascading Multi-Temporal Token Context: A Domain-Agnostic ML/DL Architecture and Methodology

**Version:** 1.0  
**Date:** 2026-07-15  
**Status:** Canonical reference — signal-agnostic, application-agnostic  
**Supersedes:** All application-specific methodology fragments; this is the transferable core.

---

## Table of Contents

1. [Scope and Objective](#1-scope-and-objective)
2. [Problem Framing](#2-problem-framing)
3. [Architectural Blueprint](#3-architectural-blueprint)
4. [Multi-Temporal Tokenization and Stateful Context](#4-multi-temporal-tokenization-and-stateful-context)
5. [Dual-Path Inference: Certainty Generator and Anticipation Trajectory](#5-dual-path-inference)
6. [Complex Systems Engineering Loss (Core)](#6-complex-systems-engineering-loss)
7. [Model Zoo: Deployment-Target Architectures](#7-model-zoo)
8. [Training Curriculum and Protocol](#8-training-curriculum-and-protocol)
9. [Validation and Evaluation Framework](#9-validation-and-evaluation-framework)
10. [Inference and Deployment](#10-inference-and-deployment)
11. [Transferability Framework](#11-transferability-framework)
12. [Ablation and Interpretability](#12-ablation-and-interpretability)
13. [Implementation Checklist](#13-implementation-checklist)
14. [Glossary of Domain-Agnostic Terms](#14-glossary)

---

## 1. Scope and Objective

### 1.1 Purpose

This document defines a **unified, signal-agnostic, domain-transferable architecture and methodology** for building systems that perform:

1. **Real-time event-state classification** — determining the current phase relative to a critical event (normal, approaching, imminent).
2. **Probabilistic future-event forecasting** — predicting the time-to-event distribution over a horizon.
3. **Uncertainty-aware, coherent dual-head inference** — ensuring the forecast remains consistent with what the current-state classifier has already confirmed.

### 1.2 Design Principles

| Principle | Description |
|-----------|-------------|
| **Signal agnosticism** | The architecture accepts any time-series modality (1D signals, feature streams, sensor readings). No modality-specific assumptions are hard-coded. |
| **Domain agnosticism** | Replace "seizure," "failure," "drawdown," or "anomaly" with any critical event definition. The methodology is invariant to the event semantics. |
| **Stateful temporal reasoning** | Explicit memory across sequence chunks — not window-independent training. The system maintains causal context over long horizons. |
| **Coherent dual-path inference** | Past-state certainty and future-risk forecast are jointly optimized with a coherence penalty. |
| **Uncertainty calibration** | Every prediction is accompanied by a calibrated uncertainty estimate suitable for downstream decision policies. |
| **Deployment-aware model zoo** | A spectrum of architectures from microcontroller (3K parameters) to cloud (1.5M parameters) shares the same training methodology. |

### 1.3 When to Use This Methodology

This methodology applies when your system must:

- Monitor streaming time-series data from one or more sensors/modalities.
- Classify whether the system is in a normal, approaching-event, or imminent-event state.
- Predict how far away a critical event is (time-to-event / countdown).
- Produce uncertainty estimates suitable for automated alerting or decision thresholds.
- Operate causally — no future information may leak into the current prediction.
- Maintain coherent beliefs across the past-state and future-forecast heads.

---

## 2. Problem Framing

### 2.1 Generalized Event Model

Let a **critical event** be any point in time where the system transitions from a pre-event phase into an event phase. We define three temporal states:

| State | Definition | Label Convention |
|-------|-----------|-----------------|
| **Inter-event** (normal) | No event anticipated within the forecast horizon. The system is in its baseline regime. | $\tau < 0$ |
| **Pre-event** (approaching) | An event is anticipated within the forecast horizon but not yet imminent. | $\tau \in [\theta, H]$ |
| **Onset** (imminent) | The event is imminent — within a short critical window before occurrence. | $\tau \in [0, \theta)$ |

Where:
- $\tau$ = time remaining until the next event (minutes, seconds, or domain-appropriate units).
- $\theta$ = onset window threshold (the boundary between "approaching" and "imminent").
- $H$ = maximum forecast horizon.

**Key requirement:** Labels are derived from event timestamps only — no future information is encoded in model weights beyond what is available at inference time.

### 2.2 Input Modality Abstraction

Any time-series modality is abstracted as a stream:

$$\mathcal{M} = \{(\mathbf{x}_t^{(m)}, t)\}_{t=0}^{T}$$

where:
- $m \in \{1, \ldots, M\}$ indexes modalities.
- $\mathbf{x}_t^{(m)} \in \mathbb{R}^{d_m}$ is the feature vector at time $t$ for modality $m$.
- $d_m$ = dimensionality of modality $m$ (raw signal channels, extracted features, or embeddings).

Modalities can be:
- Raw sensor signals (vibration, acoustic, electrical, optical).
- Extracted feature streams (statistical moments, spectral bands, domain-specific indicators).
- Pre-trained embeddings from upstream models.
- Exogenous data (time-of-day, environmental readings, contextual metadata).

**The architecture makes no assumption about what each modality encodes** — it learns modality-specific representations through dedicated encoders before fusion.

### 2.3 Causal Constraint

All predictions at time $t$ may only depend on observations up to time $t$:

$$P(\text{event} \mid \text{data}_{\le t})$$

This is enforced architecturally through:
1. Causal attention masks in transformer layers.
2. Forward-only recurrent/state-space state propagation.
3. Temporal ordering guarantees in data loading (chronological, patient/sequence-sequential).

---

## 3. Architectural Blueprint

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL INPUT STREAMS                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │Modality 1│  │Modality 2│  │Modality 3│  │   ...    │             │
│  │ (stream) │  │ (stream) │  │ (stream) │  │          │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │             │             │             │                    │
│  ┌────▼─────────────▼─────────────▼─────────────▼────┐              │
│  │          MODALITY-SPECIFIC ENCODERS                │              │
│  │   (LSTM / TCN / Transformer — one per modality)    │              │
│  └──────────────────────┬────────────────────────────┘              │
│                         │ z_t^(1), z_t^(2), ..., z_t^(M)             │
│  ┌──────────────────────▼────────────────────────────┐              │
│  │            MODALITY FUSION (Attention Gate)        │              │
│  │                 → Token u_t                       │              │
│  └──────────────────────┬────────────────────────────┘              │
│                         │                                           │
│  ┌──────────────────────▼────────────────────────────┐              │
│  │         MULTI-SCALE TOKEN CASCADE                  │              │
│  │                                                    │              │
│  │   u_t^(S0) ──► u_t^(S1) ──► u_t^(S2) ──► u_t^(S3) │              │
│  │    (short)     (minutes)    (hours)     (long)     │              │
│  │       │            │           │           │        │              │
│  │       └────────────┼───────────┼───────────┘        │              │
│  │              UNet-like Skip Connections              │              │
│  └──────────────────────┬────────────────────────────┘              │
│                         │                                           │
│  ┌──────────────────────▼────────────────────────────┐              │
│  │            CROSS-SCALE FUSION DECODER              │              │
│  └──┬──────────┬──────────────┬──────────────┬───────┘              │
│     │          │              │              │                       │
│  ┌──▼───┐ ┌───▼────┐ ┌──────▼──────┐ ┌─────▼──────┐                │
│  │Past- │ │Future  │ │Uncertainty  │ │Dynamics    │                │
│  │State │ │Forecast│ │/Bayesian    │ │Residual    │                │
│  │Head  │ │Head    │ │Head         │ │Head        │                │
│  └──┬───┘ └───┬────┘ └──────┬──────┘ └─────┬──────┘                │
│     │         │             │              │                        │
│  ┌──▼─────────▼─────────────▼──────────────▼──────┐                │
│  │     COHERENCE + DISTANCE-AWARE OBJECTIVE        │                │
│  └────────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Specifications

#### 3.2.1 Modality-Specific Encoders

For each modality $m$, a dedicated encoder $E_m$ maps the raw stream to a latent representation:

$$\mathbf{z}_t^{(m)} = E_m(\mathbf{x}_{\le t}^{(m)}; \phi_m)$$

**Supported encoder types (per-modality):**

| Encoder | Parameters | Best For | Characteristics |
|---------|-----------|----------|-----------------|
| **BiLSTM** | ~100K–300K | Medium-length sequences (seconds–minutes) | Bidirectional context, mature optimization |
| **TCN** (Temporal ConvNet) | ~280K | Long sequences, fast training | Dilated causal convolutions, parallelizable |
| **Transformer** | ~500K+ | Very long-range dependencies | Self-attention, best accuracy ceiling |
| **CNN-LSTM Hybrid** | ~150K | Local patterns + temporal context | Convolutional feature extraction + recurrence |
| **InceptionTime-1D** | ~430K | Multi-scale temporal patterns | Parallel convolutional kernels at multiple scales |

**Selection guideline:** Choose based on deployment constraints and sequence length, not on modality type. The same encoder architecture can process vibration, price, ECG, or any 1D feature stream equally well.

#### 3.2.2 Modality Fusion (Attention Gate)

Encoded modality representations are fused via a learned attention gate:

$$\mathbf{u}_t = \sum_{m=1}^{M} \alpha_t^{(m)} \cdot \mathbf{z}_t^{(m)}$$

$$\alpha_t^{(m)} = \text{softmax}\left(\mathbf{w}^\top \tanh(\mathbf{W} [\mathbf{z}_t^{(1)} \| \cdots \| \mathbf{z}_t^{(M)}])\right)$$

This allows the model to dynamically weight modalities based on signal quality, informativeness, and contextual relevance. When a modality is missing or corrupted, the gate can learn to down-weight it automatically.

**Fusion alternatives:**
- **Early fusion:** Concatenate raw features before encoding (simpler, but no per-modality specialization).
- **Late fusion (gated):** Encode each modality independently, then fuse via learned gate (recommended).
- **Cross-attention fusion:** Use one modality's encoding to query another (for paired modalities).

#### 3.2.3 Multi-Scale Token Cascade

The fused token $\mathbf{u}_t$ is processed through a cascade of temporal scales:

$$\mathbf{u}_t^{(S0)} \rightarrow \mathbf{u}_t^{(S1)} \rightarrow \mathbf{u}_t^{(S2)} \rightarrow \mathbf{u}_t^{(S3)}$$

| Scale | Temporal Context | Receptive Field | Purpose |
|-------|-----------------|-----------------|---------|
| **S0** | Seconds–sub-minute | Short | Sharp signal morphology, rapid changes |
| **S1** | Minutes | Medium | Event build-up patterns, trend emergence |
| **S2** | Hours | Long | Circadian/session-level context, baseline drifts |
| **S3** | Multi-hour / session | Very long | Cross-session memory, long-term regime detection |

Each scale maintains its own hidden state $\mathbf{h}_t^{(s)}$, propagated forward causally:

$$\mathbf{h}_t^{(s)} = f_s(\mathbf{h}_{t-1}^{(s)}, \mathbf{u}_t^{(s)})$$

**UNet-like skip connections** pass fine-grained representations from earlier scales directly to the fusion decoder, preventing information loss through the cascade bottleneck.

#### 3.2.4 Cross-Scale Fusion Decoder

The fusion decoder aggregates multi-scale representations:

$$\mathbf{f}_t = \text{FusionDecoder}\left(\mathbf{u}_t^{(S0)}, \mathbf{u}_t^{(S1)}, \mathbf{u}_t^{(S2)}, \mathbf{u}_t^{(S3)}\right)$$

Implementation options:
- **Concatenation + MLP** (simplest, recommended for initial implementation).
- **Cross-attention** over scale representations (higher capacity).
- **Gated sum with learned scale weights** (interpretable, lightweight).

#### 3.2.5 Prediction Heads

Four heads operate on the fused representation $\mathbf{f}_t$:

**1. Past-State Classification Head:** $p_{\text{past}}(y_{\le t} \mid \mathbf{f}_t)$
- Binary or multi-class: inter-event vs. pre-event vs. onset.
- Acts as the **certainty generator** — its confidence gates the forecast head.

**2. Future Forecast Head:** $p_{\text{fut}}(y_{t:t+H} \mid \mathbf{f}_t, p_{\text{past}})$
- Predicts event risk/time-to-event distribution over horizon $H$.
- Conditioned on past-state head output (coherent heads architecture).

**3. Uncertainty / Bayesian Head:** $\sigma_t = g_\sigma(\mathbf{f}_t)$
- Produces prediction uncertainty (epistemic or aleatoric).
- Feeds the uncertainty calibration loss.

**4. Dynamics Residual Head:** $\mathbf{r}_t = g_r(\mathbf{f}_t)$
- Models the innovation residual between predicted and observed state transitions.
- Used in the Neural Kalman-style dynamics correction.

---

## 4. Multi-Temporal Tokenization and Stateful Context

### 4.1 Token Generation

At each time step $t$, the fused token $\mathbf{u}_t \in \mathbb{R}^{d_{\text{token}}}$ is generated from modality encodings as described in §3.2.2. This token is the fundamental unit of information passed through the temporal cascade.

### 4.2 Scale Construction

Multi-scale token streams are generated through hierarchical temporal aggregation:

$$\mathbf{u}_t^{(s+1)} = \text{Aggregate}\left(\{\mathbf{u}_{t-k}^{(s)}\}_{k=0}^{K_s}\right)$$

Where $K_s$ is the aggregation window for scale $s$. Aggregation functions include:
- **Strided temporal convolution** (learned aggregation).
- **Attention-based pooling** (adaptive aggregation).
- **Exponential moving average** (fixed, interpretable).

### 4.3 Stateful Memory

Each scale maintains a recurrent/state-space carry variable:

$$\mathbf{h}_t^{(s)} = \text{StateUpdate}^{(s)}\left(\mathbf{h}_{t-1}^{(s)}, \mathbf{u}_t^{(s)}\right)$$

Implementation options for state update:
- **LSTM/GRU cells** — proven, widely available.
- **State-Space Models (S4/Mamba)** — efficient for very long sequences.
- **Linear recurrent units** — fast, parallelizable training.

**Critical property:** State is carried across data-loader batch boundaries during training. This requires:
1. Chronological data ordering (no random shuffling across time).
2. Hidden state detach/transfer between batches.
3. Sequence-level (not sample-level) sampling.

### 4.4 Causal Validity

The cascade guarantees that $\mathbf{h}_t^{(s)}$ depends only on observations at times $\le t$. This is enforced by:
- Causal convolutions (no future padding in TCN layers).
- Forward-direction RNN unrolling.
- Causal attention masks in transformer layers.

---

## 5. Dual-Path Inference: Certainty Generator and Anticipation Trajectory

### 5.1 Design Rationale

A system that only forecasts the future without confirming the present state is prone to hallucination. A system that only classifies the present without anticipating the future provides no lead time. The dual-path design addresses both.

### 5.2 Past-State Head (Certainty Generator)

The past-state head answers: *"What phase are we in right now with respect to a critical event?"*

**Output form:** A probability distribution over $K$ states:
- $p_0$: inter-event (normal)
- $p_1$: pre-event (approaching)
- $p_2$: onset (imminent)

**Architecture:** MLP head on $\mathbf{f}_t$, typically 2–3 layers with ReLU/GELU + Dropout.

### 5.3 Future Forecast Head (Anticipation Trajectory)

The future forecast head answers: *"How far away is the next critical event?"*

**Output form:** A probability distribution over the forecast horizon — either:
- **Continuous:** predicted time-to-event $\hat{\tau}_t \in [0, H]$ (regression).
- **Discretized:** probabilities over $B$ time buckets (classification over bins).
- **Parametric:** parameters of a distribution (e.g., Weibull, log-normal) over time-to-event.

**Coherent conditioning:** The forecast head receives the pre-activation logits from the past-state head as an additional input, ensuring the forecast does not contradict the current-state assessment.

### 5.4 Coherence Requirement

If past-state confidence is high for a stable inter-event context, future-risk probability should not spike without dynamics evidence. If past-state indicates pre-event escalation, future-risk must reflect that trend.

Formally:

$$\text{KL}\left(p_{\text{fut}}(y_{t:t+H} \mid \mathbf{f}_t) \;\|\; g(p_{\text{past}}(y_{\le t}), \Delta\mathbf{h}_t)\right) < \epsilon$$

where $g(\cdot)$ maps past certainty and state innovation $\Delta\mathbf{h}_t$ to an expected forecast prior.

---

## 6. Complex Systems Engineering Loss (Core)

### 6.1 Composite Objective

The total loss is a weighted composition of six terms:

$$\mathcal{L}_{\text{total}} = \sum_{k} \lambda_k \mathcal{L}_k$$

with weights normalized: $\sum_k \lambda_k = 1$ (after normalization).

### 6.2 Loss Components

#### 6.2.1 Event-State Classification Loss ($\mathcal{L}_{\text{cls}}$)

$$\mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^{N} w_i \cdot \ell(y_i, \hat{y}_i)$$

Where:
- $y_i \in \{0, 1\}$ or $\{0, 1, 2\}$ (binary or multi-class event state).
- $\hat{y}_i$ is the predicted probability from the past-state head.
- $\ell$ is focal loss (recommended for class imbalance) or cross-entropy.
- $w_i$ are per-sample weights addressing class imbalance.

**Focal loss variant:**
$$\ell_{\text{focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's assigned probability to the true class, $\alpha_t$ balances positive/negative, and $\gamma$ focuses learning on hard examples.

**Class-weighting strategy:** Dynamically compute positive/negative ratio per batch and set `pos_weight` accordingly. Apply onset-aware boosting: samples within the onset window receive higher weight than far-pre-event samples.

#### 6.2.2 Future Forecast Loss ($\mathcal{L}_{\text{fut}}$)

For regression-based countdown prediction:

$$\mathcal{L}_{\text{fut}} = \frac{1}{N_{\text{active}}} \sum_{i: y_i \in \text{active}} w_i^{\text{temp}} \cdot \rho(\hat{\tau}_i - \tau_i)$$

Where:
- $\hat{\tau}_i$ = predicted time-to-event.
- $\tau_i$ = true time-to-event.
- $\rho$ = robust loss function (Huber, Smooth L1).
- $w_i^{\text{temp}}$ = temporal weighting emphasizing near-onset predictions.
- Active samples = pre-event + onset (inter-event samples excluded from regression).

**Temporal weighting function:**
$$w^{\text{temp}}(\tau) = 1 + \beta \cdot \exp(-\tau / \tau_0)$$

where $\beta$ controls the boost magnitude and $\tau_0$ controls the focus time constant. Clipped to $[1, w_{\max}]$.

#### 6.2.3 Distance-to-Anticipated-Event Loss ($\mathcal{L}_{\text{dist}}$)

$$\mathcal{L}_{\text{dist}} = w(\tau) \cdot \rho(\hat{\tau} - \tau)$$

with $w(\tau)$ increasing as $\tau \to 0$ (event approaches). This gives direct feedback to the uncertainty head on proximity to anticipated events.

**Purpose:** Separates the "how far" regression signal from the "how certain" uncertainty signal, preventing them from confounding each other.

#### 6.2.4 Dual-Head Coherence Loss ($\mathcal{L}_{\text{coh}}$)

$$\mathcal{L}_{\text{coh}} = \text{KL}\left(p_{\text{fut}}(y_{t:t+H}) \;\|\; g(p_{\text{past}}(y_{\le t}), \Delta\mathbf{h}_t)\right)$$

**Practical implementation:** Instead of full KL, use a contrastive penalty:
- If past-state = inter-event (high confidence) and forecast = high risk → penalty.
- If past-state = pre-event (high confidence) and forecast = no risk → penalty.
- If past-state and forecast agree → no penalty.

**Simplified surrogate:**
$$\mathcal{L}_{\text{coh}} = \| \sigma(\text{fut\_logit}) - \text{past\_prob} \|_2^2$$

when simplified for regression + classification dual-head setups.

#### 6.2.5 Dynamics Residual Correction Loss ($\mathcal{L}_{\text{dyn}}$)

Neural Kalman / Neural Process style innovation penalty:

$$\mathbf{h}_{t|t-1} = f_\theta(\mathbf{h}_{t-1}, \mathbf{u}_t)$$
$$\mathbf{h}_t = \mathbf{h}_{t|t-1} + \mathbf{K}_t \mathbf{r}_t$$

Where $\mathbf{r}_t$ is the innovation residual from observed modalities and $\mathbf{K}_t$ is a learned gain.

$$\mathcal{L}_{\text{dyn}} = \|\mathbf{r}_t\|_2^2 + \lambda_{\text{smooth}} \|\mathbf{h}_t - \mathbf{h}_{t-1}\|_2^2$$

Penalizes unstable or incoherent residual corrections while allowing the model to learn adaptive state updates.

#### 6.2.6 Uncertainty Calibration Loss ($\mathcal{L}_{\text{unc}}$)

$$\mathcal{L}_{\text{unc}} = \mathcal{L}_{\text{NLL}} + \lambda_{\text{ECE}} \cdot \text{ECE}(\hat{p}, y)$$

Where:
- $\mathcal{L}_{\text{NLL}}$ = negative log-likelihood under the predicted distribution.
- $\text{ECE}$ = Expected Calibration Error (binned reliability metric).

**For regression uncertainty:** Use a Gaussian NLL where the model outputs both mean $\hat{\tau}$ and variance $\hat{\sigma}^2$:

$$\mathcal{L}_{\text{NLL}} = \frac{1}{2} \left[\log(\hat{\sigma}^2) + \frac{(\hat{\tau} - \tau)^2}{\hat{\sigma}^2}\right]$$

### 6.3 Safe State Classification Mode

For applications where regressing exact time-to-event values would encode future-event timing in network weights (undesirable for safety-critical or memory-contaminated settings), use the **safe 3-state mode**:

- The forecast head is repurposed as an **onset head**: $P(\text{state} = \text{onset} \mid \text{data})$.
- No countdown value is regressed.
- Total loss becomes: $\mathcal{L}_{\text{total}} = \lambda_{\text{alert}} \mathcal{L}_{\text{alert}} + \lambda_{\text{onset}} \mathcal{L}_{\text{onset}}$.
- This avoids encoding precise future timing while still distinguishing "approaching" from "imminent."

### 6.4 Ranking / Monotonicity Constraint

When regression is used, enforce that predicted countdown decreases monotonically as the event approaches:

$$\mathcal{L}_{\text{rank}} = \frac{1}{N_{\text{pairs}}} \sum_{(i,j) \in \text{consecutive}} \text{ReLU}(\hat{\tau}_j - \hat{\tau}_i) \cdot \mathbb{1}[\text{both active}]$$

### 6.5 Loss Weighting Configuration

| Loss Term | Symbol | Typical Weight | Sensitivity |
|-----------|--------|---------------|-------------|
| Classification | $\mathcal{L}_{\text{cls}}$ | 0.35 | High — drives event detection |
| Forecast / Regression | $\mathcal{L}_{\text{fut}}$ | 0.25 | Medium — drives timing accuracy |
| Distance-to-Event | $\mathcal{L}_{\text{dist}}$ | 0.10 | Low — fine-tunes proximity signal |
| Coherence | $\mathcal{L}_{\text{coh}}$ | 0.10 | Low–Medium — prevents head divergence |
| Dynamics Residual | $\mathcal{L}_{\text{dyn}}$ | 0.10 | Low — regularizes state transitions |
| Uncertainty | $\mathcal{L}_{\text{unc}}$ | 0.10 | Low — calibrates confidence |
| Ranking | $\mathcal{L}_{\text{rank}}$ | 0.05 | Low — enforces monotonicity |

**Tuning principle:** Start with classification and regression as the dominant terms. Phase in coherence and uncertainty terms gradually after the primary heads converge. Use ablation studies to finalize weights per application.

---

## 7. Model Zoo: Deployment-Target Architectures

The methodology supports a spectrum of architectures sharing the same training protocol but optimized for different deployment targets.

### 7.1 Architecture Catalog

| Architecture | Parameters | Target Hardware | Characteristics |
|-------------|-----------|----------------|-----------------|
| **EEGNet-style** | ~3K | Microcontroller (ESP32, nRF52) | Depthwise-separable convolutions; INT8 quantizes to ~3 KB |
| **MobileNet-1D** | ~60K | Smartwatch, IoT gateway | Depthwise-separable CNN; INT8 quantizes to ~60 KB |
| **TCN** | ~280K | Wearable, edge server | Dilated causal convolutions; fully parallelizable |
| **InceptionTime-1D** | ~430K | Edge server | Multi-scale parallel convolutions |
| **BiLSTM + Attention** | ~200K–300K | Server, cloud | Bidirectional LSTM with temporal attention |
| **Temporal Transformer** | ~500K | Cloud (single modality) | Self-attention; best long-range dependency capture |
| **Multimodal Transformer** | ~1.5M | Cloud (multimodal) | Cross-attention fusion of 3+ modalities |

### 7.2 Shared Properties

All architectures in the zoo share:
1. **Dual-head output** (past-state classification + future forecast).
2. **Coherent heads option** (forecast head receives classification logits).
3. **The same loss interface** — swap models without changing the training loop.
4. **Standardized weight initialization** (Xavier uniform for linear, Kaiming for convolutional).

### 7.3 Architecture Selection Flowchart

```
Need real-time on <1MB RAM device?
├── Yes → EEGNet-style (3K params, INT8 quantize)
└── No → Need on-device wearable inference?
    ├── Yes → MobileNet-1D (60K params, TFLite/CoreML)
    └── No → Have long sequences (>10K steps)?
        ├── Yes → TCN or Temporal Transformer
        └── No → Need multimodal fusion?
            ├── Yes → Multimodal Transformer
            └── No → BiLSTM + Attention (proven baseline)
```

---

## 8. Training Curriculum and Protocol

### 8.1 Four-Phase Curriculum

#### Phase A: Self-Supervised Pretraining (Optional)

**Goal:** Learn useful representations without labels.

**Methods:**
- **Masked reconstruction:** Mask random segments of each modality stream, predict the masked values.
- **Temporal contrastive learning:** Pull together representations of temporally adjacent windows, push apart distant ones.
- **Modality alignment:** Align representations across modalities that co-occur in time.

**When to use:** When labeled event data is scarce. Skip if sufficient labeled data is available.

#### Phase B: Supervised Dual-Head Training

**Goal:** Train the full architecture with the composite loss.

**Protocol:**
1. Initialize with pretrained weights from Phase A (or random if skipped).
2. Train all components end-to-end with the composite loss (§6).
3. Use a warm-up schedule: start with only $\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{fut}}$, phase in other terms after $N_{\text{warmup}}$ epochs.
4. Apply class-weighting for imbalance (dynamic pos_weight per batch).
5. Monitor validation metrics after each epoch.

**Recommended optimizer:** AdamW with cosine annealing + linear warmup.

**Learning rate:**
- $1\mathrm{e}{-3}$ initial (small models: EEGNet, MobileNet).
- $1\mathrm{e}{-4}$ initial (medium models: TCN, InceptionTime, BiLSTM).
- $5\mathrm{e}{-5}$ initial (large models: Transformers).

**Batch size:** As large as memory allows. For stateful models, batch size affects hidden state granularity.

**Gradient accumulation:** Use when batch size is constrained by memory. Effective batch size = batch_size × accumulation_steps.

#### Phase C: Calibration Pass

**Goal:** Calibrate prediction uncertainties and optimize decision thresholds.

**Protocol:**
1. Freeze or use a very low learning rate on the feature extractor.
2. Train/optimize the uncertainty head with $\mathcal{L}_{\text{unc}}$ as primary objective.
3. Compute optimal decision thresholds on a held-out calibration set (maximize balanced accuracy, minimize false-alert rate subject to sensitivity constraints).
4. Apply temperature scaling or isotonic regression for probability calibration.

**Threshold selection:** Sweep thresholds on the calibration set and select the one that maximizes a domain-specific utility function (e.g., $0.7 \cdot \text{sensitivity} + 0.3 \cdot \text{specificity}$).

#### Phase D: Longitudinal Deployment-Style Replay

**Goal:** Validate the model under streaming conditions that simulate real deployment.

**Protocol:**
1. Feed data in strict chronological order (no shuffling).
2. Maintain and propagate hidden state across the entire sequence.
3. Apply the causal prediction smoother (§9.4) on the raw outputs.
4. Measure metrics in a streaming fashion (cumulative metrics over time).
5. Test for prediction stability, state drift, and memory degradation.

### 8.2 Data Loading Requirements

#### Chronological Ordering

Samples must be presented in temporal order. For multi-sequence datasets:
1. Group samples by sequence/recording/session identifier.
2. Sort within each group by timestamp.
3. Present groups in a consistent order (not randomly shuffled across groups, unless explicitly testing cross-group generalization).

#### Stateful Data Loading

For stateful models:
1. Use a stateful data loader that carries hidden states across batch boundaries.
2. Detach hidden states between sequences to prevent gradient flow across sequence boundaries.
3. Implement hidden state managers that track $(h, c)$ tuples per sequence.

#### Class Imbalance Handling

Critical events are typically rare. Strategies:
1. **Focal loss** (§6.2.1) — built-in imbalance handling.
2. **Dynamic class weighting** — compute pos_weight per batch.
3. **Balanced sampling** — oversample pre-event/onset samples, undersample inter-event.
4. **Data augmentation** — time-warp, amplitude-scale, add noise, time-shift for minority class.
5. **Online augmentation** — stochastic augmentation during data loading without changing dataset size.

### 8.3 Distributed Training

**Multi-GPU (DDP):**
1. Use `DistributedDataParallel` with `DistributedSampler`.
2. Ensure class weights are computed consistently across ranks (all-reduce pos/neg counts).
3. For stateful models: synchronize hidden states at sequence boundaries or keep per-rank state independent.

**Mixed precision:** Use automatic mixed precision (AMP) with gradient scaling for memory efficiency on supported hardware.

### 8.4 Regularization

| Technique | Application |
|-----------|------------|
| **Dropout** | Applied after each major layer (typical rate: 0.2–0.3) |
| **Weight decay** | AdamW decoupled weight decay (1e-4 to 1e-5) |
| **Gradient clipping** | Clip gradients to max norm (typical: 1.0–5.0) |
| **Label smoothing** | Smooth binary targets (typical: 0.05–0.1) |
| **Early stopping** | Patience of 10–20 epochs on validation loss |
| **Stochastic depth** | Randomly drop layers during training (for deep models) |

---

## 9. Validation and Evaluation Framework

### 9.1 Mandatory Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Event sensitivity** (recall) | Fraction of events correctly anticipated | Maximize subject to FPR constraint |
| **False-alert rate** (FPR) | Alerts raised during inter-event periods | Minimize subject to sensitivity constraint |
| **Balanced accuracy** | $0.5 \cdot (\text{sensitivity} + \text{specificity})$ | Maximize |
| **Precision** | Fraction of alerts that are true events | High (reduces alert fatigue) |
| **F1-score** | Harmonic mean of precision and recall | Maximize |
| **Horizon-wise distance-to-event error** | MAE / MedAE of time-to-event predictions | Minimize (in domain units) |
| **Coherence error** | Divergence between past-state and future-forecast heads | Minimize |
| **Uncertainty calibration** | ECE, NLL, Brier score | ECE < 0.05 (well-calibrated) |
| **Prediction stability** | Variance of predictions under streaming conditions | Low jitter |
| **Lead time** | How far in advance events are detected | Maximize useful lead time |

### 9.2 Per-Sequence / Per-Group Analysis

Report metrics broken down by sequence/group (patient, machine, portfolio, etc.):
- Identify hardest and easiest sequences.
- Compute difficulty score: weighted combination of classification error and regression MAE.
- Flag sequences with unusually high error for targeted investigation.

### 9.3 Confusion Matrix (3-State)

For 3-state classification (inter-event, pre-event, onset), report:
- Full $3 \times 3$ confusion matrix with row-normalized percentages.
- Per-class recall (diagonal entries).
- Most common misclassifications (off-diagonal).

### 9.4 Causal Prediction Smoother

For deployment, apply a zero-retraining causal smoother on raw model outputs:

$$\tilde{p}_t = \alpha_t p_t + (1 - \alpha_t) \tilde{p}_{t-1} + \text{streak\_bonus}_t$$

Where:
- $\alpha_t = \alpha_{\text{rise}}$ if $p_t \ge \tilde{p}_{t-1}$ (fast escalation).
- $\alpha_t = \alpha_{\text{fall}}$ if $p_t < \tilde{p}_{t-1}$ (slow decay, avoids flickering).
- $\text{streak\_bonus}_t$ adds a small confidence increment when $K$ consecutive windows exceed a threshold.

**Typical parameters:**
- $\alpha_{\text{rise}} = 0.50$ (fast alarm escalation).
- $\alpha_{\text{fall}} = 0.12$ (slow alarm decay).
- Streak threshold = 0.30, streak window = 8, max bonus = 0.15.

### 9.5 Decision Policy

Convert model outputs into actionable alerts:

1. **Alert trigger:** Smoothed past-state probability > threshold $\theta_{\text{alert}}$.
2. **Alert persistence:** Alert remains active for at least $T_{\text{min\_alert}}$ after trigger.
3. **Cooldown:** After alert ends, suppress new alerts for $T_{\text{cooldown}}$ to prevent rapid toggling.
4. **Escalation:** If forecast head predicts $\hat{\tau} < \theta_{\text{onset}}$ AND past-state confirms pre-event, escalate to high-priority alert.

**Threshold optimization:** Choose $\theta_{\text{alert}}$ to maximize a utility function:
$$U(\theta) = \text{sensitivity} - \alpha \cdot \text{FPR}$$

where $\alpha$ encodes the relative cost of false alerts vs. missed events.

### 9.6 Epoch Visualization Panels

For qualitative validation, generate multi-panel figures each epoch showing:
1. Raw signal trace(s) with event markers.
2. Derived physiology/feature overlays (HR, HRV, spectral power, etc.).
3. Model inference map (past-state probability over time).
4. Countdown prediction vs. ground truth.
5. Confusion matrix for the monitored window.
6. Token/channel activation heatmap (if using attention-based models).

**Automated anomaly detection** on these panels flags: flat/constant signals, implausible feature ranges, missing modalities, low-variance inference maps, and token collapse (single dominant channel).

---

## 10. Inference and Deployment

### 10.1 Deployment Modes

| Mode | Description | Latency | Use Case |
|------|------------|---------|----------|
| **Streaming** | Process each new sample as it arrives, maintain hidden state | Real-time | Wearable, industrial IoT |
| **Batch retrospective** | Process entire recordings offline | Non-real-time | Research, audit, backfill |
| **Edge-cloud hybrid** | Lightweight feature extraction on-device, inference in cloud | Near-real-time | Smartwatch → phone → cloud |

### 10.2 Quantization

For edge deployment:
- **INT8 quantization** reduces model size ~4× with minimal accuracy loss.
- **Post-training quantization:** Calibrate on a representative dataset.
- **Quantization-aware training:** Simulate quantization during training for better accuracy.

### 10.3 State Management in Production

1. Initialize hidden state to zeros at the start of a monitoring session.
2. Propagate state forward with each inference step.
3. Detach state from computation graph (no gradient tracking during inference).
4. Reset state when the monitoring context changes (new session, device restart, user change).
5. Monitor state norm for drift detection — if $\|\mathbf{h}_t\|$ grows unbounded, apply state decay or reset.

### 10.4 Monitoring and Drift Detection

Deploy with:
- **Prediction distribution monitoring:** Track mean/variance of output probabilities over time.
- **Feature drift detection:** Compare live feature distributions to training distributions (Kolmogorov-Smirnov, PSI).
- **State health checks:** Monitor hidden state norms for anomalous growth.
- **Alert rate monitoring:** Track alerts per unit time — sudden changes may indicate model degradation.

---

## 11. Transferability Framework

### 11.1 The Core Invariant

Across all domains, the invariant is:

> **Dual coherence between confirmed current state and probabilistic future forecast under uncertainty-aware dynamics.**

Any application that requires:
1. Classifying the current phase relative to a critical event.
2. Forecasting when that event will occur.
3. Ensuring the forecast does not contradict the current-state assessment.
4. Providing calibrated uncertainty for downstream decision-making.

—can use this methodology directly by substituting domain-specific terminology.

### 11.2 Domain Adaptation Template

To adapt this methodology to a new domain, replace the following:

| Generic Term | Replace With (Examples) |
|-------------|------------------------|
| Critical event | Seizure onset, machine failure, market crash, cardiac arrest, network outage, pipeline rupture |
| Inter-event state | Inter-ictal, normal operation, bull market, sinus rhythm, steady state |
| Pre-event state | Pre-ictal, degraded operation, bearish divergence, arrhythmia precursor, pressure build-up |
| Onset state | Ictal onset, failure imminent, crash trigger, arrest onset, rupture point |
| Modality stream | ECG/EEG/PPG, vibration/temperature/power, price/volume/sentiment, latency/throughput/error-rate |
| Time-to-event $\tau$ | Minutes to seizure, hours to failure, days to drawdown, seconds to arrest |
| Forecast horizon $H$ | 10 min (seizure), 24 hr (industrial), 5 days (financial), 60 sec (cardiac) |

### 11.3 Domain-Specific Instantiation Examples

#### Example A: Industrial Predictive Maintenance

| Element | Instantiation |
|---------|--------------|
| Event | Machine component failure |
| Inter-event | Normal operation |
| Pre-event | Degraded performance (increased vibration, temperature drift) |
| Onset | Imminent failure (critical threshold breach) |
| Modalities | Vibration (accelerometer), thermal (thermocouple), power draw (current sensor), acoustic (microphone) |
| Horizon | 24–72 hours |
| Output | Maintenance alert with estimated time-to-failure and confidence |

#### Example B: Financial Risk Forecasting

| Element | Instantiation |
|---------|--------------|
| Event | Drawdown exceeding threshold / regime break |
| Inter-event | Stable/trending market |
| Pre-event | Divergence signals, volatility clustering |
| Onset | Critical risk level, stop-loss imminent |
| Modalities | Price series, volume, volatility indices, macro indicators, sentiment scores |
| Horizon | 1–5 trading days |
| Output | Risk alert with probability of drawdown and expected magnitude |

#### Example C: Network Anomaly Detection

| Element | Instantiation |
|---------|--------------|
| Event | Service outage, DDoS attack, cascading failure |
| Inter-event | Normal operation (baseline latency, throughput) |
| Pre-event | Anomalous traffic patterns, resource pressure |
| Onset | Critical threshold breach, failover imminent |
| Modalities | Latency, throughput, error rate, connection count, CPU/memory |
| Horizon | 5–60 minutes |
| Output | Incident alert with predicted time-to-outage and affected services |

#### Example D: Medical Early Warning (Cross-Condition)

| Element | Instantiation |
|---------|--------------|
| Event | Cardiac arrest, respiratory failure, sepsis onset, hypoglycemic event |
| Inter-event | Stable vitals |
| Pre-event | Subtle physiological deviations |
| Onset | Acute deterioration |
| Modalities | ECG, PPG, SpO₂, respiratory rate, blood pressure, temperature |
| Horizon | 5–60 minutes |
| Output | Clinical early warning score with trajectory and uncertainty |

### 11.4 Adaptation Checklist

When transferring to a new domain, complete these steps:

1. **[ ] Define the critical event** — precise onset criteria and annotation protocol.
2. **[ ] Identify modalities** — which sensor streams or data sources are available?
3. **[ ] Establish temporal labels** — derive $\tau$ values from event timestamps.
4. **[ ] Choose forecast horizon $H$ and onset threshold $\theta$** — domain-appropriate values.
5. **[ ] Select model from zoo** — based on deployment constraints (§7.3).
6. **[ ] Configure loss weights** — based on domain priorities (sensitivity vs. specificity).
7. **[ ] Set up chronological data loading** — sequence-sequential, no random shuffling.
8. **[ ] Define decision policy** — alert thresholds, persistence, cooldown (§9.5).
9. **[ ] Establish evaluation metrics** — domain-specific utility function for threshold optimization.
10. **[ ] Plan for drift monitoring** — feature distribution tracking, state health checks (§10.4).

---

## 12. Ablation and Interpretability

### 12.1 Standard Ablation Suite

To validate the contribution of each methodology component, run controlled ablation experiments:

| Ablation | What is Removed | Expected Impact |
|----------|----------------|-----------------|
| No coherence loss | $\mathcal{L}_{\text{coh}} = 0$ | Forecast and past-state heads diverge; implausible risk spikes during inter-event |
| No distance-to-event loss | $\mathcal{L}_{\text{dist}} = 0$ | Blurred proximity signal; late-event predictions degrade |
| No stateful memory | Hidden state reset each batch | Loss of long-range context; degraded on extended sequences |
| No skip connections | Remove UNet-like pass-through | Fine-grained signal detail lost; reduced sensitivity to rapid changes |
| No uncertainty head | Remove Bayesian head + $\mathcal{L}_{\text{unc}}$ | Overconfident predictions; no trustworthiness signal for decision policy |
| No dynamics residual | Remove $\mathcal{L}_{\text{dyn}}$ | Unconstrained state transitions; potential instability in streaming |
| Non-coherent heads | Remove class logit connection to forecast head | Weaker coupling between certainty and anticipation |
| Random ordering | Replace chronological with shuffled data loading | Stateful context broken; impossible to learn temporal dependencies |

### 12.2 Interpretability Tools

| Tool | What It Reveals |
|------|----------------|
| **Attention weight visualization** | Which time steps the model attends to for each prediction |
| **Modality gate analysis** | How the model weights each modality over time and context |
| **Token activation heatmaps** | Which latent channels are most active; detect token collapse |
| **Gradient-based saliency** | Which input features most influence predictions (Integrated Gradients, SmoothGrad) |
| **Counterfactual analysis** | What would change the prediction? Perturb inputs and observe output shift |
| **State trajectory plots** | PCA/t-SNE of hidden states over time; reveals regime structure |

### 12.3 Token Collapse Detection

Monitor for **token collapse** — the phenomenon where one or few latent channels dominate and the token diversity collapses:

- **Dominant channel ratio:** If a single channel accounts for >95% of token activation mass, flag.
- **Normalized entropy:** If entropy over channels drops below 0.20, flag.
- **Transition density:** Low density of dominant-channel switches indicates frozen token representation.

---

## 13. Implementation Checklist

### 13.1 Minimum Viable Implementation

For a functional proof-of-concept:

- [ ] **Data pipeline:** Load time-series data, align with event timestamps, generate $\tau$ labels.
- [ ] **Model:** Single-modality BiLSTM with dual classification + regression heads.
- [ ] **Loss:** $\mathcal{L}_{\text{cls}}$ (focal BCE) + $\mathcal{L}_{\text{fut}}$ (Smooth L1 on active samples).
- [ ] **Training:** AdamW, cosine annealing, early stopping on validation loss.
- [ ] **Evaluation:** Sensitivity, specificity, balanced accuracy, MAE on time-to-event.
- [ ] **Visualization:** Loss curves, confusion matrix, prediction vs. ground truth overlay.

### 13.2 Production-Grade Implementation

Add these for a production system:

- [ ] **Multi-modality support** with modality-specific encoders + attention gate fusion.
- [ ] **Multi-scale token cascade** with UNet skip connections.
- [ ] **Stateful data loading** with chronological ordering and hidden state management.
- [ ] **Full composite loss** including coherence, distance, dynamics, and uncertainty terms.
- [ ] **Causal prediction smoother** for deployment.
- [ ] **Decision policy** with alert persistence and cooldown logic.
- [ ] **Calibration pass** with threshold optimization.
- [ ] **Longitudinal streaming validation** (Phase D).
- [ ] **Ablation suite** to validate each component's contribution.
- [ ] **Drift monitoring** for production deployment.
- [ ] **Model quantization** for edge deployment.

---

## 14. Glossary of Domain-Agnostic Terms

| Term | Definition |
|------|-----------|
| **Critical event** | The event of interest that the system aims to anticipate (seizure, failure, crash, etc.) |
| **Inter-event state** | Period when no event is anticipated; baseline/normal regime |
| **Pre-event state** | Period when an event is anticipated within the forecast horizon; approaching phase |
| **Onset state** | Period immediately preceding the event; imminent phase |
| **Time-to-event ($\tau$)** | Time remaining until the next critical event, measured in domain-appropriate units |
| **Forecast horizon ($H$)** | Maximum lookahead for future-event prediction |
| **Onset threshold ($\theta$)** | Time boundary distinguishing pre-event from onset |
| **Modality** | A distinct sensor stream, data source, or feature type |
| **Token** | The fused latent representation at a single time step |
| **Scale** | A level in the temporal cascade with a specific receptive field size |
| **Stateful memory** | Hidden state propagated across time steps and batch boundaries |
| **Coherent heads** | Architecture where the forecast head receives the past-state head's output, ensuring consistency |
| **Certainty generator** | The past-state classification head — produces the confidence signal |
| **Anticipation trajectory** | The future forecast head — produces the time-to-event signal |
| **Causal smoother** | Post-hoc exponential moving average applied to raw predictions for deployment stability |
| **Token collapse** | Degenerate state where one latent channel dominates, reducing representational capacity |

---

## Appendix A: Reference Configurations

### A.1 Loss Configuration Template (YAML)

```yaml
loss:
  classification_weight: 0.35
  regression_weight: 0.25
  distance_weight: 0.10
  coherence_weight: 0.10
  dynamics_weight: 0.10
  uncertainty_weight: 0.10
  ranking_weight: 0.05

  classification_loss_type: focal    # "bce" or "focal"
  regression_loss: smoothl1           # "mse", "weighted_mse", or "smoothl1"
  focal_alpha: 0.25
  focal_gamma: 2.0
  label_smoothing: 0.05

  # Temporal focus
  classification_temporal_boost: 0.5
  regression_temporal_boost: 1.0
  temporal_focus_tau_min: 5.0         # in domain units
  classification_temporal_max_multiplier: 4.0
  regression_temporal_max_multiplier: 3.0

  # Onset auxiliary loss
  onset_aux_weight: 0.0               # >0 to enable
  onset_aux_window_min: 2.0           # onset window in domain units

  # Safe state mode (disables regression)
  loss_type: countdown                # "countdown" or "state"
  onset_threshold_min: 2.0
```

### A.2 Training Configuration Template (YAML)

```yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 1.0e-4
  weight_decay: 1.0e-5
  gradient_clip_norm: 2.0
  gradient_accumulation_steps: 1

  # Scheduler
  lr_scheduler: cosine
  warmup_epochs: 5
  min_lr: 1.0e-6

  # Early stopping
  early_stopping_patience: 15
  early_stopping_min_delta: 0.001

  # Mixed precision
  use_amp: true

  # Distributed
  distributed: false
  world_size: 1

  # Stateful
  stateful: false
  keep_hidden_across_epochs: true

  # Checkpointing
  save_best_only: true
  save_frequency_epochs: 5
```

### A.3 Data Configuration Template (YAML)

```yaml
data:
  dataset_root: /path/to/dataset
  sequence_length_seconds: 60
  feature_step_seconds: 1.0
  forecast_horizon: 10               # max time-to-event (domain units)
  output_countdown_max: 10.0
  onset_window_minutes: 2.0

  # Modality dimensions (adjust per application)
  modality_1_feature_dim: 12
  modality_2_feature_dim: 8
  modality_3_feature_dim: 6

  # Augmentation
  online_augmentation: true
  aug_probability: 0.7
  aug_time_warp_rates: [0.9, 0.95, 1.05, 1.1]
  aug_amplitude_scales: [0.9, 0.95, 1.05, 1.1]
  aug_noise_levels: [0.01, 0.02, 0.05]
  aug_time_shifts: [1, 2, 3, -1, -2, -3]

  # Data ordering
  chronological: true
  patient_sequential: true

  # Caps (set to 0 to disable)
  max_recordings: 0
  max_samples_per_recording: 0
```

### A.4 Model Configuration Template (YAML)

```yaml
model:
  architecture: bilstm_attention    # Options: bilstm_attention, cnn_lstm, tcn, 
                                     #          eegnet, mobilenet_1d, inception_time,
                                     #          temporal_transformer, multimodal_transformer,
                                     #          multimodal_early_fusion

  # Shared
  dropout: 0.25
  use_batch_norm: true
  coherent_heads: true
  output_countdown_max: 10.0

  # LSTM-specific
  hidden_dim: 128
  num_lstm_layers: 2
  use_attention: true
  num_attention_heads: 4

  # TCN-specific
  tcn_num_channels: [64, 64, 128, 128]
  tcn_kernel_size: 3

  # Transformer-specific
  transformer_d_model: 64
  transformer_nhead: 4
  transformer_num_layers: 4
  transformer_dim_feedforward: 256
```

---

## Appendix B: References and Further Reading

- **Focal Loss:** Lin et al. (2017), "Focal Loss for Dense Object Detection" — the foundation for class-imbalanced classification with a focusing parameter.
- **Temporal Convolutional Networks:** Bai et al. (2018), "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" — dilated causal convolutions as an RNN alternative.
- **InceptionTime:** Ismail Fawaz et al. (2020), "InceptionTime: Finding AlexNet for Time Series Classification" — multi-scale convolutional kernels for temporal data.
- **EEGNet:** Lawhern et al. (2018), "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces" — depthwise-separable convolutions for efficient biopotential processing.
- **Neural State-Space Models:** Gu et al. (2022), "Efficiently Modeling Long Sequences with Structured State Spaces" (S4) — foundation for long-range stateful sequence modeling.
- **Uncertainty Calibration:** Guo et al. (2017), "On Calibration of Modern Neural Networks" — ECE metric and temperature scaling.
- **Attention Mechanisms:** Vaswani et al. (2017), "Attention Is All You Need" — transformer architecture foundation.
- **UNet Skip Connections:** Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation" — inspiration for multi-scale skip connections.

---

*This document is the canonical, signal-agnostic, domain-transferable reference for the multimodal stateful cascading multi-temporal token context architecture and methodology. For application-specific instantiations, create derived documents that reference this core methodology and specify domain-specific parameter choices.*
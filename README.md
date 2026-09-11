Absolutely. The current README is too bare for a take-home assignment. Your README should make the reviewer understand **what you built, why you built it, how it works, how you evaluated it, what the numbers mean, and how to reproduce it** without digging through the code.

Below is a much stronger, attractive, recruiter/reviewer-friendly README. I’ve kept it consistent with the actual project decisions and results we established, including the important limitations and the fact that the LLM judge did **not** use a paid API.

**Replace your entire `README.md` with this single block:**

````markdown
# 🛡️ TrustSupport
## Evidence-Grounded Customer Support Agent

> **A trust-aware AI support pipeline that classifies customer intent, retrieves similar historical support cases, drafts evidence-grounded replies, and decides whether a case should be auto-handled or escalated to a human.**

Built for the **Hiver SDE Intern Take-Home Assignment**.

---

## 📌 Quick Overview

Traditional customer-support automation often follows a simple rule:

> **High confidence → automate**

TrustSupport uses a more cautious approach:

> **Intent confidence + historical evidence + input quality + risk → automation decision**

The system is designed around the idea that **a correct intent prediction does not automatically make a case safe to automate**.

For every incoming customer message, TrustSupport answers three questions:

1. 🎯 **What is the customer's primary support intent?**
2. 🔎 **What resolution approach is supported by similar historical interactions?**
3. 🧑‍💼 **Should the case be auto-handled or escalated to a human?**

### Headline Result

| Metric | TrustSupport |
|---|---:|
| Intent Accuracy | **65.5%** |
| Macro-F1 | **57.8%** |
| Weighted-F1 | **67.1%** |
| Escalation Accuracy | **~53%** |
| Golden Evaluation Set | **200 examples** |
| Leakage-Safe Reply Evaluation | **200 examples** |
| LLM-Judged Reply Sample | **50 examples** |

The headline number should **not** be interpreted as end-to-end customer-support success. See [What Is Misleading About the Headline Number?](#-what-is-misleading-about-the-headline-number).

---

# 🧭 Table of Contents

- [Problem](#-problem)
- [Solution](#-solution)
- [Key Idea](#-key-idea-evidence-confidence-gating)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Why VirginTrains?](#-why-virgintrains)
- [Intent Taxonomy](#-intent-taxonomy)
- [Pipeline](#-pipeline)
- [Components](#-components)
- [Resolution Evidence](#-resolution-evidence)
- [Auto-Handle vs Escalate](#-auto-handle-vs-escalate)
- [Example](#-example)
- [Evaluation Methodology](#-evaluation-methodology)
- [Baselines](#-baselines)
- [Results](#-results)
- [Reply Quality Evaluation](#-reply-quality-evaluation)
- [LLM Judge](#-llm-judge)
- [Data Leakage Protection](#-data-leakage-protection)
- [Failure Analysis](#-top-5-failure-modes)
- [What Is Misleading About the Headline Number?](#-what-is-misleading-about-the-headline-number)
- [Golden Evaluation Set](#-golden-evaluation-set)
- [Reproducibility](#-reproducibility)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Important Design Decisions](#-important-design-decisions)
- [Limitations](#-limitations)
- [One-Week Improvement Plan](#-one-week-improvement-plan)
- [Assignment Deliverables](#-assignment-deliverables)
- [License and Dataset Attribution](#-license-and-dataset-attribution)
- [Author](#-author)

---

# 🎯 Problem

Customer-support automation has three connected problems:

### 1. Intent Classification

The system needs to understand what the customer is asking.

Examples:

- "Where is my train?" → `train_status`
- "Can I change my ticket?" → `ticket_change`
- "Can I get compensation for the delay?" → `refund_compensation`
- "Is there Wi-Fi on the train?" → `wifi_connectivity`

### 2. Resolution Recommendation

Knowing the intent is not enough.

For example:

> "My train is delayed by two hours. Can I claim compensation?"

A useful system should look at **historical support interactions** and determine how similar cases were previously handled.

### 3. Automation Decision

Even if the system understands the message, it should not automatically respond to every case.

Examples of cases that may require escalation:

- insufficient information
- safety-related issues
- active disruptions
- financial disputes
- highly case-specific situations
- weak historical evidence
- low intent confidence

TrustSupport therefore treats **automation as a decision problem**, not simply a classification problem.

---

# 💡 Solution

TrustSupport combines five main capabilities:

- 🎯 Intent classification
- 🔎 Historical case retrieval
- 📚 Resolution evidence extraction
- ✍️ Evidence-grounded reply generation
- 🛡️ Risk-aware automation gating

The overall workflow is:

```text
                 Customer Message
                        │
                        ▼
              ┌──────────────────┐
              │  Input Quality   │
              │     Checker      │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Intent Detection │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Historical Case  │
              │    Retrieval     │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Resolution       │
              │ Evidence         │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Reply Generation │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Risk + Confidence│
              │     Analysis     │
              └────────┬─────────┘
                       │
              ┌────────┴─────────┐
              ▼                  ▼
        AUTO-HANDLE           ESCALATE
```

---

# 🛡️ Key Idea: Evidence-Confidence Gating

The central design decision in TrustSupport is the **Evidence-Confidence Gate**.

Instead of asking:

> "Is the classifier confident?"

the system asks:

> "Is there enough evidence and confidence to safely automate this case?"

The decision considers:

```text
Intent Confidence
        +
Historical Retrieval Evidence
        +
Resolution Agreement
        +
Input Quality
        +
Risk Signals
        ↓
Automation Decision
```

This creates a distinction between:

### 🟢 AUTO-HANDLE

The case has:

- sufficient information
- reasonable intent confidence
- relevant historical evidence
- acceptable resolution agreement
- no strong risk signal

### 🔴 ESCALATE

The case may have:

- insufficient information
- safety concerns
- active disruption
- financial dispute
- case-specific complexity
- low classifier confidence
- weak historical evidence

This approach intentionally prefers **safe escalation over unsupported automation**.

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │   Customer Message  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Input Quality     │
                         │       Analysis      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Intent Classifier │
                         │   TF-IDF + Logistic │
                         │      Regression     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Historical Retriever│
                         │    TF-IDF Search    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Resolution Evidence │
                         │      Extraction     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Reply Generator   │
                         │ Evidence-grounded  │
                         │       Draft        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evidence-Confidence│
                         │        Gate        │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                   AUTO-HANDLE             ESCALATE
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         Final Support Output
```

---

# 📊 Dataset

The project uses the **Customer Support on Twitter** dataset by Thought Vector.

Dataset:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

The dataset contains Twitter conversations between customers and customer-support accounts.

### Dataset Statistics

| Property | Value |
|---|---:|
| Total tweets | 2,811,774 |
| Customer / inbound tweets | 1,537,843 |
| Brand / outbound tweets | 1,273,931 |
| Unique authors | 702,777 |
| Raw dataset size | ~493 MB |

The original dataset is **not committed to this repository** because of its size and licensing restrictions.

---

# 🚆 Why VirginTrains?

Several support brands were explored before selecting the target brand.

VirginTrains was selected because:

- it contains a substantial number of support interactions
- it has a high proportion of linked historical responses
- the conversations provide useful evidence for several support intents
- the data contains enough variation for intent classification and retrieval experiments

The resulting processed dataset contains:

> **27,416 linked VirginTrains customer-support interactions**

Each processed interaction contains:

```text
customer_tweet_id
brand_tweet_id
created_at
customer_message
brand_response
```

---

# 🏷️ Intent Taxonomy

The final taxonomy contains **11 primary intents**.

| Intent | Meaning |
|---|---|
| `train_status` | Train delays, arrival/departure information, running status |
| `ticket_change` | Changing, modifying or altering an existing ticket |
| `refund_compensation` | Refunds, compensation and delay repayment |
| `booking_ticket` | Booking or purchasing tickets |
| `fare_price` | Ticket prices, fares and pricing questions |
| `seat_reservation` | Seat reservations, assigned seats and seating |
| `wifi_connectivity` | Wi-Fi and onboard connectivity |
| `station_facilities` | Station facilities and station-related information |
| `lost_property` | Lost property and lost items |
| `complaint_feedback` | Complaints, feedback and service experience |
| `other` | Messages that do not confidently fit another intent |

### Taxonomy Rules

The system follows several rules:

1. Assign **one primary intent**.
2. For multi-intent messages, select the main request or trigger.
3. Let the escalation mechanism capture additional complexity.
4. Do not force an uncertain message into an unrelated intent.
5. Use `other` when the message genuinely does not fit.
6. Very short or context-dependent messages may be escalated even when a classifier can technically produce a label.

---

# 🔄 Pipeline

## Step 1: Input Quality

The input-quality component checks whether a message contains enough information for reliable processing.

It detects patterns such as:

- URL-only messages
- number-only messages
- reference-only messages
- extremely short messages
- conversational fragments
- journey/time fragments
- non-support questions
- incomplete requests

Example:

```text
"Manchester"
```

Result:

```text
Input Quality: Insufficient
Decision: ESCALATE
Reason: very_short_message
```

Another example:

```text
"How are you?"
```

Result:

```text
Input Quality: Non-support question
Decision: ESCALATE
```

---

# 🎯 Step 2: Intent Detection

The final classifier uses:

```text
TF-IDF Vectorization
        +
Logistic Regression
```

The training data was created from historical VirginTrains interactions using conservative intent-labeling rules.

The final V2 training dataset contains:

> **12,013 examples**

Golden evaluation examples were excluded from the training data.

### Training Distribution

| Intent | Examples |
|---|---:|
| train_status | 5,309 |
| refund_compensation | 1,546 |
| seat_reservation | 1,140 |
| complaint_feedback | 1,035 |
| wifi_connectivity | 802 |
| fare_price | 772 |
| station_facilities | 621 |
| ticket_change | 376 |
| booking_ticket | 231 |
| lost_property | 181 |

---

# 🔎 Step 3: Historical Retrieval

TrustSupport retrieves similar historical customer messages using:

```text
TF-IDF
+
Cosine Similarity
+
Unigram + Bigram Features
```

For each customer message, the system retrieves the top historical cases.

Example query:

```text
My train is delayed by two hours, can I claim compensation?
```

Possible historical matches include:

```text
"My train is delayed by 2 hours..."
"My train was delayed over 2 hours..."
"Can I claim compensation for my delayed train?"
```

The system does not treat similarity alone as proof.

A high lexical similarity can still point to an incorrect resolution.

Therefore retrieval similarity is combined with **resolution evidence** and other signals.

---

# 📚 Step 4: Resolution Evidence

The historical responses are scanned for resolution actions.

Examples of extracted action categories include:

```text
delay_repay
refund
contact_aftersales
contact_customer_resolutions
direct_message
station_staff
website_form
ticket_validity
wifi_support
lost_property_team
```

The system counts how frequently retrieved historical responses support each action.

For example:

```text
Top retrieved cases:

delay_repay      → 2/5
website_form     → 2/5
other actions    → 1/5
```

This gives the system a measure of **resolution agreement**.

Important:

> Resolution evidence is treated as evidence of historical behavior, not as a guarantee of current policy correctness.

---

# ✍️ Step 5: Reply Generation

The reply generator creates a support draft using:

- predicted intent
- historical cases
- extracted resolution evidence
- confidence
- escalation state

The goal is to avoid inventing unsupported policies.

Instead of generating:

```text
"You are definitely eligible for a refund."
```

the system prefers evidence-aware wording such as:

```text
"Similar historical VirginTrains cases were directed
toward the Delay Repay process. Your eligibility would
need to be confirmed based on the journey details."
```

This is deliberately more conservative.

---

# 🧠 Step 6: Auto-Handle vs Escalate

The decision layer evaluates:

```text
Intent confidence
Retrieval similarity
Resolution agreement
Input quality
Risk signals
```

### Risk Categories

The current system checks for:

- `safety`
- `active_disruption`
- `financial_dispute`
- `case_specific`
- `insufficient_information`

### Escalation Examples

```text
Customer:
"Manchester"

→ ESCALATE
→ Reason: very_short_message
```

```text
Customer:
"My train is currently stuck and passengers are being
asked to leave the train. What should I do?"

→ Potential ESCALATE
→ Reason: active disruption / case-specific situation
```

The system is intentionally designed so that **automation is earned through evidence**.

---

# 🧪 Evaluation Methodology

The evaluation uses a manually reviewed golden dataset.

### Golden Set

```text
200 examples
```

Each example contains:

```text
id
customer_message
historical_response
intent
expected_resolution
should_escalate
escalation_reason
```

The examples were created using:

```text
AI / rule-based pre-label suggestions
                ↓
Human final review
                ↓
Final golden label
```

Every final label was human-reviewed.

### Golden Set Distribution

| Decision | Count |
|---|---:|
| AUTO / No escalation | 108 |
| ESCALATE | 92 |
| Total | 200 |

---

# 🥊 Baselines

Two baselines were implemented.

## Baseline 1: Majority Class

The system always predicts the most frequent intent.

Most frequent class:

```text
train_status
```

Results:

| Metric | Majority |
|---|---:|
| Accuracy | 10.5% |
| Macro-F1 | 1.9% |

This establishes the minimum useful reference point.

---

## Baseline 2: TF-IDF + Logistic Regression V1

The first training-data version used conservative customer-message keyword rules.

Results:

| Metric | V1 |
|---|---:|
| Accuracy | 40.5% |
| Macro-F1 | 35.2% |
| Weighted-F1 | 34.5% |

---

## Improved V2 Training Data

A second training-data construction strategy used both:

- customer-message phrases
- historical-response evidence

This improved the training set from:

```text
9,564 examples
```

to:

```text
12,013 examples
```

### V1 → V2

| Metric | V1 | V2 |
|---|---:|---:|
| Accuracy | 40.5% | **53.5%** |
| Macro-F1 | 35.2% | **48.7%** |
| Weighted-F1 | 34.5% | **49.3%** |

Macro-F1 improved by:

> **+13.5 percentage points**

This experiment showed that the way historical interactions are converted into training labels has a significant effect on classifier performance.

---

# 📈 Final TrustSupport Results

The final TrustSupport evaluation added input-quality handling and the evidence-confidence decision layer.

### Intent Classification

| Metric | Score |
|---|---:|
| Accuracy | **65.5%** |
| Macro-F1 | **57.8%** |
| Weighted-F1 | **67.1%** |

### Per-Intent Performance

| Intent | Precision | Recall | F1 |
|---|---:|---:|---:|
| booking_ticket | 0.64 | 0.58 | 0.61 |
| complaint_feedback | 0.55 | 0.52 | 0.54 |
| fare_price | 0.62 | 0.62 | 0.62 |
| lost_property | 0.00 | 0.00 | 0.00 |
| other | 0.79 | 0.67 | 0.73 |
| refund_compensation | 0.81 | 0.74 | 0.77 |
| seat_reservation | 0.77 | 0.74 | 0.76 |
| station_facilities | 0.23 | 1.00 | 0.38 |
| ticket_change | 0.65 | 0.52 | 0.58 |
| train_status | 0.48 | 0.57 | 0.52 |
| wifi_connectivity | 0.88 | 0.83 | 0.86 |

### Important Observation

`lost_property` has zero examples in the current golden set.

Therefore:

> The 0.00 F1 score does **not** prove that the classifier cannot handle lost-property cases. The evaluation set does not contain a positive example for that intent.

This is an important evaluation limitation.

---

# 🚦 Escalation Results

Final escalation evaluation:

| Metric | Result |
|---|---:|
| Escalation Accuracy | ~52.5% |
| Escalation Precision | ~49.0% |
| Escalation Recall | ~79.4% |
| Escalation F1 | ~60.6% |
| Auto-handle Precision | ~62.8% |

The system intentionally favors escalation when evidence is weak.

This produces a trade-off:

```text
More escalation
      ↓
Fewer unsupported automated responses
      ↓
But lower automation coverage
```

Therefore escalation quality needs further calibration.

---

# 🧾 Leakage-Safe Reply Evaluation

Reply evaluation was run with explicit leakage protection.

For each golden interaction:

- the matching customer tweet was excluded
- the matching brand response was excluded
- the excluded tweet IDs were passed to retrieval
- retrieval results were checked for leakage

### Results

```text
Golden examples evaluated:          200
Golden interaction pairs matched:  195
Tweet IDs excluded:                 390
Retrieval leakage detected:        0
```

Five golden examples did not have an exact matching processed interaction and therefore could not be mapped back to original tweet IDs.

This limitation is reported rather than silently ignored.

---

# ✍️ Automated Reply Quality

Leakage-safe reply evaluation produced:

| Metric | Result |
|---|---:|
| Intent Accuracy | 65.5% |
| Escalation Accuracy | 53.0% |
| Average Lexical Grounding | 0.190 |
| Average Retrieval Similarity | 0.232 |
| Average Evidence Strength | 0.247 |
| Average Evidence Agreement | 0.131 |
| Average Reply Length Score | 3.02 / 5 |
| Replies with Historical Evidence | 161 / 200 |
| Intent-Relevant Replies | 136 / 200 |
| Unsupported Claim Patterns | 0 |
| Empty Replies | 0 |

The low average evidence agreement is important.

It indicates that retrieved historical cases often do not strongly agree on a single resolution.

This is one reason why retrieval similarity should not be used as the sole automation criterion.

---

# 🤖 LLM Judge

A 50-example subset was created for LLM-assisted reply evaluation.

The sample contained:

```text
25 AUTO-HANDLE
25 ESCALATE
```

The rubric evaluated:

1. Relevance
2. Resolution correctness
3. Grounding
4. Helpfulness
5. Safety
6. Tone
7. Overall quality

### LLM Judge Results

| Dimension | Score |
|---|---:|
| Relevance | 3.32 / 5 |
| Resolution Correctness | 2.94 / 5 |
| Grounding | 2.88 / 5 |
| Helpfulness | 2.72 / 5 |
| Safety | 5.00 / 5 |
| Tone | 5.00 / 5 |
| Overall | **3.64 / 5** |

### Important Evaluation Note

The LLM judge was performed as an **LLM-assisted evaluation exercise**, not through a paid production API pipeline.

No paid API dependency is required to run the core TrustSupport system.

### Human Agreement

An independently collected human score set was **not completed in the final run**.

Therefore:

> No Cohen's kappa, correlation, or other judge-human agreement coefficient is reported.

This is a known evaluation gap and is included deliberately rather than fabricating an agreement number.

---

# 🔐 Data Leakage Protection

Data leakage is particularly dangerous in this project because the retrieval database contains historical conversations.

If the exact golden interaction appears in retrieval, the system could retrieve the answer it is being evaluated against.

To prevent this:

```text
Golden Customer Tweet
        +
Golden Brand Response
        ↓
Exclude both IDs
        ↓
Historical Retriever
        ↓
Top-K Search
```

The final leakage-safe evaluation reported:

```text
Retrieval leakage detected: 0
```

This makes the reply evaluation more meaningful.

---

# ⚠️ Top 5 Failure Modes

## 1. Mixed-Intent Messages

Some customer messages contain several requests.

Example:

```text
A delay problem combined with a refund or ticket-change request.
```

The single-intent taxonomy forces the classifier to choose one primary intent.

### Hypothesis

A multi-label or hierarchical intent model would represent these cases better.

---

## 2. Heterogeneous `other` Class

The `other` class contains many different types of messages.

Example:

```text
Positive feedback about a staff member
```

can be confused with:

```text
complaint_feedback
```

### Hypothesis

`other` should eventually be divided into meaningful subcategories or handled through an abstention mechanism.

---

## 3. Short / Context-Poor Messages

Example:

```text
"9:25 Litchfield Trent Valley to London Euston"
```

The message may contain journey information but not an explicit question.

The classifier can interpret this as:

```text
train_status
```

while the human label may be:

```text
other
```

### Hypothesis

A dedicated context-sufficiency model could distinguish between:

```text
valid support request
```

and

```text
conversation fragment requiring context
```

---

## 4. Ticket Semantics Overlap

`booking_ticket` and `ticket_change` share vocabulary such as:

```text
ticket
booking
journey
change
```

This can produce confusion between:

```text
"I want to book a ticket"
```

and:

```text
"Can I change my existing ticket?"
```

### Hypothesis

The classifier should explicitly model the distinction between:

```text
new transaction
```

and:

```text
modification of existing transaction
```

---

## 5. Lexical Retrieval Does Not Guarantee Correct Resolution

TF-IDF retrieval can return textually similar cases that had different resolutions.

For example:

```text
A ticket acceptance question
```

may retrieve cases about:

```text
station facilities
```

because both contain overlapping words such as station, ticket, train, etc.

### Hypothesis

Semantic embeddings followed by a reranker should improve retrieval quality.

---

# 🚨 What Is Misleading About the Headline Number?

The headline:

> **65.5% intent accuracy**

is useful, but it is not the same thing as:

> **65.5% of customer issues are successfully resolved.**

Several factors make the number incomplete:

### 1. Accuracy is measured on only 200 golden examples

A 200-example evaluation set is useful for a take-home assignment, but it is not a production-scale benchmark.

### 2. Accuracy hides class imbalance

Macro-F1 is therefore also reported.

```text
Accuracy: 65.5%
Macro-F1: 57.8%
```

The gap shows that performance is not uniform across intents.

### 3. Lost-property performance is unmeasured

The golden set currently contains no positive `lost_property` examples.

### 4. Intent correctness is not reply correctness

A correct intent can still produce:

- weak retrieval
- incorrect resolution evidence
- an unhelpful reply
- an incorrect automation decision

### 5. Retrieval similarity is not evidence correctness

A similarity score tells us that messages look similar.

It does not prove that the historical resolution applies to the current customer.

### 6. Historical support behavior may be outdated

The dataset contains historical Twitter support conversations.

Past responses should therefore be treated as evidence of historical behavior rather than guaranteed current policy.

---

# 🧪 Golden Evaluation Set

The final golden dataset is:

```text
data/golden/golden_set_final.csv
```

### Size

```text
200 examples
```

### Labeling Process

```text
Historical interaction sampling
          ↓
Pre-label suggestions
          ↓
Human review
          ↓
Final intent + resolution + escalation labels
```

Each row includes:

```text
id
customer_message
historical_response
intent
expected_resolution
should_escalate
escalation_reason
```

### Evaluation Principles

The golden set was kept separate from training.

Exact normalized customer-message + historical-response pairs were excluded from training to reduce leakage.

---

# 🔁 Reproducibility

## Requirements

Recommended environment:

```text
Python 3.13
pip
Git
```

The project dependencies are pinned in:

```text
requirements.txt
```

---

## 1. Clone the Repository

```bash
git clone https://github.com/nischayskth-commits/Hiver-SDE-Assignment.git
cd Hiver-SDE-Assignment
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Setup

Download the Kaggle dataset:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Place the downloaded file here:

```text
data/raw/twcs.csv
```

The raw dataset is intentionally excluded from Git.

---

# ⚙️ Running the Pipeline

The main system is implemented in:

```text
src/trustsupport.py
```

The main class is:

```python
TrustSupport
```

A basic health check can be performed with:

```bash
python -c "from src.trustsupport import TrustSupport; print(TrustSupport().analyze('My train is delayed by two hours, can I claim compensation?'))"
```

Example type of output:

```text
Intent:
refund_compensation

Confidence:
high

Historical Evidence:
Delay Repay / compensation-related cases

Decision:
AUTO-HANDLE

Draft Reply:
Similar historical VirginTrains cases were directed
toward the Delay Repay process. Eligibility should be
confirmed based on the journey details.
```

---

# 🧪 Running Evaluation

## Majority Baseline

```bash
python evaluation/baseline_majority.py
```

## TF-IDF V1 Baseline

```bash
python evaluation/baseline_tfidf.py
```

## TF-IDF V2 Baseline

```bash
python evaluation/baseline_tfidf_v2.py
```

## TrustSupport Evaluation

```bash
python evaluation/evaluate_trustsupport.py
```

## Leakage-Safe Reply Evaluation

```bash
python evaluation/evaluate_replies.py
```

The evaluation scripts use the prepared processed and golden datasets included in the repository.

---

# 🧰 Main Components

| Component | File | Purpose |
|---|---|---|
| Main Agent | `src/trustsupport.py` | End-to-end pipeline |
| Input Quality | `src/intent/input_quality.py` | Detects incomplete/non-support messages |
| Intent Discovery | `src/intent/discover_intents.py` | Exploratory intent discovery |
| Intent Examples | `src/intent/sample_intent_examples.py` | Intent inspection |
| Retriever | `src/retrieval/historical_retriever.py` | Historical similarity search |
| Resolution Evidence | `src/retrieval/resolution_evidence.py` | Extracts historical resolution actions |
| Decision Gate | `src/decision/evidence_gate.py` | AUTO-HANDLE vs ESCALATE |
| Reply Generator | `src/generation/reply_generator.py` | Evidence-grounded reply drafting |
| Majority Baseline | `evaluation/baseline_majority.py` | Trivial baseline |
| TF-IDF Baseline | `evaluation/baseline_tfidf.py` | Simple baseline |
| Improved Baseline | `evaluation/baseline_tfidf_v2.py` | V2 training-data experiment |
| Main Evaluation | `evaluation/evaluate_trustsupport.py` | Intent + escalation evaluation |
| Reply Evaluation | `evaluation/evaluate_replies.py` | Leakage-safe reply evaluation |
| LLM Judge | `evaluation/evaluate_llm_judge.py` | LLM-assisted reply scoring |
| Judge Set | `evaluation/create_judge_set.py` | Creates judge sample |
| Failure Analysis | `evaluation/inspect_failures.py` | Error inspection |
| Multi-Intent Analysis | `evaluation/analyze_multi_intent.py` | Multi-intent investigation |

---

# 📁 Project Structure

```text
Hiver-SDE-Assignment/
│
├── README.md
├── requirements.txt
├── repair_golden.py
│
├── data/
│   ├── raw/
│   │   └── twcs.csv                  # Local only, not committed
│   │
│   ├── processed/
│   │   ├── virgintrains_support_pairs.csv
│   │   ├── training_pairs.csv        # V1 local artifact
│   │   └── training_pairs_v2.csv
│   │
│   └── golden/
│       └── golden_set_final.csv
│
├── src/
│   ├── trustsupport.py
│   │
│   ├── data/
│   │   ├── analyze_brands.py
│   │   ├── analyze_resolution_quality.py
│   │   ├── build_brand_dataset.py
│   │   ├── build_training_dataset.py
│   │   ├── build_training_dataset_v2.py
│   │   ├── inspect_dataset.py
│   │   └── sample_conversations.py
│   │
│   ├── decision/
│   │   └── evidence_gate.py
│   │
│   ├── generation/
│   │   └── reply_generator.py
│   │
│   ├── intent/
│   │   ├── discover_intents.py
│   │   ├── input_quality.py
│   │   └── sample_intent_examples.py
│   │
│   └── retrieval/
│       ├── historical_retriever.py
│       └── resolution_evidence.py
│
├── evaluation/
│   ├── baseline_majority.py
│   ├── baseline_tfidf.py
│   ├── baseline_tfidf_v2.py
│   ├── evaluate_trustsupport.py
│   ├── evaluate_replies.py
│   ├── evaluate_llm_judge.py
│   ├── create_judge_set.py
│   ├── inspect_failures.py
│   └── analyze_multi_intent.py
│
└── report/
    └── REPORT_FINAL.docx
```

> Note: `training_pairs.csv` is the earlier V1 training artifact and is not required for the final TrustSupport run.

---

# 🧰 Technology Stack

### Language

- Python 3.13

### Machine Learning

- scikit-learn
- TF-IDF
- Logistic Regression
- Cosine Similarity

### NLP / Retrieval

- Text normalization
- N-gram features
- Lexical retrieval
- Resolution-action extraction

### Evaluation

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- LLM-assisted rubric scoring
- Leakage-safe retrieval evaluation

### Development

- Git
- GitHub
- VS Code
- PowerShell

---

# 🧠 Important Design Decisions

The project contains several non-obvious decisions.

### 1. Outbound Brand Tweets Were Used as Support Responses

Brand support accounts primarily appear in outbound responses.

This distinction was important when selecting the target brand.

---

### 2. Conversation Links Were Used

The dataset's:

```text
in_response_to_tweet_id
```

and:

```text
response_tweet_id
```

relationships were used to construct customer-support pairs.

---

### 3. VirginTrains Was Selected After Brand Exploration

The target brand was not chosen arbitrarily.

Multiple support accounts were compared based on the availability and quality of linked responses.

---

### 4. Eleven Intents Were Chosen

The taxonomy was kept small enough for reliable labeling while covering the major VirginTrains support themes.

---

### 5. `other` Was Retained

Forcing every message into a specific intent would artificially inflate confidence.

`other` provides an explicit escape hatch.

---

### 6. One Primary Intent

The current classifier predicts one primary intent.

For multi-intent messages, the dominant request is selected and complexity can influence escalation.

---

### 7. Historical Responses Are Evidence, Not Truth

A historical response is not automatically treated as current policy.

This is especially important for:

- refunds
- compensation
- ticket rules
- operational disruptions

---

### 8. TF-IDF Was Chosen for the Baseline

TF-IDF was selected because it is:

- lightweight
- reproducible
- interpretable
- fast enough for a take-home evaluation
- easy to compare against more complex approaches later

---

### 9. Resolution Evidence Was Added Separately

Similarity alone was considered insufficient.

The system therefore extracts resolution actions from retrieved responses.

---

### 10. Input Quality Was Added as a Separate Layer

Very short messages can cause unreliable predictions.

Instead of forcing classification, the system can escalate these cases.

---

### 11. Golden Examples Are Excluded

Golden examples are excluded from training and retrieval wherever an exact interaction mapping exists.

This reduces evaluation leakage.

---

### 12. Paid LLM API Was Not Required

The core pipeline remains runnable without a paid LLM API.

The LLM judge was used as an evaluation aid rather than making the production pipeline dependent on a paid external model.

---

# ⚠️ Limitations

TrustSupport is a research-style prototype rather than a production customer-support system.

### Dataset Limitations

The dataset is historical Twitter support data.

Policies and procedures may have changed.

### Labeling Limitations

The training labels are derived using conservative heuristic rules rather than a fully human-labeled training corpus.

### Taxonomy Limitations

Some intents overlap semantically.

### Multi-Intent Limitation

Only one primary intent is currently predicted.

### Retrieval Limitation

TF-IDF captures lexical similarity but not deep semantic meaning.

### Resolution Limitation

Historical resolution extraction is heuristic and does not guarantee policy correctness.

### Escalation Limitation

The current gate is conservative and requires further calibration.

### Evaluation Limitation

The golden set contains only 200 examples.

### Judge-Human Agreement Limitation

An independent human scoring set was not completed, so no judge-human agreement coefficient is reported.

### Missing Positive Example

The golden set currently contains no positive `lost_property` examples.

---

# 🚀 One-Week Improvement Plan

If one additional week were available, the development plan would be:

## Day 1: Taxonomy Refinement

- Review confusing intents
- Add more examples
- Revisit `other`
- Add explicit multi-intent handling

## Day 2: Better Retrieval

Replace or augment TF-IDF with:

```text
Sentence Embeddings
        +
Top-K Retrieval
        +
Reranking
```

This should improve semantic matching.

## Day 3: Multi-Intent Detection

Move from:

```text
one message → one intent
```

toward:

```text
one message → primary intent + secondary intent
```

---

## Day 4: Resolution Extraction

Improve resolution extraction using:

- structured response parsing
- intent-specific resolution templates
- confidence scoring
- contradictory-evidence detection

---

## Day 5: Escalation Calibration

Tune the gate against a larger human-reviewed set.

Measure:

```text
Automation Coverage
Escalation Recall
False Automation Rate
False Escalation Rate
```

---

## Day 6: Human Evaluation

Expand the reply-quality evaluation.

Collect independent human ratings and calculate:

- agreement
- correlation
- rubric-level agreement

---

## Day 7: Final Evaluation

Run:

- ablation studies
- final leakage checks
- failure analysis
- reproducibility test
- README verification
- report cleanup

---

# 📦 Assignment Deliverables

| Requirement | Status |
|---|---|
| Runnable repository | ✅ |
| Single-brand support agent | ✅ |
| Intent classification | ✅ |
| Historical resolution grounding | ✅ |
| Auto-handle / escalation decision | ✅ |
| 150–250 golden examples | ✅ 200 |
| Automated evaluation metrics | ✅ |
| Reply-quality evaluation | ✅ |
| LLM judge rubric | ✅ |
| Judge-human agreement | ⚠️ Not completed |
| Two baselines | ✅ |
| Top 5 failure modes | ✅ |
| Misleading headline number | ✅ |
| One-week improvement plan | ✅ |
| Decision log | ✅ |
| Final report | ✅ |

---

# 🔬 Research / Engineering Takeaway

The main lesson from this project is:

> **Customer-support automation should not treat model confidence as sufficient evidence for action.**

A classifier can be confident while:

- the input is incomplete
- historical cases disagree
- the retrieved example is only lexically similar
- the request is financially sensitive
- the issue is actively unfolding
- the historical policy may no longer apply

TrustSupport therefore separates:

```text
Understanding
     ↓
Evidence
     ↓
Risk
     ↓
Action
```

This makes the system easier to reason about and provides a clear place for future improvements.

---

# 📌 Current Headline

```text
TrustSupport

65.5% Intent Accuracy
57.8% Macro-F1
200 Human-Reviewed Golden Examples
0 Retrieval Leakage Detected
3.64 / 5 LLM-Judged Overall Reply Quality
```

These numbers should be interpreted together with the limitations and failure analysis described above.

---

# 📜 Dataset Attribution & License

This project uses:

**Customer Support on Twitter**

by **Thought Vector**.

Dataset:

https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Dataset license:

**CC BY-NC-SA 4.0**

The dataset is not redistributed through this repository.

The project uses the historical interactions for research and evaluation purposes and does not claim that historical support responses represent current VirginTrains policies.

---

# 👤 Author

**Nischay Suman Achar**

Hiver SDE Intern Take-Home Assignment

GitHub:

https://github.com/nischayskth-commits/Hiver-SDE-Assignment

---

# ⭐ Final Note

TrustSupport is intentionally designed as an interpretable, evidence-aware prototype.

The objective is not to claim that a simple model has solved customer support.

The objective is to demonstrate a complete engineering pipeline:

```text
Dataset
   ↓
Data Processing
   ↓
Intent Taxonomy
   ↓
Training
   ↓
Retrieval
   ↓
Resolution Evidence
   ↓
Reply Generation
   ↓
Risk-Aware Decision
   ↓
Evaluation
   ↓
Failure Analysis
   ↓
Next Improvements
```

The next step toward production would be stronger semantic retrieval, better human-labeled training data, explicit multi-intent handling, calibrated escalation thresholds, and independent human evaluation.
````

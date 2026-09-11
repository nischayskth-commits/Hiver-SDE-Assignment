\# TrustSupport: Evidence-Grounded Customer Support Agent



TrustSupport is an AI-assisted customer support agent built for the Hiver SDE Intern take-home assignment.



The system is designed to answer three questions for an incoming customer message:



1\. What is the customer's primary support intent?

2\. What resolution approach is supported by similar historical support interactions?

3\. Should the case be auto-handled or escalated to a human?



The project focuses on \*\*trust-aware automation\*\* rather than treating every high-confidence classification as safe to automate.



\---



\## 1. Problem



Customer-support systems need to classify incoming messages, suggest useful responses, and decide when automation is appropriate.



The challenge is that a correct intent prediction does not necessarily mean that an automated response is safe.



TrustSupport therefore combines:



\- Intent classification

\- Historical support-case retrieval

\- Resolution evidence extraction

\- Input-quality checking

\- Evidence-confidence gating

\- Grounded reply generation



The core pipeline is:



Customer Message

→ Input Quality

→ Intent Detection

→ Similar Historical Cases

→ Resolution Evidence

→ Draft Reply

→ Risk / Confidence Analysis

→ AUTO-HANDLE or ESCALATE



\---



\## 2. Dataset



The project uses the Kaggle \*\*Customer Support on Twitter\*\* dataset by Thought Vector.



Dataset:



https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter



The dataset contains customer tweets and brand support responses.



For this project, \*\*VirginTrains\*\* was selected as the target support brand.



The original dataset is approximately 500 MB and is intentionally not committed to this repository.



Dataset license:



\*\*CC BY-NC-SA 4.0\*\*



The historical responses are treated as evidence of past support behavior and not as guaranteed current policy.



\---



\## 3. Project Structure



```text

Hiver-SDE-Assignment/

│

├── README.md

├── requirements.txt

├── repair\_golden.py

│

├── data/

│   ├── raw/

│   │   └── twcs.csv

│   │

│   ├── processed/

│   │   ├── virgintrains\_support\_pairs.csv

│   │   ├── training\_pairs.csv

│   │   └── training\_pairs\_v2.csv

│   │

│   └── golden/

│       └── golden\_set\_final.csv

│

├── src/

│   ├── trustsupport.py

│   │

│   ├── data/

│   │   ├── analyze\_brands.py

│   │   ├── analyze\_resolution\_quality.py

│   │   ├── build\_brand\_dataset.py

│   │   ├── build\_training\_dataset.py

│   │   ├── build\_training\_dataset\_v2.py

│   │   ├── inspect\_dataset.py

│   │   └── sample\_conversations.py

│   │

│   ├── decision/

│   │   └── evidence\_gate.py

│   │

│   ├── generation/

│   │   └── reply\_generator.py

│   │

│   ├── intent/

│   │   ├── discover\_intents.py

│   │   ├── input\_quality.py

│   │   └── sample\_intent\_examples.py

│   │

│   └── retrieval/

│       ├── historical\_retriever.py

│       └── resolution\_evidence.py

│

├── evaluation/

│   ├── baseline\_majority.py

│   ├── baseline\_tfidf.py

│   ├── baseline\_tfidf\_v2.py

│   ├── evaluate\_trustsupport.py

│   ├── evaluate\_replies.py

│   ├── evaluate\_llm\_judge.py

│   ├── create\_judge\_set.py

│   ├── inspect\_failures.py

│   ├── analyze\_multi\_intent.py

│   ├── trustsupport\_predictions.csv

│   ├── reply\_quality\_metrics.csv

│   └── llm\_judge\_set.csv

│

├── tests/

│

├── notebooks/

│

└── report/

&#x20;   └── REPORT\_FINAL.docx


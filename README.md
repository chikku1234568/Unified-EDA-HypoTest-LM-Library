# Hypotest

A Python library for deterministic hypothesis testing with automatic assumption checking and optional LLM-based interpretation.

Hypotest provides a clean statistical engine designed for data scientists, researchers, and engineers who need reliable and reproducible statistical testing workflows.

---

# Overview

Hypotest simplifies hypothesis testing by providing:

- Deterministic statistical engine  
- Automatic assumption validation (normality, variance homogeneity)  
- Structured result objects with statistical metadata  
- Optional LLM-based interpretation layer  
- Safe `Dataset` abstraction for robust data handling  

All statistical computations are deterministic and independent of LLM usage.

---

# Installation

Install from PyPI:

```bash
pip install LM-hypotest
```

Development install:

```bash
git clone https://github.com/chikku1234568/Unified-EDA-HypoTest-LM-Library
cd Unified-EDA-HypoTest-LM-Library
pip install -e .
```

Optional LLM support:

```bash
pip install LM-hypotest[llm]
```

---

# Quick Start

## Example: Independent t-test

```python
import pandas as pd
import numpy as np

import hypotest
from hypotest.core.dataset import Dataset
from hypotest.tests.parametric.ttest import TTest


# Create example dataset
df = pd.DataFrame({
    "group": ["A"] * 100 + ["B"] * 100,
    "value": np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(1, 1, 100),
    ])
})

# Wrap DataFrame in Dataset abstraction
dataset = Dataset(df)

# Run t-test
test = TTest()

result = test.execute(
    dataset=dataset,
    target="value",
    features=["group"],
)

print(result)
```

Example output:

```
TestResult(test='Independent t-test', feature='group', statistic=4.231, p=0.00003, significant)
```

---

# Automatic Assumption Checking

Hypotest automatically checks statistical assumptions:

```python
for assumption in result.assumptions:
    print(assumption.assumption_name, assumption.passed)
```

Example output:

```
normality True
homoscedasticity False
```

Each assumption provides:

- Statistical result  
- Interpretation  
- Recommendation  

---

# Optional LLM Interpretation

Hypotest can generate natural-language explanations using OpenAI-compatible providers such as DeepSeek.

Example:

```python
hypotest.configure(
    llm_api_key="your-api-key",
    llm_base_url="https://api.deepseek.com/v1",
    llm_model="deepseek-chat",
    enable_llm_interpretation=True,
)

print(result.explain())
```

Example output:

```
The independent t-test indicates a statistically significant difference between the two groups...
```

LLM interpretation is optional and does not affect statistical computation.

---

# Configuration

Configure hypotest globally:

```python
hypotest.configure(
    llm_api_key="your-key",
    llm_base_url="https://api.deepseek.com/v1",
    llm_model="deepseek-chat",
    enable_llm_interpretation=True,
)
```

View configuration:

```python
print(hypotest.info())
```

---

# Dataset Abstraction

Hypotest uses a Dataset wrapper to provide safe data handling:

```python
from hypotest.core.dataset import Dataset

dataset = Dataset(df)
```

This enables:

- Safe missing value handling  
- Validation before test execution  
- Future extensibility  

---

# Supported Tests (Current MVP)

- Independent t-test  

Planned:

- Welch's t-test  
- Mann-Whitney U test  
- ANOVA  
- Chi-square test  
- Correlation tests  

---

# Features

Core features implemented:

- Deterministic statistical engine  
- Automatic assumption checking  
- Structured `TestResult` objects  
- Dataset abstraction layer  
- Plug-in test registry system  
- Optional LLM interpretation  

Planned features:

- Automatic test recommendation  
- Effect size library  
- Automated reporting  
- Additional statistical tests  

---

# Example: Full Workflow

```python
import pandas as pd
import numpy as np
import hypotest

from hypotest.core.dataset import Dataset
from hypotest.tests.parametric.ttest import TTest


hypotest.configure(enable_llm_interpretation=False)

df = pd.DataFrame({
    "group": ["A"] * 50 + ["B"] * 50,
    "value": np.random.randn(100),
})

dataset = Dataset(df)

test = TTest()

result = test.execute(dataset, "value", ["group"])

print(result)

for a in result.assumptions:
    print(a.assumption_name, a.passed)

print(result.explain())  # None if LLM disabled
```

---

# Project Structure

```
hypotest/
├── core/
│   ├── dataset.py
│   ├── result.py
│
├── tests/
│   ├── parametric/
│   │   ├── ttest.py
│
├── assumptions/
│   ├── normality.py
│   ├── variance.py
│
├── llm/
│   ├── client.py
│   ├── interpreter.py
│
├── config/
│   ├── manager.py
│
├── info.py
```

---

# Requirements

Core requirements:

- Python ≥ 3.10  
- pandas ≥ 1.5  
- numpy ≥ 1.21  
- scipy ≥ 1.9  

Optional:

- OpenAI-compatible client (for LLM interpretation)

---

# Philosophy

Hypotest separates:

- Deterministic statistical computation  
- Probabilistic natural-language interpretation  

This ensures statistical correctness while enabling explainability.

---

# License

MIT License
````0

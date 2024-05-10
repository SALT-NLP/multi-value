# Multi-VALUE: The VernAcular Language Understanding Evaluation benchmark 


## Setup From PyPi
`pip install value-nlp`

```python
from multivalue import Dialects
southern_am = Dialects.SoutheastAmericanEnclaveDialect()
print(southern_am)
print(southern_am.transform("I talked with them yesterday"))
print(southern_am.executed_rules)
```

## Setup From Source
### Prerequisites: 
* [anaconda](https://www.anaconda.com/products/individual)

1. Create a virtual environment
```bash
conda create --name value python=3.7.13
conda activate value
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Install spaCy English pipeline and nltk wordnet
```python
bash downloads.sh
```

4. Confirm that your setup is correct by running the unittest
```bash
python -m unittest tests.py
```

### Build Multi-VALUE CoQA (optional)
1. Pull data
```bash
bash pull_coqa.sh
```

2. Run for each dialect
```bash
python -m src.build_coqa_value --dialect aave &
python -m src.build_coqa_value --dialect appalachian &
python -m src.build_coqa_value --dialect chicano &
python -m src.build_coqa_value --dialect indian &
python -m src.build_coqa_value --dialect multi &
python -m src.build_coqa_value --dialect singapore &


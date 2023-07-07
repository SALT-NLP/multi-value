# Multi-VALUE: The VernAcular Language Understanding Evaluation benchmark 

## Setup
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
python -m spacy download en_core_web_sm
python 
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('cmudict')
>>> quit()
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


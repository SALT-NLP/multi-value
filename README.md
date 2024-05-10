# Multi-VALUE: The VernAcular Language Understanding Evaluation benchmark 


## Setup From PyPi For Your Own Projects
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
```

## Citation
```
@inproceedings{ziems-etal-2023-multi,
    title = "Multi-{VALUE}: A Framework for Cross-Dialectal {E}nglish {NLP}",
    author = "Ziems*, Caleb  and
      Held*, William  and
      Yang, Jingfeng  and
      Dhamala, Jwala  and
      Gupta, Rahul  and
      Yang, Diyi",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.44",
    doi = "10.18653/v1/2023.acl-long.44",
    pages = "744--768",
    abstract = "Dialect differences caused by regional, social, and economic factors cause performance discrepancies for many groups of language technology users. Inclusive and equitable language technology must critically be dialect invariant, meaning that performance remains constant over dialectal shifts. Current systems often fall short of this ideal since they are designed and tested on a single dialect: Standard American English (SAE). We introduce a suite of resources for evaluating and achieving English dialect invariance. The resource is called Multi-VALUE, a controllable rule-based translation system spanning 50 English dialects and 189 unique linguistic features. Multi-VALUE maps SAE to synthetic forms of each dialect. First, we use this system to stress tests question answering, machine translation, and semantic parsing. Stress tests reveal significant performance disparities for leading models on non-standard dialects. Second, we use this system as a data augmentation technique to improve the dialect robustness of existing systems. Finally, we partner with native speakers of Chicano and Indian English to release new gold-standard variants of the popular CoQA task. To execute the transformation code, run model checkpoints, and download both synthetic and gold-standard dialectal benchmark datasets, see \url{http://value-nlp.org}.",
}
```

from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from src.Dialects import (
    AfricanAmericanVernacular,
    IndianDialect,
    ColloquialSingaporeDialect,
    ChicanoDialect,
    AppalachianDialect,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

TASK = "translation"
CKPT = "facebook/nllb-200-1.3B"
src_lang = "eng_Latn"
tgt_lang_dict = {"de": "deu_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans", "gu": "guj_Gujr"}
device = 0

import evaluate


def dialect_factory(dialect):
    def dialect_transform(examples):
        D = dialect(morphosyntax=True)
        examples["src"] = [
            D.convert_sae_to_dialect(src_text) for src_text in examples["src"]
        ]
        return examples

    return dialect_transform


def flatten_factory(target):
    def flatten(example):
        example["src"] = example["translation"]["en"]
        example["tgt"] = example["translation"][target]
        del example["translation"]
        return example

    return flatten


def translate_factory(pipe):
    def translate(examples):
        examples["tgt_pred"] = [
            out["translation_text"] for out in pipe(examples["src"], batch_size=16)
        ]
        return examples

    return translate


sacrebleu = evaluate.load("sacrebleu")
model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to("cuda:" + str(device))
tokenizer = AutoTokenizer.from_pretrained(CKPT)
for lang in ["de", "gu", "zh", "ru"]:
    dataset = load_dataset(f"WillHeld/wmt19-valid-only-{lang}_en")["validation"]
    pipe = pipeline(
        TASK,
        model=model,
        tokenizer=tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang_dict[lang],
        max_length=400,
        device=device,
    )
    for dialect in [
        None,
        AfricanAmericanVernacular,
        IndianDialect,
        ColloquialSingaporeDialect,
        ChicanoDialect,
        AppalachianDialect,
    ]:
        d_dataset = dataset.map(flatten_factory(lang))
        if dialect:
            dialect_name = dialect(morphosyntax=True).dialect_name
            dialect_transform = dialect_factory(dialect)
            d_dataset = d_dataset.map(dialect_transform, num_proc=24, batched=True)
        else:
            dialect_name = "Standard American"
        d_dataset = d_dataset.map(translate_factory(pipe), batched=True)
        results = sacrebleu.compute(
            predictions=d_dataset["tgt_pred"], references=d_dataset["tgt"]
        )
        print(f"{dialect_name} en -> {lang}")
        print(results)

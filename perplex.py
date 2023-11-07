import datasets, transformers
from transformers import BertTokenizer, BertForMaskedLM
from bert_perplexity.perplexity import PerplexityPipeline
import os
import platform
from tqdm import tqdm

op_sys = platform.system()
if op_sys == "Windows":
    PRETRAINED_MODELS_DIR = "F:/pretrained_models"
    DATASETS_DIR = "F:/datasets"
elif op_sys == "Linux":
    PRETRAINED_MODELS_DIR = "/users/uestc1/zys/pretrained_models"
    DATASETS_DIR = "/users/uestc1/zys/Datasets"


# download if not exists cache_dir
def load_model(hf_name=None, cache_dir=None, download=False):
    if download:
        assert not cache_dir
        assert hf_name
    else:
        assert not hf_name
        assert cache_dir
    if download:
        model = download_model_files()
    else:
        model = load_from_cache()
    return model


# download_model_files
def download_model_files(pretrained_models_dir=PRETRAINED_MODELS_DIR, hf_name="bert-base-uncased",
                         ):
    cache_dir = os.path.join(pretrained_models_dir)
    en_tokenizer = BertTokenizer.from_pretrained(
        hf_name,
        cache_dir=cache_dir
    )
    en_model = BertForMaskedLM.from_pretrained(
        hf_name,
        cache_dir=cache_dir
    )
    en_pipeline = PerplexityPipeline(model=en_model, tokenizer=en_tokenizer)
    return en_pipeline


def load_from_cache(pretrained_models_dir=PRETRAINED_MODELS_DIR,
                    # bert-base-uncased
                    # model_name="models--bert-base-uncased/snapshots/1dbc166cf8765166998eff31ade2eb64c8a40076"
                    # bert-base-multilingual-cased
                    model_name="models--bert-base-multilingual-cased/snapshots/fdfce55e83dbed325647a63e7e1f5de19f0382ba"
                    ):
    en_tokenizer = BertTokenizer.from_pretrained(
        os.path.join(pretrained_models_dir, model_name)
    )
    en_model = BertForMaskedLM.from_pretrained(
        os.path.join(pretrained_models_dir, model_name)).cuda()
    en_pipeline = PerplexityPipeline(model=en_model, tokenizer=en_tokenizer)
    return en_pipeline


def ppl(model, text):
    result = model(text)
    return result


def main():
    model_version = "bert-base-uncased"
    model = load_model(cache_dir=True, download=False)
    datasets_dir = DATASETS_DIR
    # train_dataset = datasets.load_dataset(path='wmt16', name="de-en", split="train",
    #                                       cache_dir=datasets_dir)
    train_dataset = datasets.load_dataset(
        os.path.join(datasets_dir,
                     "wmt16"),
        name="de-en",
        split="train")
    test_texts = ["there is a book on the desk",
                  "there is a plane on the desk",
                  "there is a book in the desk"]
    for t in test_texts:
        print(ppl(model, t)["ppl"])

    data_lenth = len(train_dataset)
    score = [0] * data_lenth
    for idx, row in tqdm(enumerate(train_dataset)):
        model
        en, de = row["translation"]["en"], row["translation"]["de"]
        score = ppl(model, en)["ppl"] + ppl(model, de)["ppl"]

        if (idx + 1) % 1000 == 0:
            print(f"Score {idx + 1} k /{data_lenth} samples ...")

    index = [
        [116518, 41568, 13049, 39342, 23659, 76413],
        [12051, 113004, 57498, 51064, 47300, 47552],
        [73186, 50806, 17741, 94891, 55986, 44589],
        [69885, 114662, 32893, 103985, 85597, 84899],
    ]
    for i in index:
        for sentence in train_dataset[i]["translation"]:
            print(ppl(model, sentence["en"])["ppl"] + ppl(model, sentence["de"])["ppl"])
        print("-" * 30)


if __name__ == '__main__':
    main()

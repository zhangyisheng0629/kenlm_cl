import datasets, transformers
from transformers import BertTokenizer, BertForMaskedLM
from bert_perplexity.perplexity import PerplexityPipeline
import os


# download if not exists cache_dir
def load_model(model_name=None, cache_dir=None, download=False):
    assert not model_name or not cache_dir
    if download:
        model = download_model_files()
    else:
        model = load_from_cache()
    return model


# download_model_files
def download_model_files(pretrained_models_dir="F:/pretrained_models", hf_name="bert-base-multilingual-cased",
                         local_name="bert-base-multilingual-cased"):
    cache_dir = os.path.join(pretrained_models_dir, local_name)
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


def load_from_cache(pretrained_models_dir="F:/pretrained_models",
                    model_name="bert-base-uncased/models--bert-base-uncased"):
    en_tokenizer = BertTokenizer.from_pretrained(
        os.path.join(pretrained_models_dir, model_name)
    )
    en_model = BertForMaskedLM.from_pretrained(
        os.path.join(pretrained_models_dir, model_name))
    en_pipeline = PerplexityPipeline(model=en_model, tokenizer=en_tokenizer)
    return en_pipeline


def ppl(model, text):
    result = model(text)
    return result


def main():
    model_version = "bert-base-uncased"
    model = load_model(download=True)
    datasets_dir = "F:/datasets"
    train_dataset = datasets.load_dataset(path='wmt16', name="de-en", split="train",
                                          cache_dir=datasets_dir)
    # train_dataset = datasets.load_dataset(os.path.join(datasets_dir, "datasets"), split="train")
    test_texts = ["there is a book on the desk",
                  "there is a plane on the desk",
                  "there is a book in the desk"]
    for t in test_texts:
        print(ppl(model, t)["ppl"])
    index = [
        [116518, 41568, 13049, 39342, 23659, 76413],
        [12051, 113004, 57498, 51064, 47300, 47552],
        [73186, 50806, 17741, 94891, 55986, 44589],
        [69885, 114662, 32893, 103985, 85597, 84899],
    ]
    for i in index:
        for sentence in train_dataset[i]["translation"]:
            print(ppl(model, sentence["en"])["ppl"]+ppl(model, sentence["de"])["ppl"])
        print("-" * 30)


if __name__ == '__main__':
    main()

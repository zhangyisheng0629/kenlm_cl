import kenlm
import transformers, datasets
import os
import numpy as np

model = kenlm.Model('lm/test.arpa')
# print(model.score('this is a sentence .', bos=True, eos=True))
#
# print(model.score('there is a book on the desk .', ))
# print(model.score('there is a plane on the desk .', ))
# print(model.score('there is a book in the desk .'))


from datasets import inspect_dataset, load_dataset_builder

# inspect_dataset("wmt16", "path/to/scripts")
# builder = load_dataset_builder(
#     "path/to/scripts/wmt_utils.py",
#     language_pair=("fr", "de"),
#     subsets={
#         datasets.Split.TRAIN: ["commoncrawl_frde"],
#         datasets.Split.VALIDATION: ["euelections_dev2019"],
#     },
# )

# Standard version
# builder.download_and_prepare()
# ds = builder.as_dataset()

# Streamable version
# ds = builder.as_streaming_dataset()

# add the base_dir to the conf file
base_dir = "."
# train_dataset = datasets.load_dataset(path='wmt16', name="de-en", split="train",
#                                       cache_dir=os.path.join(base_dir, "datasets"))
train_dataset = datasets.load_dataset(os.path.join(base_dir, "datasets"), split="train")
# test_dataset = datasets.load_dataset(path='wmt16', name="de-en", split="test",
#                                      cache_dir=os.path.join(base_dir, "datasets"))

index = [
    [116518, 41568, 13049, 39342, 23659, 76413],
    [12051, 113004, 57498, 51064, 47300, 47552],
    [73186, 50806, 17741, 94891, 55986, 44589],
    [69885, 114662, 32893, 103985, 85597, 84899],
]
for i in index:
    for sentence in train_dataset[i]["translation"]:
        print(model.score(sentence["en"], bos=True, eos=True) + model.score(sentence["de"], bos=True, eos=True))

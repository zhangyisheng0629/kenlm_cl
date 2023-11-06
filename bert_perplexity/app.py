# coding=utf-8
# author: xusong
# time: 2022/8/23 16:06

from perplexity import PerplexityPipeline
from transformers import BertTokenizer, BertForMaskedLM
import gradio as gr
import time

en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
en_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
en_pipeline = PerplexityPipeline(model=en_model, tokenizer=en_tokenizer)

zh_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
zh_model = BertForMaskedLM.from_pretrained("bert-base-chinese")
zh_pipeline = PerplexityPipeline(model=zh_model, tokenizer=zh_tokenizer)


def ppl(model_version, text):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), model_version, text)
    if model_version == "bert-base-uncased":
        result = en_pipeline(text)
    else:
        result = zh_pipeline(text)
    return result["ppl"], result


examples = [
    ["bert-base-uncased", "New York City is located in the northeastern United States."],
    ["bert-base-uncased", "New York City is located in the western United States."],
    ["bert-base-chinese", "少先队员因该为老人让坐"],
]

css = "#json-container {height:: 400px; overflow: auto !important}"

corr_iface = gr.Interface(
    fn=ppl,
    inputs=[
        # gr.Dropdown(["bert-base-uncased", "bert-base-chinese"], value="bert-base-uncased"), # TODO 调整大小和位置
        gr.Radio(
            ["bert-base-uncased", "bert-base-chinese"],
            value="bert-base-uncased"
        ),
        gr.Textbox(
            value="New York City is located in the northeastern United States.",
            label="input text"
        )],
    outputs=[
        gr.Textbox(label="Perplexity"),
        gr.JSON(label="Tokens", elem_id="json-container")],
    examples=examples,
    title="BERT as Language Model",
    description='',
    css=css
)

if __name__ == "__main__":
    corr_iface.launch()

import torch
from transformers import pipeline, logging, AutoModelForCausalLM, AutoTokenizer
import gradio as gr

## 1 - Loading Model
## 1 - Loading Model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.load_adapter('checkpoint-960')

## 2 - Loading Tokenizer
tokenizer = AutoTokenizer.from_pretrained('checkpoint-960', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
def generate_dialogue(input_text):

  pipe = pipeline(task="text-generation",model=model,tokenizer=tokenizer,max_length=100)
  result = pipe(f"<s>[INST] {input_text} [/INST]")
  return result[0]['generated_text']
    
HTML_TEMPLATE = """
  <style>
    
    #app-header {
        text-align: center;
        background: rgba(255, 255, 255, 0.3); /* Semi-transparent white */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative; /* To position the artifacts */
    }
    #app-header h1 {
        color: #FF0000;
        font-size: 2em;
        margin-bottom: 10px;
    }
    .concept {
        position: relative;
        transition: transform 0.3s;
    }
    .concept:hover {
        transform: scale(1.1);
    }
    .concept img {
        width: 100px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .concept-description {
        position: absolute;
        bottom: -30px;
        left: 50%;
        transform: translateX(-50%);
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .concept:hover .concept-description {
        opacity: 1;
    }
    /* Artifacts */
    
</style>
<div id="app-header">
    <!-- Artifacts -->
    <div class="artifact large"></div>
    <div class="artifact large"></div>
    <div class="artifact large"></div>
    <div class="artifact large"></div>
    <!-- Content -->
    <h1>CHAT with fine tuned Phi-2 LLM</h1>
    <p>Generate dialogue for given some initial prompt for context.</p>
    <p>Model: Phi-2 (https://huggingface.co/microsoft/phi-2),  Dataset: oasst1 (https://huggingface.co/datasets/OpenAssistant/oasst1) </p>
"""

with gr.Blocks(theme=gr.themes.Glass(),css=".gradio-container {background: url('https://github.com/Manjunath-Yelipeta/S27_Era/blob/main/bg.jpg?raw=true')}") as interface:
    gr.HTML(value=HTML_TEMPLATE, show_label=False)

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    
    with gr.Row():

        input_text = gr.Textbox(
            label="Input Text", 
            value="Enter your prompt here: This text will set the context for the AI's response."
        )

        outputs = gr.Textbox(
            label="Answer"
        )
        inputs = [input_text]
   
    with gr.Column():
        button = gr.Button("Ask me")
        button.click(generate_dialogue, inputs=inputs, outputs=outputs)

interface.launch()

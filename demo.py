import gradio as gr
from core import Core


def process_files(question, files):
    file_names = [f.name for f in files]
    core_fn = Core(question, file_names)
    # process dataset
    core_fn.process_documents()
    result = core_fn.load_and_run()
    return result

demo = gr.Interface(
    process_files,
    inputs=['textbox', 'files'],
    outputs="textbox"
)

demo.launch()

import modal 

app = modal.App("tiny-sota-examples")
image = (modal.Image.debian_slim()
    .pip_install("torch")
    .add_local_file("../dist/tiny_sota-0.0.3-py3-none-any.whl", remote_path="/root/tiny_sota-0.0.3-py3-none-any.whl", copy=True)
     .add_local_file("./qwen_inference.py", remote_path="/root/qwen_inference.py", copy=True)
    .run_commands("pip install /root/tiny_sota-0.0.3-py3-none-any.whl"))

@app.function(image=image, gpu="A100-40GB")
def run_func():
    import subprocess
    result = subprocess.run(
        ["python", "/root/qwen_inference.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.stdout


@app.local_entrypoint()
def main():
    run_func.remote()
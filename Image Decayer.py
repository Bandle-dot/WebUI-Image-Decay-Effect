import importlib.util
import math
import os
import subprocess
import sys
import tempfile

# ==============================================================================
# 1. DEPENDENCY CHECK & AUTO-INSTALLATION
# ==============================================================================
REQUIRED_LIBS = {
    "torch": "torch torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "gradio": "gradio",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "PIL": "Pillow",
    "numpy": "numpy",
}


def install_missing_dependencies():
    missing = []
    for module_name, pip_name in REQUIRED_LIBS.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(pip_name)

    if missing:
        print("Missing required libraries. Auto-installing dependencies...")
        for pkg in missing:
            print(f"Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *pkg.split()]
            )
        print("Dependencies installed successfully!\n")


install_missing_dependencies()

# ==============================================================================
# 2. IMPORTS
# ==============================================================================
from diffusers import AutoPipelineForImage2Image
import gradio as gr
import imageio
import numpy as np
from PIL import Image
import torch

CURRENT_PIPE = None
CURRENT_DEVICE = None
MODEL_ID = "stabilityai/sd-turbo"


def get_pipeline(device: str):
    """Loads and caches the SD-Turbo model cleanly."""
    global CURRENT_PIPE, CURRENT_DEVICE

    if CURRENT_PIPE is not None and CURRENT_DEVICE == device:
        return CURRENT_PIPE

    print(f"Loading {MODEL_ID} onto {device.upper()}...")
    dtype = (
        torch.float16
        if device == "cuda" and torch.cuda.is_available()
        else torch.float32
    )

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.to(device)

    CURRENT_PIPE = pipe
    CURRENT_DEVICE = device
    return CURRENT_PIPE


def get_file_path(file_input):
    """Safely extracts the file path across different Gradio versions."""
    if file_input is None:
        return None
    if isinstance(file_input, str):
        return file_input
    if hasattr(file_input, "name"):
        return file_input.name
    return str(file_input)


def extract_initial_frame(file_path: str) -> Image.Image:
    """Extracts a PIL Image from either an image or video file."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        reader = imageio.get_reader(file_path)
        first_frame = reader.get_data(0)
        reader.close()
        return Image.fromarray(first_frame).convert("RGB")
    else:
        return Image.open(file_path).convert("RGB")


# ==============================================================================
# 3. CORE GENERATIVE DECAY PIPELINE
# ==============================================================================
def run_decay_loop(
    input_file,
    iterations,
    decay_strength,
    fps,
    device_choice,
    progress=gr.Progress(track_tqdm=True),
):
    path = get_file_path(input_file)
    if not path:
        raise gr.Error("Please upload an image or video first!")

    if device_choice == "cuda" and not torch.cuda.is_available():
        gr.Warning("CUDA (GPU) not detected on this machine. Falling back to CPU.")
        device_choice = "cpu"

    pipe = get_pipeline(device_choice)

    # 1. Prepare initial image (SD-Turbo expects 512x512)
    init_image = extract_initial_frame(path)
    init_image = init_image.resize((512, 512), Image.Resampling.LANCZOS)

    frames = [init_image]
    current_image = init_image

    prompt = "original photo, unaltered, keep exact same details, preserve scene perfectly, photorealistic, 8k"

    # Fix: Ensure effective inference steps are always at least 1
    # SD-Turbo only needs 1 real step to mutate
    calc_steps = max(2, int(math.ceil(1.0 / max(decay_strength, 0.05))))

    print(
        f"Starting decay generation for {iterations} cycles on {device_choice.upper()}..."
    )

    # 2. Feedback Loop
    with torch.inference_mode():
        for _ in progress.tqdm(
            range(int(iterations)), desc="Generating Decay Frames"
        ):
            result = pipe(
                prompt=prompt,
                image=current_image,
                num_inference_steps=calc_steps,
                strength=decay_strength,
                guidance_scale=0.0,
            ).images[0]

            frames.append(result)
            current_image = result  # Output becomes the next input

    # 3. Stitch frames into a web-compatible H.264 MP4
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_path = temp_video.name
    temp_video.close()

    writer = imageio.get_writer(
        temp_video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",  # Required for browser playback
    )
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()

    return temp_video_path, frames[-1]


# ==============================================================================
# 4. GUI SETUP
# ==============================================================================
custom_theme = gr.themes.Soft(
    primary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

with gr.Blocks(title="AI Recursive Decay Engine") as demo:
    gr.Markdown(
        """
        # 🧬 AI Recursive Visual Decay Engine
        Upload an image or video. The AI is told to make **zero changes**, but continuously feeding its own output back into itself creates an uncanny visual feedback loop.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(
                label="Input Image or Video",
                file_types=["image", "video"],
                type="filepath",
            )

            with gr.Accordion("Engine Settings", open=True):
                device_choice = gr.Radio(
                    choices=["cuda", "cpu"],
                    value="cuda" if torch.cuda.is_available() else "cpu",
                    label="Compute Hardware",
                    info="Use CUDA for near-instant generations.",
                )

                iterations = gr.Slider(
                    minimum=5,
                    maximum=100,
                    value=20,
                    step=1,
                    label="Decay Generations (Cycles)",
                    info="How many feedback loops to run.",
                )

                decay_strength = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.35,
                    step=0.05,
                    label="Mutation / Decay Step Strength",
                    info="Lower = slow subtle drift. Higher = rapid degeneration.",
                )

                fps = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=8,
                    step=1,
                    label="Output Video FPS",
                    info="Frames per second in the rendered video.",
                )

            generate_btn = gr.Button("⚡ Generate Decay Sequence", variant="primary")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Decay Video Sequence", autoplay=True)
            output_final_frame = gr.Image(label="Final Mutated Frame", type="pil")

    generate_btn.click(
        fn=run_decay_loop,
        inputs=[
            input_file,
            iterations,
            decay_strength,
            fps,
            device_choice,
        ],
        outputs=[output_video, output_final_frame],
    )

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, share=False, theme=custom_theme)
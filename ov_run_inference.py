import argparse
from PIL import Image
from optimum.intel.openvino import OVModelForVision2Seq
from transformers import AutoProcessor


def run_inference_ov(ov_model_path: str, image_path: str, prompt: str, device: str = "CPU"):
    """
    Runs inference using the exported Dolphin OpenVINO IR model.
    
    Args:
        ov_model_path (str): Path to the directory containing the OpenVINO IR model (.xml/.bin).
        image_path (str): Path to the input image.
        prompt (str): The prompt for the model.
        device (str): Target device for inference (e.g., "CPU", "GPU"). Default is "CPU".
    """
    print(f"Loading OpenVINO model from: {ov_model_path} on device: {device}")

    # Load OpenVINO model and processor
    model = OVModelForVision2Seq.from_pretrained(
        ov_model_path,
        device=device,
        compile=True,  # Compile for inference immediately
        trust_remote_code=False,
    )
    processor = AutoProcessor.from_pretrained(ov_model_path)
    tokenizer = processor.tokenizer

    print(f"Loading image from: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    print("Preparing inputs for the model...")
    # 1. Process the image
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 2. Process the prompt
    task_prompt = f"<s>{prompt} <Answer/>"
    decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    print("Running generation with OpenVINO model...")
    # Generate using OpenVINO backend
    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=4096,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    print("Decoding generated sequence...")
    sequence = tokenizer.batch_decode(outputs.sequences)[0]

    # Clean up the output string
    result = (
        sequence
        .replace(tokenizer.eos_token, "")
        .replace(tokenizer.pad_token, "")
        .replace(task_prompt, "")
        .strip()
    )

    print("\n" + "="*20 + " INFERENCE RESULT " + "="*20)
    print(f"Prompt: {prompt}")
    print("-" * 58)
    print(f"Result:\n{result}")
    print("=" * 58)


def main():
    parser = argparse.ArgumentParser(description="Run inference with Dolphin OpenVINO IR model.")
    parser.add_argument(
        "--ov_model_path",
        type=str,
        default="./ov_models/Dolphin-1.5_ov_fp16",
        help="Path to the directory with the exported OpenVINO IR model.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./demo/element_imgs/table.jpg",
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Parse the table in the image.",
        help="Prompt to guide the model's generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "AUTO", "MULTI:CPU,GPU"],
        help="Target device for OpenVINO inference. Default: CPU.",
    )
    args = parser.parse_args()

    run_inference_ov(args.ov_model_path, args.image_path, args.prompt, device=args.device)


if __name__ == "__main__":
    main()
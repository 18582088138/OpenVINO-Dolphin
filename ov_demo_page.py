"""
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: MIT
(Adapted for OpenVINO inference)
"""

import argparse
import glob
import os
import traceback

import cv2
from PIL import Image

# OpenVINO-related imports
from optimum.intel.openvino import OVModelForVision2Seq
from transformers import AutoProcessor

# Assuming utils.utils contains the necessary helper functions:
# convert_pdf_to_images, save_combined_pdf_results, prepare_image,
# parse_layout_string, process_coordinates, save_figure_to_local,
# save_outputs, setup_output_dirs
from utils.utils import *


class DOLPHIN_OpenVINO:
    def __init__(self, ov_model_path: str, device: str = "CPU"):
        """
        Initializes the DOLPHIN model using OpenVINO.
        Args:
            ov_model_path (str): Path to the directory containing the OpenVINO IR model files.
            device (str): Target device for inference ("CPU", "GPU", etc.)
        """
        self.device = device.upper()
        print(f"Loading OpenVINO model from: {ov_model_path} onto device: {self.device}")

        try:
            # Load OpenVINO IR model (encoder + decoder + past decoder)
            self.model = OVModelForVision2Seq.from_pretrained(
                ov_model_path,
                device=self.device,
                compile=True,  # Compile immediately for faster first inference
                trust_remote_code=False,
            )
            self.processor = AutoProcessor.from_pretrained(ov_model_path)
            self.tokenizer = self.processor.tokenizer
            print("OpenVINO model and processor loaded successfully.")
        except Exception as e:
            print(f"Error loading OpenVINO model from {ov_model_path}: {e}")
            traceback.print_exc()
            raise

    def chat(self, prompt, image):
        """
        Processes an image or a batch of images with the given prompt(s) using the OpenVINO model.
        Args:
            prompt (str or list[str]): Text prompt or list of prompts.
            image (PIL.Image.Image or list[PIL.Image.Image]): PIL Image or list of PIL Images.
        Returns:
            str or list[str]: Generated text or list of generated texts.
        """
        is_batch = isinstance(image, list)

        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)

        # 1. Process images
        try:
            pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        except Exception as e:
            print(f"Error processing images: {e}")
            traceback.print_exc()
            return "Error processing image." if not is_batch else ["Error processing image."] * len(images)

        # 2. Process prompts
        task_prompts = [f"<s>{p} <Answer/>" for p in prompts]
        try:
            prompt_inputs = self.tokenizer(
                task_prompts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            )
            decoder_input_ids = prompt_inputs.input_ids
            decoder_attention_mask = prompt_inputs.attention_mask
        except Exception as e:
            print(f"Error tokenizing prompts: {e}")
            traceback.print_exc()
            return "Error processing prompt." if not is_batch else ["Error processing prompt."] * len(images)

        # Note: OpenVINO models run on CPU/GPU via OpenVINO runtime; no need to .to(device)
        # The inputs are already PyTorch tensors but will be converted internally by OVModel

        # 3. Generate
        try:
            outputs = self.model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=4096,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.tokenizer.unk_token_id]] if self.tokenizer.unk_token_id is not None else None,
                return_dict_in_generate=True,
            )
        except Exception as e:
            print(f"Error during model generation: {e}")
            traceback.print_exc()
            return "Error during generation." if not is_batch else ["Error during generation."] * len(images)

        # 4. Decode and clean
        sequences = self.tokenizer.batch_decode(outputs.sequences)

        results = []
        for i, sequence in enumerate(sequences):
            cleaned_sequence = (
                sequence
                .replace(task_prompts[i], "")
                .replace(self.tokenizer.eos_token, "")
                .replace(self.tokenizer.pad_token, "")
                .strip()
            )
            if cleaned_sequence.startswith("<s>"):
                cleaned_sequence = cleaned_sequence[len("<s>") :].strip()
            results.append(cleaned_sequence)

        return results[0] if not is_batch else results


def process_document(document_path, model, save_dir, max_batch_size=None):
    """Parse documents - Handles both images and PDFs"""
    file_ext = os.path.splitext(document_path)[1].lower()

    if file_ext == ".pdf":
        images = convert_pdf_to_images(document_path)
        if not images:
            print(f"Warning: Failed to convert PDF {document_path} to images, or PDF is empty.")
            return None, []

        all_results = []
        for page_idx, pil_image in enumerate(images):
            print(f"Processing page {page_idx + 1}/{len(images)}")
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"

            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size, save_individual=False
            )
            if recognition_results is None:
                print(f"Warning: Failed to process page {page_idx + 1} of {document_path}.")
                continue

            all_results.append({"page_number": page_idx + 1, "elements": recognition_results})

        if not all_results:
            return None, []

        combined_json_path = save_combined_pdf_results(all_results, document_path, save_dir)
        return combined_json_path, all_results
    else:
        try:
            pil_image = Image.open(document_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {document_path}")
            return None, None
        except Exception as e:
            print(f"Error opening image {document_path}: {e}")
            return None, None

        base_name = os.path.splitext(os.path.basename(document_path))[0]
        return process_single_image(pil_image, model, save_dir, base_name, max_batch_size)


def process_single_image(image, model, save_dir, image_name, max_batch_size=None, save_individual=True):
    """Process a single image"""
    print("Step 1: Page-level layout and reading order parsing...")
    try:
        layout_output = model.chat("Parse the reading order of this document.", image)
        if "Error" in layout_output:
            print(f"Error in layout parsing for {image_name}: {layout_output}")
            return None, None
    except Exception as e:
        print(f"Exception during layout parsing for {image_name}: {e}")
        traceback.print_exc()
        return None, None

    print("Step 2: Element-level content parsing...")
    try:
        padded_image, dims = prepare_image(image)
        recognition_results = process_elements(layout_output, padded_image, dims, model, max_batch_size, save_dir, image_name)
    except Exception as e:
        print(f"Exception during element processing for {image_name}: {e}")
        traceback.print_exc()
        return None, None

    json_path = None
    if save_individual and recognition_results is not None:
        dummy_image_path = f"{image_name}.jpg"
        json_path = save_outputs(recognition_results, dummy_image_path, save_dir)

    return json_path, recognition_results


def process_elements(layout_results_str, padded_image, dims, model, max_batch_size, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding"""
    try:
        layout_results = parse_layout_string(layout_results_str)
    except Exception as e:
        print(f"Error parsing layout string: {layout_results_str}. Error: {e}")
        return []

    text_elements = []
    table_elements = []
    figure_results = []
    previous_box = None
    reading_order = 0

    for bbox, label in layout_results:
        try:
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )
            cropped = padded_image[y1:y2, x1:x2]

            if cropped.size > 0:
                pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                element_info = {
                    "crop": pil_crop,
                    "label": label,
                    "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                    "reading_order": reading_order,
                }
                if label == "fig":
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    figure_results.append({
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                        "text": f"[Figure](figures/{figure_filename})",
                        "figure_path": f"figures/{figure_filename}",
                    })
                elif label == "tab":
                    table_elements.append(element_info)
                else:
                    text_elements.append(element_info)
            reading_order += 1
        except Exception as e:
            print(f"Error processing bbox {bbox} with label {label}: {e}")
            traceback.print_exc()
            continue

    recognition_results = list(figure_results)
    if text_elements:
        text_parsed_results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(text_parsed_results)

    if table_elements:
        table_parsed_results = process_element_batch(table_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(table_parsed_results)

    recognition_results.sort(key=lambda x: x.get("reading_order", float('inf')))

    for result in recognition_results:
        result.pop("crop", None)

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    """Process elements of the same type in batches"""
    parsed_results = []
    actual_batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        actual_batch_size = min(actual_batch_size, max_batch_size)

    if actual_batch_size == 0:
        return []

    for i in range(0, len(elements), actual_batch_size):
        batch_elements = elements[i:i + actual_batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]

        print(f"  Processing batch of {len(crops_list)} elements with prompt: '{prompt}'")
        try:
            batch_chat_results = model.chat(prompt, crops_list)
        except Exception as e:
            print(f"    Error in model.chat for batch: {e}")
            traceback.print_exc()
            for elem in batch_elements:
                parsed_results.append({
                    "label": elem["label"],
                    "bbox": elem["bbox"],
                    "text": "Error during batch processing.",
                    "reading_order": elem["reading_order"],
                })
            continue

        for j, chat_result_text in enumerate(batch_chat_results):
            elem = batch_elements[j]
            parsed_results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": chat_result_text.strip() if isinstance(chat_result_text, str) else "Error: Invalid chat result",
                "reading_order": elem["reading_order"],
            })
    return parsed_results


def main():
    parser = argparse.ArgumentParser(description="Document parsing with DOLPHIN OpenVINO models.")
    parser.add_argument(
        "--ov_model_path",
        type=str,
        default="./ov_models/Dolphin-1.5_ov_fp16/",
        help="Path to the directory containing the exported OpenVINO IR model and processor files.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./demo/page_imgs/",
        help="Path to input image/PDF or directory of files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save parsing results (default: same as input directory or its parent).",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=4,
        help="Maximum number of document elements to parse in a single batch (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "AUTO", "MULTI:CPU,GPU"],
        help='Target device for OpenVINO inference (e.g., "CPU", "GPU"). Default: "CPU".',
    )
    args = parser.parse_args()

    print(f"Initializing DOLPHIN OpenVINO model from: {args.ov_model_path} on device: {args.device}")
    try:
        model = DOLPHIN_OpenVINO(args.ov_model_path, device=args.device)
    except Exception as e:
        print(f"Failed to initialize DOLPHIN_OpenVINO model: {e}")
        return

    if os.path.isdir(args.input_path):
        file_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".pdf", ".PDF"]
        document_files = []
        for ext in file_extensions:
            document_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        document_files = sorted(document_files)
    else:
        if not os.path.exists(args.input_path):
            print(f"Error: Input path {args.input_path} does not exist.")
            return
        file_ext = os.path.splitext(args.input_path)[1].lower()
        supported_exts = ['.jpg', '.jpeg', '.png', '.pdf']
        if file_ext not in supported_exts:
            print(f"Error: Unsupported file type: {file_ext}. Supported types: {supported_exts}")
            return
        document_files = [args.input_path]

    if not document_files:
        print(f"No documents found to process in {args.input_path}.")
        return

    effective_save_dir = args.save_dir or \
                         (args.input_path if os.path.isdir(args.input_path) else os.path.dirname(args.input_path))
    if not effective_save_dir:
        effective_save_dir = "."

    setup_output_dirs(effective_save_dir)

    print(f"\nTotal files to process: {len(document_files)}")
    for file_path in document_files:
        print(f"\nProcessing {file_path}...")
        try:
            json_path, _ = process_document(
                document_path=file_path,
                model=model,
                save_dir=effective_save_dir,
                max_batch_size=args.max_batch_size,
            )
            if json_path:
                print(f"Processing completed. Results saved in {effective_save_dir}. Main JSON: {json_path}")
            else:
                print(f"Processing for {file_path} completed, but no main JSON was generated.")

        except Exception as e:
            print(f"Critical error processing {file_path}: {e}")
            traceback.print_exc()
            continue
    print("\nAll documents processed.")


if __name__ == "__main__":
    main()
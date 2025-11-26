import argparse
import os
import shutil
from pathlib import Path
import traceback

from optimum.intel.openvino import OVModelForVision2Seq
from transformers import AutoProcessor


def export_dolphin_to_openvino(model_id: str, output_dir: str, fp16: bool = True):
    """
    Exports the ByteDance/Dolphin model to OpenVINO IR format.
    
    Args:
        model_id (str): The Hugging Face model identifier.
        output_dir (str): The directory path where the OpenVINO IR model will be saved.
                          Example: "./dolphin_ov_fp16"
        fp16 (bool): Whether to use FP16 precision (default and recommended for OpenVINO).
                     Note: OpenVINO export in optimum always uses FP16 internally for best performance.
    """
    output_path = Path(output_dir)

    # --- Cleanup ---
    if output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=False)
    print(f"Exporting model to OpenVINO IR at: {output_path}")

    try:
        # --- Step 1: Load and export to OpenVINO ---
        print("\n[Step 1/2] Loading model and exporting to OpenVINO IR...")
        ov_model = OVModelForVision2Seq.from_pretrained(
            model_id,
            export=True,          # Trigger export from PyTorch/HF to OpenVINO
            compile=False,        # Do not compile for inference yet (just export)
            load_in_8bit=False,   # Keep full precision during export
        )
        processor = AutoProcessor.from_pretrained(model_id)

        # --- Step 2: Save model and processor ---
        print(f"[Step 2/2] Saving OpenVINO IR model and processor to {output_path}...")
        ov_model.save_pretrained(output_path)
        processor.save_pretrained(output_path)

        print("✅ OpenVINO IR export completed successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred during OpenVINO export.")
        print(f"Error details: {e}")
        traceback.print_exc()
        if output_path.exists():
            print(f"Cleaning up output directory due to error: {output_path}")
            shutil.rmtree(output_path)
        return

    # --- Final Summary ---
    print("\n" + "=" * 50)
    print("✅ OpenVINO IR export process completed!")
    if output_path.exists():
        print(f"   Model saved to: {output_path.resolve()}")
        print(f"   Precision: {'FP16 (default for OpenVINO)' if fp16 else 'FP32 (not typical)'}")
        print("The output directory contains:")
        for file_item in sorted(os.listdir(output_path)):
            print(f"   - {file_item}")
    else:
        print(f"⚠️ Warning: Output directory {output_path.resolve()} was not created.")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Export ByteDance/Dolphin model to OpenVINO IR.")
    parser.add_argument("--model_id", type=str, default="ByteDance/Dolphin-1.5", help="Hugging Face model ID.")
    parser.add_argument("--output_dir", type=str, default="./ov_models/Dolphin-1.5_ov", 
                        help="Output directory for the OpenVINO IR model. Suffix '_fp16' is added by default.")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 precision (enabled by default; OpenVINO uses FP16 internally).")
    args = parser.parse_args()

    # OpenVINO typically uses FP16, so we append _fp16 for clarity
    actual_output_dir = f"{args.output_dir}_fp16"

    export_dolphin_to_openvino(args.model_id, actual_output_dir, fp16=args.fp16)


if __name__ == "__main__":
    main()
import torch
import safetensors.torch

# Import our new model and its configuration
from src.openpi.models.pi0_fast import Pi0FASTConfig
from src.openpi.models_pytorch.pi0_fast_pytorch import PI0FastPytorch

def verify_architecture_with_checkpoint():
    """
    Verifies that the programmatic model architecture matches the provided checkpoint
    by attempting to load the weights.
    """
    # --- Configuration ---
    # Path to the model.safetensors FILE itself.
    CHECKPOINT_FILE_PATH = "data/openpi-assets/checkpoints/torch_pi0_fast_base/model.safetensors"
    
    # This should match the variant used to create the checkpoint.
    MODEL_VARIANT = "gemma_2b"

    print("--- Verification Plan (Revised) ---")
    print("1. Build Programmatic Model using our new PI0FastPytorch class.")
    print("2. Attempt to load the provided checkpoint weights into this model.")
    print("3. If loading succeeds with strict=True, the architecture is correct.\n")

    # --- Step 1: Build Our Programmatic Model ---
    try:
        print(f"Building programmatic model with variant '{MODEL_VARIANT}'...")
        pi0_fast_config = Pi0FASTConfig(paligemma_variant=MODEL_VARIANT)
        
        # Instantiate our complete model wrapper, as this is what the training
        # script saves and loads.
        programmatic_model_wrapper = PI0FastPytorch(config=pi0_fast_config)
        print("Programmatic model built successfully.\n")
    except Exception as e:
        print("\n❌ FAILURE: An error occurred during model construction.")
        print(f"Please check the Pi0FASTConfig and PI0FastPytorch __init__ method.")
        print(f"\nOriginal Error:\n{e}")
        return

    # --- Step 2: Attempt to Load Weights ---
    print(f"Attempting to load weights from: {CHECKPOINT_FILE_PATH}")
    try:
        # Use safetensors.torch.load_model with strict=True.
        # This will raise a detailed error if any keys are missing, unexpected,
        # or if any tensor shapes do not match.
        missing_keys, unexpected_keys = safetensors.torch.load_model(
            programmatic_model_wrapper, CHECKPOINT_FILE_PATH, strict=True
        )
        
        # The strict=True flag already ensures this, but we double-check for clarity.
        if missing_keys or unexpected_keys:
             print("\n❌ FAILURE: Incompatible keys found during loading.")
             if missing_keys:
                 print(f"  - Missing keys (expected by model but not in checkpoint): {missing_keys}")
             if unexpected_keys:
                 print(f"  - Unexpected keys (in checkpoint but not in model): {unexpected_keys}")
        else:
             print("\n✅ SUCCESS: Checkpoint weights loaded successfully with strict checking.")
             print("Conclusion: The parameter mapping in PI0FastPytorch correctly constructs an architecture that matches the checkpoint.")

    except Exception as e:
        print("\n❌ FAILURE: An error occurred during weight loading.")
        print("This indicates a mismatch in parameter names or tensor shapes between your checkpoint and the programmatic model.")
        print("This means the parameter mapping is likely incorrect.")
        print("\nOriginal Error Detail:")
        print(e)

if __name__ == "__main__":
    verify_architecture_with_checkpoint()
"""
Modal Secret Management for HRM Training

Defines required secrets and helper functions for secret creation.
"""

import modal as modal_client

# ─────────────────────────────────────────────────────────────────
# Required Secrets
# ─────────────────────────────────────────────────────────────────

REQUIRED_SECRETS = [
    "hf-token",      # Hugging Face token (write access for push_to_hub)
    "wandb-api-key", # Weights & Biases API key
]

OPTIONAL_SECRETS = [
    "modal-token",   # Modal token (usually auto-configured)
]

# ─────────────────────────────────────────────────────────────────
# Secret Creation Helpers
# ─────────────────────────────────────────────────────────────────

def create_secrets_cli():
    """Print CLI commands to create required secrets."""
    print("Run these commands to create required secrets:")
    print()
    print("# Hugging Face Token (with write access)")
    print("modal secret create hf-token HF_TOKEN=<your_hf_token>")
    print()
    print("# Weights & Biases API Key")
    print("modal secret create wandb-api-key WANDB_API_KEY=<your_wandb_key>")
    print()
    print("# Optional: Modal token (if not using modal setup)")
    print("modal secret create modal-token MODAL_TOKEN=<your_modal_token>")
    print()

def get_secrets_for_function(*secret_names: str) -> list:
    """Get list of modal.Secret objects for function decoration."""
    return [modal_client.Secret.from_name(name) for name in secret_names]

# Default secrets for training functions
TRAINING_SECRETS = get_secrets_for_function("hf-token", "wandb-api-key")
DOWNLOAD_SECRETS = get_secrets_for_function("hf-token")
EVAL_SECRETS = get_secrets_for_function("hf-token", "wandb-api-key")

# ─────────────────────────────────────────────────────────────────
# Environment Variable Helpers
# ─────────────────────────────────────────────────────────────────

def get_env_vars() -> dict:
    """Get standard environment variables for training containers."""
    return {
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "WANDB_DIR": "/checkpoints/wandb",
        "WANDB_CACHE_DIR": "/checkpoints/wandb/cache",
        "TORCH_HOME": "/root/.cache/torch",
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        "NCCL_DEBUG": "INFO",
        "PYTHONPATH": "/root/HRM/src",
    }

# ─────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────

def validate_secrets_exist() -> bool:
    """Check if all required secrets exist (run locally)."""
    import subprocess
    result = subprocess.run(["modal", "secret", "list"], capture_output=True, text=True)
    existing = result.stdout
    missing = [s for s in REQUIRED_SECRETS if s not in existing]
    if missing:
        print(f"Missing secrets: {missing}")
        return False
    print("All required secrets exist")
    return True
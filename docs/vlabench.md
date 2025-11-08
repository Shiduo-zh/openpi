# Additional Features for π₀ and π₀.₅ Models

This repository extends the original openpi framework with additional configuration examples and features, specifically supporting stop-gradient training for π₀ and π₀.₅ models.

## Stop-Gradient Training Support

We have added configuration examples in `src/openpi/training/config.py` that support stop-gradient training for Jax implementations of π₀ and π₀.₅ models. This allows for more stable training by preventing gradients from flowing through certain parts of the model during specific training phases.

### Key Features:

- **Stop-gradient configurations**: New training configurations that properly implement stop-gradient operations during training for π₀ and π₀.₅ models
- **Cross-framework support**: Both JAX and PyTorch implementations are supported
- **Flexible checkpoint compatibility**: Configurations work with existing π₀ and π₀.₅ checkpoints

### Usage:

To use the new stop-gradient training configurations:

1. Ensure you have the correct checkpoint for your model (π₀ or π₀.₅)
2. Use the appropriate training configuration from `src/openpi/training/config.py`
3. Run training as usual with the new configuration

For example:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py your_stop_gradient_config --exp-name=my_experiment --overwrite
```

For PyTorch training:
```bash
uv run scripts/train_pytorch.py your_stop_gradient_config --exp_name my_experiment
```

### Important Notes:

- Make sure your checkpoint is compatible with the model type you're training
- The stop-gradient configurations are designed to improve training stability
- When using PyTorch, ensure you've applied the transformers library patches as described in the main README
- Some features available in JAX may not yet be supported in PyTorch (supported for pifast are in progress)

The additional configuration examples can be found throughout the `src/openpi/training/config.py` file, alongside the original openpi configurations.
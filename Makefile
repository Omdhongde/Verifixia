# Makefile for common development tasks
# Usage: make <target>

PYTHON ?= python3.11
VENV_DIR := Backend/venv

.PHONY: help backend env install train-sklearn download-models clean

help:
	@echo "available targets:"
	@grep -E '^[a-zA-Z_-]+:' Makefile | awk -F: '{print "  "$$1}'

# create virtual environment and install dependencies
env: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	@echo "Creating virtualenv using $(PYTHON)"
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/python -m pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r Backend/requirements.txt
	@echo "virtualenv ready"

env-pytorch: env
	@echo "Installing CPU PyTorch wheels"
	$(VENV_DIR)/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# start backend server
backend: env
	@echo "Starting backend (press Ctrl+C to stop)"
	cd Backend && ./run_backend.sh

# convenience wrapper to run training script
train-sklearn: env
	@echo "Training sklearn detector using local dataset"
	cd scripts && ../$(VENV_DIR)/bin/python train_sklearn.py

# download pretrained models using helper script
download-models: env
	@echo "Downloading pretrained model files"
	cd scripts && ../$(VENV_DIR)/bin/python download_pretrained_models.py $(filter-out $@,$(MAKECMDGOALS))

# remove virtualenv (dangerous!)
clean:
	rm -rf $(VENV_DIR)

# allow passing extra arguments to download-models
%:
	@:

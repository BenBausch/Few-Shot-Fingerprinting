# Few-Shot Fingerprinting

## Overview
This repository contains the code for few-shot fingerprinting implementation.

## Running the Code

### Prerequisites
Make sure you have all required dependencies installed.

### Configuration
1. Locate the generic config files in the `configs/` directory
2. Create a copy of the appropriate config file
3. Modify the configuration according to your needs:
    - Set `"MODE": "TRAIN",` for training
    - Set `"MODE": "TEST",` for testing/inference
    - Adjust other parameters as needed
4. For Testing set the model checkpoint

### Execution
Run the code using:
```bash
python src/main.py --config="path/to/your/config.yaml"
```

Example:
```bash
# For training
python src/main.py --config="configs/train_config.yaml"

# For testing
python src/main.py --config="configs/test_config.yaml"
```

## Additional Information
For more detailed information about configuration options and parameters, please refer to the documentation in the `configs/` directory.

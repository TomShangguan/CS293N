# CS293N: Concept-Based Explanations for Network Traffic Analysis

This project applies concept-based explanations to network traffic analysis, focusing on classifying network traffic as benign or attack traffic. It builds upon the netFound, a deep learning model for network traffic analysis.

## Overview

The system uses a multi-stage approach:
1. Extract and process network packet headers from PCAP files
2. Generate human-readable descriptions of the PCAP files using LLMs
3. Compute similarities between concepts and descriptions
4. Train the concept mapping and output mapping functions with the fine-tuned netFound model
5. Train an output mapping layer to connect concepts to final classification decisions

This implementation is based on the research paper: "Toward Trustworthy Learning-Enabled Systems with Concept-Based Explanations".

## Requirements

### Python Dependencies

```
python>=3.8
torch
pyarrow
pandas
datasets
transformers
scikit-learn  
huggingface_hub
google-generativeai (or any other LLM)
```

## File Structure

- `packet_headers.txt`: Contains extracted packet header information
- `concepts.txt`: Contains the human-readable security concepts for mapping
- `Concept&OutputMappingTrainning.ipynb`: Jupyter notebook for training both concept mapping and output mapping
- `extract_pcap_header.py`: Script for extracting headers from PCAP files
- `similiarty.py`: Use text embeddings to calculate semantic similarities between network state descriptions and security concepts. Quantizes the similarties to scores into a range (0-4) to indicate how strongly each concept is expressed in each network state.
- `state_description.py`: Uses LLM to generate network state descriptions
- `attack_data/`: Directory containing attack traffic samples (in /0/) and benign traffic samples (in /1/)

## How to Run

You can run the entire pipeline using the main script, which executes steps 1-3 in sequence:

```bash
python main.py
```

Alternatively, you can execute each step individually as described below:

### 1. Extract PCAP Headers

This step processes the PCAP files from the attack_data directory, extracting header information while excluding payload data:

```bash
# Process all PCAPfiles in the attack_data folder
python -c "import extract_pcap_header as header; header.batch_process_pcap_folder('attack_data')"

# Or process a single file
python extract_pcap_header.py --pcap_file example.pcap --output packet_headers.txt
```

This creates text files with packet headers, which are more human-readable than raw PCAP data.

### 2. Generate State Descriptions

Generate human-readable descriptions of the network states using Google Gemini or any other LLM:

```bash
# Set your Google API key
export GOOGLE_API_KEY="your-api-key"

# Run the state description generator
python -c "import state_description as sd; sd.generate_state_descriptions()"
```

This step creates descriptions of what's happening in each network session and saves them to a descriptions file.

### 3. Compute Concept Similarities

Calculate similarities between the defined concepts and the generated descriptions:

```bash
# Run the similarity computation
python -c "import similiarty as sim; sim.cos_similarity_concept_statement()"
```

This script uses text embeddings to compute semantic similarities between network states and security concepts, quantizing the results into scores (0-10).

### 4. Train the Concept Mapping and Output Mapping

```bash
# Run the concept and output mapping notebook
jupyter notebook "Concept&OutputMappingTrainning.ipynb"
```

This notebook requires several dependencies:
- System dependencies: cmake, g++, libpcap-dev, git, parallel, make
- Python dependencies: pyarrow, pandas, datasets, transformers, scikit-learn, huggingface_hub, torch, google-generativeai
- Fine-tuned netFound model

The notebook will:
- Set up the netFound environment
- Build necessary C++ tools (PcapPlusPlus, PcapSplitter)
- Load the preprocessed packet headers and Arrow files
- Process the concepts and concept scores
- Train the concept mapper on top of the netFound model
- Freeze the trained concept mapper
- Train an output head that maps concept scores to final classification decisions (attack vs. benign)


## Data Organization

The project uses a specific directory structure for data organization:
- `attack_data/0/`: Contains attack traffic data
- `attack_data/1/`: Contains benign traffic data

## Model Architecture

The model is built on top of the netFound base model with multiple layers:

1. **NetFoundBase**: Processes the tokenized packet data through transformer layers
2. **AttentivePooling**: Pools the transformer outputs
3. **ConceptMapper**: Maps the pooled representations to concept scores
4. **OutputHead**: Maps concept scores to final classification decisions

## References

- "Toward Trustworthy Learning-Enabled Systems with Concept-Based Explanations"
- [netFound](https://github.com/snlucsb/netFound)

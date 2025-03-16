 Enhanced Verification System for Natural Reasoning

## Overview

The Enhanced Verification System is a critical component of the Natural Reasoning Training System, designed to validate knowledge and concepts through multiple methods:

1. **External API Integration** - Leverages Wikipedia for fact-checking
2. **Verification Pipeline** - Intelligently selects the best verification method based on domain
3. **Verification Statistics** - Tracks the effectiveness of different verification methods

## System Architecture

The verification system consists of several interconnected components:

```
┌─────────────────────────┐      ┌─────────────────────────┐
│                         │      │                         │
│  External Verification  │◄────►│   Verification Cache    │
│                         │      │                         │
└───────────┬─────────────┘      └─────────────────────────┘
            │
            ▼
┌─────────────────────────┐      
│                         │     
│ Verification Pipeline   │
│                         │     
└───────────┬─────────────┘      
            │
            ▼
┌─────────────────────────┐
│                         │
│  Knowledge Integration  │
│                         │
└─────────────────────────┘
```

## Key Features

### 1. External API Integration

The system integrates with external knowledge sources:

- **Wikipedia API** - For general knowledge verification
- **Heuristic Verification** - For domains where external verification is not available

### 2. Domain-Specific Verification

The system selects the appropriate verification method based on the concept's domain:

- **General Knowledge Domains** → Wikipedia
- **Other domains** → Heuristic verification

### 3. Verification Statistics

The system tracks:

- Number of concepts verified by each method
- Success rate of each verification method
- Average confidence scores

## Usage

### Command Line Interface

The training system now supports the following commands:

```bash
# Train with API verification
julia training_system_modular.jl train --dataset your_dataset.jsonl --use-api

# Run inference with a trained model
julia training_system_modular.jl inference --model trained_model.json --query "Your query here"
```

### API Key Configuration

To use external APIs, create an `api_keys.json` file with your API endpoints:

```json
{
    "wikipedia": "https://en.wikipedia.org/w/api.php"
}
```

The system will automatically create a template file if one doesn't exist.

## Testing

The verification system includes a test script (`test_verification.jl`) that demonstrates:

- External API verification for different domains
- Verification caching
- Verification statistics

Run the test script with:

```bash
julia test_verification.jl
```

## Example Usage

```julia
# Include the verification system
include("modules/verification_system.jl")
using .VerificationSystem

# Initialize the verification system
VerificationSystem.init_verification_system()

# Verify a concept
concept = "The Pythagorean theorem states that a^2 + b^2 = c^2"
domain = "Math"
is_valid, confidence = VerificationSystem.verify_concept(concept, domain)

println("Verification result: is_valid=$is_valid, confidence=$confidence")

# Print verification statistics
VerificationSystem.print_verification_summary()

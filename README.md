# DSPy-Helpers

Useful DSPy Modules

## IterativeRefiner

**IterativeRefiner** is a modular Python class designed for iterative refinement of outputs from a generative model using the `dspy` framework. This class performs a multi-step process to enhance and optimize model outputs through critical analysis and feedback loops.

#### Key Features:.
- **Automated Iterative Refinement**: Iteratively refines model outputs through a multi-phase process involving detailed critiques and improvement suggestions.
- **Extensibility**: Designed for easy extension and integration with other modules in `dspy` or other machine learning pipelines.

#### How It Works:
1. **Generate Phase**: The class first generates initial outputs based on given input parameters.
2. **Critique Phase**: The generated outputs are evaluated based on predefined criteria, identifying strengths and areas for improvement.
3. **Refinement Phase**: Guided by the critique, the class refines the initial outputs to align better with the specified quality standards.
4. **Iteration**: The refinement process can be repeated multiple times for continuous improvement.

#### Example Usage:
```python
import dspy
import IterativeRefiner

class MySignature(dspy.Signature):
    input_text: str = dspy.InputField(desc= 'The text input for the model')
    generated_summary: str = dspy.OutputField(desc='The generated summary of the input text'})

# Instantiate the IterativeRefiner with the signature
refiner = IterativeRefiner(MySignature)

# Run the forward pass for iterative refinement
refined_result = refiner.forward(parameters = {
    "input_text": "Artificial intelligence is rapidly transforming industries by automating tasks, enhancing decision-making, and enabling innovative solutions."
}, refine_iterations=3)

# Output the final refined result
print("Final refined result:", refined_result.generated_summary)
```
---

## SelfConsistency

A DSPy module that implements self-consistency sampling by generating multiple outputs and selecting the most representative one based on Levenshtein distance.

### Overview

The `SelfConsistency` module improves the reliability of language model outputs by:
1. Generating multiple responses
2. Computing normalized Levenshtein distances between all pairs
3. Selecting the response that is most similar to all other responses

This approach helps filter out potential outliers and select the most "average" or representative response from multiple generations.

### Installation

```bash
pip install python-Levenshtein  # For string distance calculations
```

### Usage

```python
# Define your signature
class TextGeneration(dspy.Signature):
    """Simple text generation signature."""
    context = dspy.InputField()
    response = dspy.OutputField()

# Create SelfConsistency module
generator = SelfConsistency(
    signature=TextGeneration,
    generations_number=5,  # Number of generations to produce
    cooldown_between_generations_sec=0  # Optional cooldown between generations
)

# Use the module
result = generator(
    context="What are the key features of Python?"
)
print(result.response)
```

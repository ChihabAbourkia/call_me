*This project has been created as part of the 42 curriculum by kaboruki.*

# Call Me Maybe

Introduction to function calling in LLMs — translating natural language into structured function calls using constrained decoding.

## Description

This project implements a function calling system that takes natural language prompts and converts them into structured JSON function calls. Given a prompt like "What is the sum of 2 and 3?", the system outputs the correct function name (`fn_add_numbers`) and arguments (`{"a": 2.0, "b": 3.0}`) instead of answering directly.

The key technique used is **constrained decoding**, which modifies the LLM's logits at each generation step to guarantee valid JSON output and correct schema compliance, even with a small 0.6B parameter model.

## Instructions

### Prerequisites

- Python 3.13+
- `uv` package manager

### Installation

```bash
make install
```

### Usage

```bash
# Run with default input/output paths
make run

# Run with custom paths
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Linting

```bash
make lint
make lint-strict  # optional, stricter checking
```

## Algorithm Explanation

### Constrained Decoding Approach

The system generates JSON output token-by-token using constrained decoding to ensure 100% valid structure:

1. **Function Name Selection**: At each step, every token is decoded to a string. If appending that string to the current partial name would make it impossible for any valid function name to still be a prefix, the token is masked with `-inf`. This guarantees the generated name matches exactly one function.

2. **String Values**: Generated token-by-token until a closing double-quote is encountered. No constraints needed beyond valid string generation.

3. **Number Values**: Only tokens that keep the accumulated value as a valid number (integer or float) are allowed. Tokens containing `,` or `}` that aren't exactly those delimiters are masked. The `,` and `}` tokens are only allowed when a valid number has already been generated.

4. **JSON Structure**: The JSON structure is built incrementally by the main loop — the system prompt, function name, and parameters are concatenated into the context, ensuring the model always sees valid partial JSON.

### Why This Works

Small language models (0.6B parameters) are unreliable at generating structured output from prompts alone (~30% success rate). By intervening at the logits level before each token selection, we guarantee:
- 100% valid JSON syntax
- Correct schema compliance (right keys, right types)
- Proper function name selection

## Design Decisions

- **Incremental Context Building**: Rather than generating the entire JSON at once, the system builds the output piece by piece, giving the model maximum context at each step.
- **Prefix-Based Masking**: For function names and boolean values, tokens are masked based on prefix matching against valid options.
- **Type-Aware Number Generation**: Number generation validates that the accumulated string remains a parseable float at each step.
- **Pydantic Validation**: Input files are validated using Pydantic models to catch malformed data early.

## Performance Analysis

- **Accuracy**: Near-perfect function selection and argument extraction due to constrained decoding eliminating invalid outputs.
- **Speed**: Each prompt is processed sequentially through the model; typical processing time is reasonable for the 0.6B model.
- **Reliability**: 100% valid JSON output guaranteed by construction — every output is parseable and schema-compliant.

## Challenges Faced

- **Tokenizer Complexity**: Tokenizer outputs don't always align with clean string boundaries (leading spaces, subword splits), requiring careful handling when decoding tokens to strings.
- **Number Generation**: Ensuring the model generates valid numbers required careful masking of tokens that would create invalid numeric strings.
- **Context Management**: Building the JSON incrementally while keeping the context valid for the model required precise string concatenation.

## Testing Strategy

1. Verify output JSON is valid and parseable
2. Check that function names match the definitions exactly
3. Validate parameter types match the schema (numbers as floats, strings as strings)
4. Test with various prompt types: simple addition, string operations, regex substitution
5. Run `make lint` to ensure code quality

## Example Usage

```bash
# Default run
make run

# Check output
cat data/output/function_calling_results.json
```

Expected output format:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Resources

### References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/) — Model loading and tokenizer API
- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — Model specifications and capabilities
- [Constrained Decoding / Grammar-Sampling](https://blog.vllm.ai/2024/09/05/sampling.html) — Overview of guided generation techniques
- [Pydantic V2 Documentation](https://docs.pydantic.dev/latest/) — Data validation and settings management
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) — Conceptual background on LLM function calling

### AI Usage

AI tools were used for the following tasks:

- **Code scaffolding**: Initial structure of the `constrained_decoding.py` module and the token-by-token generation loop was drafted with AI assistance, then reviewed and refined manually.
- **Pydantic model design**: AI helped draft the `json_validator.py` models based on the input JSON schemas.
- **Documentation**: AI assisted in writing docstrings and parts of this README (algorithm explanation, design decisions).
- **Debugging**: AI was used to reason about edge cases in number generation and tokenizer behavior.

All AI-generated code was reviewed, tested, and modified to fit the project's constraints and coding standards.

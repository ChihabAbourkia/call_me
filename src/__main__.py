import json
from pathlib import Path

from .parser import argparser
from .loader import prompts_loader, function_loader
from .constrained_decoding import (
    system_prompt_builder,
    generate_function_name,
    generate_string_value,
    generate_number_value,
    generate_bool_value,
)
from llm_sdk import Small_LLM_Model


def _get_function_by_name(functions: list, name: str) -> object:
    """Retrieve a function definition by name.

    Args:
        functions: List of Functions pydantic models.
        name: Function name to look up.

    Returns:
        The matching Functions model.

    Raises:
        ValueError: If function name is not found.
    """
    for f in functions:
        if f.name == name:
            return f
    raise ValueError(f"Function '{name}' not found in definitions.")


def main() -> None:
    """Main entry point: loads data, runs constrained decoding, writes output."""
    parser = argparser()

    functions = function_loader(parser.functions_definition)
    if not functions:
        raise RuntimeError("No function definitions found.")
    prompts = prompts_loader(parser.input)
    if not prompts:
        raise RuntimeError("No input prompts found.")

    prompt_system = system_prompt_builder(functions)
    function_names = [f.name for f in functions]

    try:
        model = Small_LLM_Model(parser.model)
    except OSError:
        raise RuntimeError(f"Model {parser.model} not found or failed to download.")

    results = []
    for p in prompts:
        prompt = p.prompt
        context = (
            f"{prompt_system}\n"
            f'{{"prompt": "{prompt}",'
            f'"name": "'
        )
        print(f"Processing: {prompt}")

        func_name = generate_function_name(model, context, function_names)
        func_def = _get_function_by_name(functions, func_name)

        context += f'{func_name}", "parameters": {{'

        parameters: dict = {}
        param_items = list(func_def.parameters.items())
        for i, (param_name, param_info) in enumerate(param_items):
            context += f'"{param_name}": '
            param_type = param_info.type

            if param_type == "string":
                context += '"'
                val = generate_string_value(model, context)
                context += val + '"'
                parameters[param_name] = val
            elif param_type == "number":
                val = generate_number_value(model, context)
                context += val
                parameters[param_name] = float(val)
            elif param_type == "integer":
                val = generate_number_value(model, context)
                context += val
                parameters[param_name] = int(float(val))
            elif param_type == "boolean":
                val = generate_bool_value(model, context)
                context += str(val).lower()
                parameters[param_name] = val
            else:
                val = generate_string_value(model, context)
                context += val + '"'
                parameters[param_name] = val

            if i < len(param_items) - 1:
                context += ", "

        context += "}}"

        results.append({
            "prompt": prompt,
            "name": func_name,
            "parameters": parameters,
        })
        print(f"  -> {func_name}({parameters})")

    output_path = Path(parser.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()

from typing import Any, List, Optional


def system_prompt_builder(functions: list) -> str:
    """Build a system prompt listing available functions.

    Args:
        functions: List of Functions pydantic models.

    Returns:
        Formatted system prompt string.
    """
    lines = [
        "You are a natural language to function call system.",
        "Given this function registry:",
    ]
    for f in functions:
        param_parts = []
        for p, info in f.parameters.items():
            param_parts.append(f'"{p}": {{"type": "{info.type}"}}')
        params_str = ", ".join(param_parts)
        lines.append(
            f'{{"name": "{f.name}", "description": "{f.description}", '
            f'"parameters": {{{params_str}}}}}'
        )
    lines.append(
        "Chose the appropriate function "
        "and its parameters based on the user input."
    )
    return "\n".join(lines)


def generate_function_name(
    model: Any,
    context: str,
    function_names: List[str],
    max_tokens: int = 30,
) -> str:
    """Generate a function name using constrained decoding.

    At each step, every token is decoded to a string. If appending that
    string to the current partial name would make it impossible for any
    valid function name to still be a prefix, the token is masked with -inf.

    Args:
        model: Object with encode(), decode(), get_logits_from_input_ids().
        context: The full prompt context ending with '"name": "'.
        function_names: List of valid function name strings.
        max_tokens: Safety limit on number of tokens.

    Returns:
        The generated function name (without quotes).
    """
    generated = ""
    found_valid = False

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + generated)[0].tolist()
        )

        for token_id in range(len(logits)):
            token_str = model.decode([token_id])
            if token_str is None:
                token_str = ""
            combined = generated + token_str
            if not any(fn.startswith(combined) for fn in function_names):
                logits[token_id] = float("-inf")

        best_id = _argmax(logits)
        best_str = model.decode([best_id])
        if best_str is None:
            best_str = ""

        if best_str == '"' and found_valid:
            break

        generated += best_str

        if generated in function_names:
            found_valid = True

    return generated


def generate_string_value(
    model: Any,
    context: str,
    max_tokens: int = 100,
) -> str:
    """Generate a string parameter value token by token.

    Generates until a closing double-quote is encountered.

    Args:
        model: Object with encode(), decode(), get_logits_from_input_ids().
        context: Prompt context ending with an opening '"'.
        max_tokens: Safety limit.

    Returns:
        The generated string value (without quotes).
    """
    value = ""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + value)[0].tolist()
        )
        best_id = _argmax(logits)
        best_str = model.decode([best_id])
        if best_str is None:
            best_str = ""
        if '"' in best_str:
            break
        value += best_str
    return value.strip()


def _is_float(s: str) -> bool:
    """Check if string is a valid float representation."""
    try:
        float(s)
        return True
    except (ValueError, OverflowError):
        return False


def generate_number_value(
    model: Any,
    context: str,
    max_tokens: int = 30,
) -> str:
    """Generate a numeric parameter value (float or int) with constrained decoding.

    Only allows tokens that keep the accumulated value as a valid number.
    Stops at ',' or '}' which are the JSON delimiters after a value.

    Args:
        model: Object with encode(), decode(), get_logits_from_input_ids().
        context: Prompt context ending just before the number.
        max_tokens: Safety limit.

    Returns:
        The generated number as a string (e.g. "2.0" or "42").
    """
    number = ""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + number)[0].tolist()
        )
        for token_id in range(len(logits)):
            token_str = model.decode([token_id])
            if token_str is None:
                token_str = ""
            # If token contains , or } but is not exactly , or }, mask it
            if ("," in token_str and token_str != ",") or (
                "}" in token_str and token_str != "}"
            ):
                logits[token_id] = float("-inf")
            # If token is , or } but we don't have a valid number yet, mask it
            elif (token_str == "," or token_str == "}") and not _is_float(
                number
            ):
                logits[token_id] = float("-inf")
            # For any other token, check if number+token is still valid
            elif (
                token_str != ","
                and token_str != "}"
                and not _is_float(number + token_str)
            ):
                logits[token_id] = float("-inf")

        best_id = _argmax(logits)
        best_str = model.decode([best_id])
        if best_str is None:
            best_str = ""
        if best_str == "," or best_str == "}":
            break
        number += best_str
    return number


def generate_bool_value(
    model: Any,
    context: str,
    max_tokens: int = 10,
) -> bool:
    """Generate a boolean parameter value with constrained decoding.

    Only allows tokens that are prefixes of 'true' or 'false'.

    Args:
        model: Object with encode(), decode(), get_logits_from_input_ids().
        context: Prompt context ending just before the boolean value.
        max_tokens: Safety limit.

    Returns:
        Boolean value (True or False).
    """
    output = ""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(
            model.encode(context + output)[0].tolist()
        )
        for token_id in range(len(logits)):
            token_str = model.decode([token_id])
            if token_str is None:
                token_str = ""
            if not any(
                s.startswith(output + token_str) for s in ("true", "false")
            ):
                logits[token_id] = float("-inf")

        best_id = _argmax(logits)
        best_str = model.decode([best_id])
        if best_str is None:
            best_str = ""
        combined = output + best_str
        if "true" in combined.lower():
            return True
        if "false" in combined.lower():
            return False
        output += best_str
    return False


def _argmax(logits: List[float]) -> int:
    """Return the index of the maximum value in a list of floats."""
    best_idx = 0
    best_val = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > best_val:
            best_val = logits[i]
            best_idx = i
    return best_idx


def format_json(text: str) -> Optional[str]:
    """Extract the first complete JSON object from text.

    Finds the first '{' and matches braces to find the end.

    Args:
        text: Text that may contain a JSON object.

    Returns:
        The JSON substring, or None if no complete JSON found.
    """
    start = text.find("{")
    if start == -1:
        return None
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        if text[i] == "}":
            brace_count -= 1
        if brace_count == 0:
            return text[start : i + 1]
    return None

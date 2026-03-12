"""Reverse skill — reverses a string. Used for testing."""


async def execute(input_data: dict) -> dict:
    """Reverse the input text."""
    return {"text": input_data["text"][::-1]}

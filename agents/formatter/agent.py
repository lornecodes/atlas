"""Formatter agent — wraps content with style markers."""

from atlas.runtime.base import AgentBase


class FormatterAgent(AgentBase):
    async def execute(self, input_data: dict) -> dict:
        content = input_data["content"]
        style = input_data["style"]

        if style == "markdown":
            formatted = f"# Output\n\n{content}"
        elif style == "uppercase":
            formatted = content.upper()
        elif style == "bullet":
            lines = content.split(". ")
            formatted = "\n".join(f"- {line.strip()}" for line in lines if line.strip())
        else:
            formatted = f"[{style}] {content}"

        return {
            "formatted": formatted,
            "style_applied": style,
        }

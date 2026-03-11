"""Summarizer agent — trivial word-boundary truncation."""

from atlas.runtime.base import AgentBase


class SummarizerAgent(AgentBase):
    async def execute(self, input_data: dict) -> dict:
        text = input_data["text"]
        max_length = input_data.get("max_length", 200)

        words = text.split()
        summary_words = []
        char_count = 0

        for word in words:
            if char_count + len(word) + (1 if summary_words else 0) > max_length:
                break
            summary_words.append(word)
            char_count += len(word) + (1 if len(summary_words) > 1 else 0)

        summary = " ".join(summary_words)
        return {
            "summary": summary,
            "token_count": len(summary_words),
        }

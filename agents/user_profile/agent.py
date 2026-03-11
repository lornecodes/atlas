"""User profile agent — processes nested user data."""

from atlas.runtime.base import AgentBase


class UserProfileAgent(AgentBase):
    async def execute(self, input_data: dict) -> dict:
        user = input_data["user"]
        action = input_data["action"]
        address = user.get("address", {})

        return {
            "status": f"{action} completed",
            "user_name": user["name"],
            "city": address.get("city", "unknown"),
        }

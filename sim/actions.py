"""Action schema: pydantic models for validation and JSON schema export for guided decoding."""

from __future__ import annotations

import copy
import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _Action(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MoveArmy(_Action):
    type: Literal["move_army"]
    from_province: int
    to_province: int
    troops: int = Field(ge=1)


class Attack(_Action):
    type: Literal["attack"]
    from_province: int
    to_province: int
    troops: int = Field(ge=1)


class Build(_Action):
    type: Literal["build"]
    province: int
    kind: Literal["troops", "market", "fort"]


class TradeOffer(_Action):
    type: Literal["trade_offer"]
    to_faction: int
    gold: int = Field(ge=1)


class ProposeTreaty(_Action):
    type: Literal["propose_treaty"]
    to_faction: int
    treaty: Literal["peace", "alliance", "trade_pact"]


class RespondTreaty(_Action):
    type: Literal["respond_treaty"]
    proposal_id: int
    accept: bool


class Pass(_Action):
    type: Literal["pass"]


Action = Annotated[
    MoveArmy | Attack | Build | TradeOffer | ProposeTreaty | RespondTreaty | Pass,
    Field(discriminator="type"),
]
ACTION_TYPES: tuple[str, ...] = (
    "move_army",
    "attack",
    "build",
    "trade_offer",
    "propose_treaty",
    "respond_treaty",
    "pass",
)


class ActionResponse(BaseModel):
    """One faction's output for a turn: a structured action plus a free-text public statement."""

    model_config = ConfigDict(extra="forbid")

    action: Action
    diplomatic_message: str


def response_json_schema(max_message_chars: int) -> dict[str, Any]:
    """JSON schema for guided decoding, with the message length cap taken from config."""
    schema = copy.deepcopy(ActionResponse.model_json_schema())
    schema["properties"]["diplomatic_message"]["maxLength"] = max_message_chars
    return schema


class ParseResult(BaseModel):
    response: ActionResponse | None
    error: str | None


def parse_response(text: str, max_message_chars: int) -> ParseResult:
    """Parse raw model output. Any failure yields response=None (resolved as pass, counted)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return ParseResult(response=None, error=f"invalid JSON: {e}")
    try:
        response = ActionResponse.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(map(str, first["loc"])) or "<root>"
        return ParseResult(
            response=None,
            error=f"schema violation: {e.error_count()} errors (first: {first['type']} at {where})",
        )
    if len(response.diplomatic_message) > max_message_chars:
        return ParseResult(response=None, error="diplomatic_message too long")
    return ParseResult(response=response, error=None)

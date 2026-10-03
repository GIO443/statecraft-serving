from __future__ import annotations

import json

import pytest

from sim.actions import ACTION_TYPES, parse_response, response_json_schema

MAX_CHARS = 50

VALID_ACTIONS = [
    {"type": "move_army", "from_province": 0, "to_province": 1, "troops": 5},
    {"type": "attack", "from_province": 0, "to_province": 2, "troops": 5},
    {"type": "build", "province": 0, "kind": "fort"},
    {"type": "trade_offer", "to_faction": 1, "gold": 10},
    {"type": "propose_treaty", "to_faction": 1, "treaty": "alliance"},
    {"type": "respond_treaty", "proposal_id": 3, "accept": True},
    {"type": "pass"},
]


def _wrap(action: dict, message: str = "hello") -> str:
    return json.dumps({"action": action, "diplomatic_message": message})


def test_valid_actions_cover_all_types() -> None:
    assert sorted(a["type"] for a in VALID_ACTIONS) == sorted(ACTION_TYPES)


@pytest.mark.parametrize("action", VALID_ACTIONS, ids=lambda a: a["type"])
def test_parse_valid(action: dict) -> None:
    result = parse_response(_wrap(action), MAX_CHARS)
    assert result.error is None
    assert result.response is not None
    assert result.response.action.model_dump() == action


@pytest.mark.parametrize(
    ("text", "error_prefix"),
    [
        ("not json", "invalid JSON"),
        ('{"action": {"type": "pass"}', "invalid JSON"),
        (_wrap({"type": "nuke"}), "schema violation"),
        (
            _wrap({"type": "attack", "from_province": 0, "to_province": 1, "troops": 0}),
            "schema violation",
        ),
        (_wrap({"type": "pass", "extra": 1}), "schema violation"),
        (json.dumps({"action": {"type": "pass"}}), "schema violation"),
        (_wrap({"type": "pass"}, "x" * (MAX_CHARS + 1)), "diplomatic_message too long"),
    ],
)
def test_parse_invalid(text: str, error_prefix: str) -> None:
    result = parse_response(text, MAX_CHARS)
    assert result.response is None
    assert result.error is not None and result.error.startswith(error_prefix)


def test_schema_error_names_first_failure() -> None:
    """The two failure modes seen from the unguided 1.5B model must be identifiable."""
    missing = parse_response(json.dumps({"action": {"type": "pass"}}), MAX_CHARS)
    assert missing.error is not None and "missing at diplomatic_message" in missing.error
    flat = parse_response(json.dumps({"type": "pass", "diplomatic_message": ""}), MAX_CHARS)
    assert flat.error is not None and "action" in flat.error


def test_message_at_cap_is_valid() -> None:
    assert parse_response(_wrap({"type": "pass"}, "x" * MAX_CHARS), MAX_CHARS).response


def test_json_schema() -> None:
    schema = response_json_schema(MAX_CHARS)
    json.dumps(schema)  # must be serializable for the guided decoding request
    assert schema["properties"]["diplomatic_message"]["maxLength"] == MAX_CHARS
    assert set(schema["required"]) == {"action", "diplomatic_message"}
    mapping = schema["properties"]["action"]["discriminator"]["mapping"]
    assert set(mapping) == set(ACTION_TYPES)
    # Exporting must not mutate the cached pydantic schema.
    assert response_json_schema(MAX_CHARS + 1)["properties"]["diplomatic_message"]["maxLength"] == (
        MAX_CHARS + 1
    )

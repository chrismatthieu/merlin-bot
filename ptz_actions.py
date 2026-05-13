"""PTZ preset names for HTTP/MCP and brain (motion implemented in voice.py via uvc-util)."""

PTZ_ACTIONS = frozenset(
    {"look_left", "look_right", "look_up", "look_down", "look_center", "look_around"}
)

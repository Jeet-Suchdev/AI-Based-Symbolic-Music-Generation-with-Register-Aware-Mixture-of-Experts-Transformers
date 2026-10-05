from enum import IntEnum

class Register(IntEnum):
    SPECIAL = 0
    LOW = 1
    MID = 2
    HIGH = 3

def pitch_to_register(pitch: int, low_max=47, mid_max=71) -> Register:
    if pitch < 0 or pitch > 127:
        return Register.SPECIAL
    if pitch <= low_max:
        return Register.LOW
    if pitch <= mid_max:
        return Register.MID
    return Register.HIGH

def token_string_to_register(token: str, low_max=47, mid_max=71) -> Register:
    if token.startswith("Pitch_"):
        try:
            return pitch_to_register(int(token.rsplit("_", 1)[1]), low_max, mid_max)
        except ValueError:
            return Register.SPECIAL
    if token.startswith("PitchDrum_"):
        return Register.SPECIAL
    return Register.SPECIAL

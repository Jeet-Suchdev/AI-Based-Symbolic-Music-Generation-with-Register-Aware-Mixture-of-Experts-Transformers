from musicgen.data.register import Register, pitch_to_register

def test_register_boundaries():
    assert pitch_to_register(0) == Register.LOW
    assert pitch_to_register(47) == Register.LOW
    assert pitch_to_register(48) == Register.MID
    assert pitch_to_register(71) == Register.MID
    assert pitch_to_register(72) == Register.HIGH
    assert pitch_to_register(127) == Register.HIGH

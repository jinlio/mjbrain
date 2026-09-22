def test_packages_import():
    import brain  # noqa: F401

def test_random_bot():
    from eval import RandomBot

    b = RandomBot()
    assert b.react([], 0, ["dahai:1p", "pass"]) in {"dahai:1p", "pass"}

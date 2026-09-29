from scripts.get_data import members


def test_members_selects_only_the_named_scene():
    names = ["tandt/truck/", "tandt/truck/a.jpg", "tandt/train/b.jpg", "tandt/truck2/c.jpg"]
    assert members(names, "tandt/truck") == ["tandt/truck/", "tandt/truck/a.jpg"]

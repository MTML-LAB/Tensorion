from tensorion.config_resolvers import append_to_list


def test_append_to_list_keeps_scalar_string_as_single_item():
    result = append_to_list(["base"], "mytag")
    assert result == ["base", "mytag"]


def test_append_to_list_skips_empty_string_items():
    result = append_to_list(["base"], ["", "extra"])
    assert result == ["base", "extra"]

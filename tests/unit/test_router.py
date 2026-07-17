from src.router.rule_router import route, RoutePath


def test_simple_query_routes_fast():
    assert route("What's the weather today?").path == RoutePath.FAST


def test_scheduling_query_routes_reasoning():
    decision = route(
        "Schedule three tasks across two agents avoiding conflicts, "
        "prioritizing the deadline-sensitive one first."
    )
    assert decision.path == RoutePath.REASONING


def test_multi_tool_request_routes_reasoning():
    decision = route(
        "Check the calendar and send an email",
        requested_tools=["calendar_read", "email_send", "contact_lookup"],
    )
    assert decision.path == RoutePath.REASONING


def test_long_request_routes_reasoning():
    long_text = " ".join(["word"] * 45)
    assert route(long_text).path == RoutePath.REASONING


def test_short_conversational_routes_fast():
    assert route("Thanks, that's helpful!").path == RoutePath.FAST


def test_fixture_examples_route_as_expected(sample_tasks):
    for msg in sample_tasks["fast_path_examples"]:
        assert route(msg).path == RoutePath.FAST, f"Expected FAST for: {msg!r}"
    for msg in sample_tasks["reasoning_path_examples"]:
        assert route(msg).path == RoutePath.REASONING, f"Expected REASONING for: {msg!r}"

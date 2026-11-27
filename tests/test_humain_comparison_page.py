from html.parser import HTMLParser
from pathlib import Path


class _PageContract(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tabs: set[str] = set()
        self.has_tablist = False
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.has_tablist |= attributes.get("role") == "tablist"
        if tag == "button" and attributes.get("data-bab"):
            self.tabs.add(attributes["data-bab"] or "")

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def test_comparison_page_exposes_the_three_bab_pilot_and_method_caveat():
    """Breaks if the local report drops a bāb, a model, or its validity warning."""
    page = Path(__file__).parents[1] / "outputs" / "humain_falcon_comparison.html"
    contract = _PageContract()
    contract.feed(page.read_text(encoding="utf-8"))
    visible_text = " ".join(contract.text)

    assert contract.has_tablist
    assert contract.tabs == {"nawawi", "cayni", "mazandarani"}
    assert "HUMAIN M3" in visible_text
    assert "Falcon-H1-Arabic-34B-Instruct" in visible_text
    assert "not a controlled benchmark" in visible_text

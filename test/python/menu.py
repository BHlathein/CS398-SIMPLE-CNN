from browser import document, html

class RulesModal:
    def __init__(self):
        self.init_rules()

    def toggle_rules(self, ev):
        modal = document["rules-modal"]
        current_display = modal.style.display
        modal.style.display = "block" if (current_display == "none" or not current_display) else "none"

    def close_rules(self, ev):
        document["rules-modal"].style.display = "none"

    def click_outside(self, ev):
        modal = document["rules-modal"]
        if ev.target == modal:
            modal.style.display = "none"

    def handle_keypress(self, ev):
        if ev.key == "Escape":
            document["rules-modal"].style.display = "none"

    def init_rules(self):
        document["rules-btn"].bind("click", self.toggle_rules)
        document["close-rules"].bind("click", self.close_rules)
        document["rules-modal"].bind("click", self.click_outside)
        document.bind("keydown", self.handle_keypress)
        document["rules-modal"].style.display = "none"

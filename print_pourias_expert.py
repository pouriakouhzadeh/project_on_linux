import os
import sys
import time

class BlinkingTextPrinter:
    BOLD_GREEN = '\033[1;32m'
    RESET = '\033[0m'
    BLINK = '\033[5m'
    NO_BLINK = '\033[25m'

    letters = {
        'A': [
            "  $  ",
            " $ $ ",
            "$$$$$",
            "$   $",
            "$   $"
        ],
        'E': [
            "$$$$$",
            "$    ",
            "$$$$$",
            "$    ",
            "$$$$$"
        ],
        'I': [
            "$$$$$",
            "  $  ",
            "  $  ",
            "  $  ",
            "$$$$$"
        ],
        'O': [
            " $$$ ",
            "$   $",
            "$   $",
            "$   $",
            " $$$ "
        ],
        'P': [
            "$$$$ ",
            "$   $",
            "$$$$ ",
            "$    ",
            "$    "
        ],
        'R': [
            "$$$$ ",
            "$   $",
            "$$$$ ",
            "$  $ ",
            "$   $"
        ],
        'S': [
            " $$$$",
            "$    ",
            " $$$ ",
            "    $",
            "$$$$ "
        ],
        'T': [
            "$$$$$",
            "  $  ",
            "  $  ",
            "  $  ",
            "  $  "
        ],
        'U': [
            "$   $",
            "$   $",
            "$   $",
            "$   $",
            " $$$ "
        ],
        'X': [
            "$   $",
            " $ $ ",
            "  $  ",
            " $ $ ",
            "$   $"
        ],
        "'": [
            "  $  ",
            "  $  ",
            "     ",
            "     ",
            "     "
        ],
        ' ': [
            "     ",
            "     ",
            "     ",
            "     ",
            "     "
        ]
    }

    def __init__(self, text):
        self.text = text.upper()

    def print_text(self, duration=5, interval=0.5):
        end_time = time.time() + duration
        while time.time() < end_time:
            self._print_with_effect(self.BLINK)
            time.sleep(interval)
            self._print_with_effect(self.NO_BLINK)
            time.sleep(interval)

    def _print_with_effect(self, effect):
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
        for row in range(5):
            for char in self.text:
                print(f"{self.BOLD_GREEN}{effect}{self.letters[char][row]}{self.RESET}", end="  ")
            print()
        sys.stdout.flush()


# استفاده از کلاس
# blinking_printer = BlinkingTextPrinter("POURIA'S EXPERT")
# blinking_printer.print_text(duration=10, interval=0.5)

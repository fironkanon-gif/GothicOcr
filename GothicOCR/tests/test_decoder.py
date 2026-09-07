from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.text_decoder import TextDecoder

decoder = TextDecoder(Path(__file__).resolve().parents[1] / "data/labels.json")
assert len(decoder.labels) == 25
print("PASS: labels and decoder loaded")

# Run this only in an environment that has tflite-runtime or TensorFlow installed.
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1] / "models/gothic_ocr.tflite"

def get_interpreter():
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
            return Interpreter
        except ImportError as exc:
            raise RuntimeError("ثبتي tflite-runtime أو TensorFlow لفحص النموذج محليًا.") from exc

Interpreter = get_interpreter()
interpreter = Interpreter(model_path=str(MODEL))
interpreter.allocate_tensors()
print("INPUT:", interpreter.get_input_details())
print("OUTPUT:", interpreter.get_output_details())

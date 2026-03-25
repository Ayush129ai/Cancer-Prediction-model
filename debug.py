import sys
import traceback

print("Testing import...")
try:
    import flask_ui.app
    print("Success")
except Exception as e:
    print("FAILED")
    with open("err.log", "w") as f:
        f.write(traceback.format_exc())
    traceback.print_exc()

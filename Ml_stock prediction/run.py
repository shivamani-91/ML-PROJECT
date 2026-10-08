#!/usr/bin/env python3
"""
AI Stock Price Predictor – LSTM & GRU
Quick Launcher Script
"""

import os
import sys
import webbrowser
import threading
import time

def main():
    print("=" * 70)
    print("   AI Stock Price Predictor – LSTM & GRU")
    print("   B.Tech Computer Science & Engineering - Machine Learning Project")
    print("=" * 70)
    print("[*] Checking environment...")

    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    if "KERAS_BACKEND" not in os.environ:
        try:
            import tensorflow
            os.environ["KERAS_BACKEND"] = "tensorflow"
        except ImportError:
            os.environ["KERAS_BACKEND"] = "torch"

    try:
        import flask
        import keras
        import sklearn
        import pandas
        import numpy
        print("[+] Core libraries detected successfully:")
        print(f"    - Flask: {flask.__version__}")
        print(f"    - Keras: {keras.__version__}")
        print(f"    - Scikit-Learn: {sklearn.__version__}")
        print(f"    - Pandas: {pandas.__version__}")
        print(f"    - NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"[!] Missing dependency: {e}")
        print("[!] Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Launch browser after 1.5 seconds
    def open_browser():
        time.sleep(1.5)
        url = "http://127.0.0.1:5000"
        print(f"\n[+] Opening web application in default browser: {url}")
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    from backend.app import app, load_initial_dataset
    load_initial_dataset()
    
    print("\n[+] Server is running at http://127.0.0.1:5000")
    print("[+] Press CTRL+C to stop the server.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

if __name__ == "__main__":
    main()

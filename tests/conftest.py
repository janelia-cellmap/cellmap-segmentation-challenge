import os

# Force non-interactive matplotlib backend (Windows CI Tk is unusable).
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

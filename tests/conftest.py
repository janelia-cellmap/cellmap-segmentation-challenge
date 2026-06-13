import matplotlib

# Use the non-interactive Agg backend so that tests importing matplotlib.pyplot
# work on headless/Windows CI runners where Tkinter/Tcl-Tk is unavailable.
matplotlib.use("Agg")

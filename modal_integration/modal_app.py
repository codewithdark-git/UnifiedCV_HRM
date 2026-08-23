"""
Shared Modal App Instance

All modal functions across modules should import this app to avoid
circular imports and ensure all functions are registered on the same App.
"""

from modal import App

# Single shared app instance
app = App("ihrm")
"""
Main entry point for Streamlit Cloud deployment
Redirects to the clean app interface
"""

import streamlit as st
import subprocess
import sys
import os

# Redirect to the main app
exec(open('app_clean.py').read())
import os

# Gunicorn configuration for production
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = 2  # Adjust based on Render's resources (Free tier should be 1-2)
threads = 4
timeout = 120  # ML analysis might take some time
accesslog = "-"
errorlog = "-"
loglevel = "info"

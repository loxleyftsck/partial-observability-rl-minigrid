"""
Simple HTTP server for dashboard
"""
import http.server
import socketserver
import webbrowser
import os

PORT = 8080

os.chdir(os.path.dirname(__file__))

Handler = http.server.SimpleHTTPRequestHandler

print(f" Starting dashboard server at http://localhost:{PORT}")
print("Press Ctrl+C to stop")

# Open browser automatically
webbrowser.open(f"http://localhost:{PORT}")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()

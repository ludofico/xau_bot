#!/usr/bin/env python3
"""
cTrader OAuth Token Acquisition Script.
Run this once to get your access_token and refresh_token.
"""
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests
import json

# Your cTrader app credentials
CLIENT_ID = "20387_LuSzT1nOeFSC5FwhySR048rLYPgcAt3e3zA8JwFC9iHNk9WPA5"
CLIENT_SECRET = "9pvGNR53AxgyftVlkSKF7BowvuxsVF32f4Hebi4BYCd1ij5R7h"
REDIRECT_URI = "http://localhost:8080/callback"

# cTrader OAuth endpoints
AUTH_URL = "https://openapi.ctrader.com/apps/auth"
TOKEN_URL = "https://openapi.ctrader.com/apps/token"


class OAuthHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback."""
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == "/callback":
            params = parse_qs(parsed.query)
            
            if "code" in params:
                code = params["code"][0]
                print(f"\n✅ Authorization code received!")
                
                # Exchange code for tokens
                tokens = self.exchange_code(code)
                
                if tokens:
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    
                    html = f"""
                    <html>
                    <head><title>Success!</title></head>
                    <body style="font-family: sans-serif; padding: 40px;">
                        <h1>✅ Authorization Successful!</h1>
                        <p>Copy these values to your <b>config/aggressive.yaml</b>:</p>
                        <pre style="background: #f0f0f0; padding: 20px; border-radius: 8px;">
ctrader:
  enabled: true
  client_id: "{CLIENT_ID}"
  client_secret: "{CLIENT_SECRET}"
  account_id: "{tokens.get('accountId', 'YOUR_ACCOUNT_ID')}"
  access_token: "{tokens['accessToken']}"
  refresh_token: "{tokens['refreshToken']}"
                        </pre>
                        <p>You can close this window now.</p>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode())
                    
                    # Print to console too
                    print("\n" + "=" * 60)
                    print("COPY THESE VALUES TO config/aggressive.yaml:")
                    print("=" * 60)
                    print(f'  access_token: "{tokens["accessToken"]}"')
                    print(f'  refresh_token: "{tokens["refreshToken"]}"')
                    print("=" * 60)
                    
                    # Stop server
                    import threading
                    threading.Thread(target=self.server.shutdown).start()
                else:
                    self.send_error(500, "Token exchange failed")
            else:
                self.send_error(400, "No authorization code received")
        else:
            self.send_error(404)
    
    def exchange_code(self, code: str) -> dict:
        """Exchange authorization code for access token."""
        try:
            response = requests.post(TOKEN_URL, data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET
            })
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Token exchange error: {response.text}")
                return None
        except Exception as e:
            print(f"Token exchange error: {e}")
            return None
    
    def log_message(self, format, *args):
        pass  # Suppress logs


def main():
    print("=" * 60)
    print("cTrader OAuth Authorization")
    print("=" * 60)
    
    # Build auth URL
    auth_url = f"{AUTH_URL}?client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=accounts"
    
    print("\n1. Opening browser for authorization...")
    print(f"   If browser doesn't open, go to:\n   {auth_url}\n")
    
    # Open browser
    webbrowser.open(auth_url)
    
    print("2. Waiting for callback on http://localhost:8080/callback ...")
    print("   (Login to cTrader and authorize the app)\n")
    
    # Start callback server
    server = HTTPServer(("localhost", 8080), OAuthHandler)
    server.serve_forever()
    
    print("\n✅ Done! Update your config/aggressive.yaml with the tokens above.")


if __name__ == "__main__":
    main()

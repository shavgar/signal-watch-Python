from flask import Flask, jsonify
import requests

app = Flask(__name__)

ENDPOINTS = [
    "https://api.bybit.com/v5/market/tickers?category=spot",
    "https://api.bytick.com/v5/market/tickers?category=spot",
]

@app.route("/")
def test_bybit():
    results = []

    headers = {
        "User-Agent": "Signal-Watch/1.0"
    }

    for url in ENDPOINTS:
        try:
            r = requests.get(
                url,
                headers=headers,
                timeout=10
            )

            result = {
                "url": url,
                "status": r.status_code
            }

            if r.status_code == 200:
                data = r.json()

                tickers = data.get("result", {}).get("list", [])

                usdt_pairs = [
                    x for x in tickers
                    if x.get("symbol", "").endswith("USDT")
                ]

                result["success"] = True
                result["usdt_pairs"] = len(usdt_pairs)

            else:
                result["success"] = False
                result["response"] = r.text[:300]

            results.append(result)

        except Exception as e:
            results.append({
                "url": url,
                "success": False,
                "error": str(e)
            })

    return jsonify({
        "service": "Signal Watch Bybit Test",
        "results": results
    })


@app.route("/health")
def health():
    return "OK", 200

            "Detector error:",
            exc
        )

        return jsonify({

            "success":
                False,

            "error":
                str(
                    exc
                ),

        }), 500


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status":
            "OK"

    }), 200


# ============================================================
# TELEGRAM TEST
# ============================================================

@app.route("/telegram-test")
def telegram_test():

    try:

        send_telegram(

            "🧪 TELEGRAM TEST — SIGNAL WATCH V2\n\n"

            "Telegram: OK\n"

            "Bybit chart button: ready\n\n"

            "🚀 Pump / 🔻 Crash: enabled\n"

            "⚡ Early Pump: enabled\n"

            "🔄 U-Turn: enabled\n"

            "🪜 Staircase: enabled",

            button_url=
                bybit_spot_url(
                    "BTCUSDT"
                ),

            button_text=
                "📈 Open BTC on Bybit"
        )


        return jsonify({

            "success":
                True,

            "message":
                "Telegram test sent",

        })


    except Exception as exc:

        return jsonify({

            "success":
                False,

            "error":
                str(
                    exc
                ),

        }), 500


# ============================================================
# LOCAL START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )

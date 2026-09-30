import os
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from flask import Flask, jsonify


app = Flask(__name__)

# ============================================================
# SIGNAL WATCH CONFIGURATION
# ============================================================

BYBIT_ENDPOINTS = [
    "https://api.bybit.com/v5/market/tickers?category=spot",
    "https://api.bytick.com/v5/market/tickers?category=spot",
]

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

WINDOWS = {
    5: 10.0,    # 5 minutes: ±10%
    10: 15.0,   # 10 minutes: ±15%
    15: 20.0,   # 15 minutes: ±20%
}

REARM_MARGIN = 2.0

# Keep enough history for the longest window.
HISTORY_KEEP_SECONDS = 25 * 60

# Because cron runs roughly once per minute,
# allow up to ~90 seconds difference from ideal reference time.
REFERENCE_TOLERANCE_SECONDS = 90

UK_TIMEZONE = ZoneInfo("Europe/London")

# ============================================================
# IN-MEMORY STATE
# ============================================================

history = []

# Example:
# signal_state["ABCUSDT:5"] = "PUMP"
signal_state = {}

lock = threading.Lock()

first_run = True
last_success = None
last_market_count = 0


# ============================================================
# BYBIT
# ============================================================

def fetch_bybit_prices():
    headers = {
        "User-Agent": "Signal-Watch/1.0"
    }

    errors = []

    for url in BYBIT_ENDPOINTS:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=15
            )

            if response.status_code != 200:
                errors.append(
                    f"{url}: HTTP {response.status_code}"
                )
                continue

            data = response.json()

            if data.get("retCode") != 0:
                errors.append(
                    f"{url}: {data.get('retMsg')}"
                )
                continue

            prices = {}

            for ticker in data.get("result", {}).get("list", []):
                symbol = ticker.get("symbol", "")
                last_price = ticker.get("lastPrice")

                if not symbol.endswith("USDT"):
                    continue

                try:
                    price = float(last_price)
                except (TypeError, ValueError):
                    continue

                if price > 0:
                    prices[symbol] = price

            if len(prices) < 10:
                errors.append(
                    f"{url}: only {len(prices)} USDT pairs returned"
                )
                continue

            return prices, url

        except Exception as exc:
            errors.append(
                f"{url}: {str(exc)}"
            )

    raise RuntimeError(
        "All Bybit endpoints failed: "
        + " | ".join(errors)
    )


# ============================================================
# HISTORY
# ============================================================

def clean_old_history(now):
    cutoff = now - HISTORY_KEEP_SECONDS

    while history and history[0]["timestamp"] < cutoff:
        history.pop(0)


def save_snapshot(timestamp, prices):
    history.append({
        "timestamp": timestamp,
        "prices": prices
    })

    clean_old_history(timestamp)


def find_reference_snapshot(now, minutes):
    target = now - (minutes * 60)

    best = None

    # We deliberately use a snapshot at or BEFORE the target.
    # This prevents a "5 minute" comparison from accidentally
    # using only 4 minutes of history.
    for snapshot in history:
        ts = snapshot["timestamp"]

        if ts <= target:
            if best is None or ts > best["timestamp"]:
                best = snapshot

    if best is None:
        return None

    difference = target - best["timestamp"]

    if difference > REFERENCE_TOLERANCE_SECONDS:
        return None

    return best


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_signals(now, current_prices):
    alerts = []

    for minutes, threshold in WINDOWS.items():
        reference = find_reference_snapshot(
            now,
            minutes
        )

        if reference is None:
            continue

        reference_prices = reference["prices"]

        for symbol, current_price in current_prices.items():
            reference_price = reference_prices.get(symbol)

            if not reference_price or reference_price <= 0:
                continue

            change = (
                (current_price - reference_price)
                / reference_price
            ) * 100

            state_key = f"{symbol}:{minutes}"
            previous_state = signal_state.get(state_key)

            # ----------------------------
            # PUMP
            # ----------------------------

            if change >= threshold:
                if previous_state != "PUMP":
                    alerts.append({
                        "symbol": symbol,
                        "direction": "PUMP",
                        "window": minutes,
                        "change": change,
                        "reference_price": reference_price,
                        "current_price": current_price,
                    })

                signal_state[state_key] = "PUMP"

            # ----------------------------
            # CRASH
            # ----------------------------

            elif change <= -threshold:
                if previous_state != "CRASH":
                    alerts.append({
                        "symbol": symbol,
                        "direction": "CRASH",
                        "window": minutes,
                        "change": change,
                        "reference_price": reference_price,
                        "current_price": current_price,
                    })

                signal_state[state_key] = "CRASH"

            # ----------------------------
            # RE-ARM
            # ----------------------------

            elif abs(change) <= max(
                0,
                threshold - REARM_MARGIN
            ):
                signal_state.pop(
                    state_key,
                    None
                )

    return alerts


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing"
        )

    if not TELEGRAM_CHAT_ID:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing"
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=15
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Telegram error: "
            + response.text[:500]
        )


def format_price(price):
    if price >= 1000:
        return f"${price:,.2f}"

    if price >= 1:
        return f"${price:.4f}"

    if price >= 0.01:
        return f"${price:.6f}"

    return f"${price:.8f}"


def send_signal(alert):
    direction = alert["direction"]

    icon = (
        "🚀"
        if direction == "PUMP"
        else "🔻"
    )

    change = alert["change"]

    change_text = (
        f"+{change:.2f}%"
        if change >= 0
        else f"{change:.2f}%"
    )

    now_uk = datetime.now(
        UK_TIMEZONE
    ).strftime(
        "%d %b %Y %H:%M:%S"
    )

    message = (
        f"{icon} SIGNAL WATCH — {direction}\n\n"
        f"Coin: {alert['symbol']}\n"
        f"Move: {change_text}\n"
        f"Window: {alert['window']} minutes\n\n"
        f"Reference: "
        f"{format_price(alert['reference_price'])}\n"
        f"Current: "
        f"{format_price(alert['current_price'])}\n\n"
        f"Detected: {now_uk} UK"
    )

    send_telegram(message)


# ============================================================
# MAIN DETECTOR
# ============================================================

def run_detector():
    global first_run
    global last_success
    global last_market_count

    with lock:
        now = time.time()

        current_prices, endpoint = fetch_bybit_prices()

        alerts = detect_signals(
            now,
            current_prices
        )

        save_snapshot(
            now,
            current_prices
        )

        last_success = now
        last_market_count = len(current_prices)

        if first_run:
            send_telegram(
                "✅ SIGNAL WATCH ONLINE\n\n"
                f"Bybit connection: OK\n"
                f"USDT Spot markets: "
                f"{len(current_prices)}\n\n"
                "Thresholds:\n"
                "5 min: ±10%\n"
                "10 min: ±15%\n"
                "15 min: ±20%\n\n"
                "Building price history now.\n"
                "Full monitoring becomes available "
                "after approximately 15 minutes."
            )

            first_run = False

        for alert in alerts:
            try:
                send_signal(alert)
            except Exception as exc:
                print(
                    "Telegram signal error:",
                    exc
                )

        return {
            "success": True,
            "markets": len(current_prices),
            "endpoint": endpoint,
            "history_snapshots": len(history),
            "signals_detected": len(alerts),
            "warmup": get_warmup_status(now),
        }


# ============================================================
# WARMUP STATUS
# ============================================================

def get_warmup_status(now=None):
    if now is None:
        now = time.time()

    if not history:
        return {
            "5m": False,
            "10m": False,
            "15m": False,
        }

    oldest_age = (
        now - history[0]["timestamp"]
    )

    return {
        "5m": oldest_age >= (5 * 60),
        "10m": oldest_age >= (10 * 60),
        "15m": oldest_age >= (15 * 60),
    }


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def home():
    warmup = get_warmup_status()

    last_run_text = None

    if last_success:
        last_run_text = datetime.fromtimestamp(
            last_success,
            UK_TIMEZONE
        ).strftime(
            "%d %b %Y %H:%M:%S"
        )

    return jsonify({
        "service": "Signal Watch",
        "status": "online",
        "markets_last_seen": last_market_count,
        "history_snapshots": len(history),
        "last_success_uk": last_run_text,
        "warmup": warmup,
        "thresholds": {
            "5_min": "±10%",
            "10_min": "±15%",
            "15_min": "±20%",
        }
    })


@app.route("/run")
def run():
    try:
        result = run_detector()

        return jsonify(result), 200

    except Exception as exc:
        print(
            "Detector error:",
            exc
        )

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "OK"
    }), 200


@app.route("/telegram-test")
def telegram_test():
    try:
        send_telegram(
            "🧪 SIGNAL WATCH TEST\n\n"
            "Render → Telegram connection "
            "is working correctly.\n\n"
            "5 min: ±10%\n"
            "10 min: ±15%\n"
            "15 min: ±20%"
        )

        return jsonify({
            "success": True,
            "message": "Telegram test sent"
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500

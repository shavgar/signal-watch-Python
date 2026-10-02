import os
import re
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

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


def clean_secret(value):
    """
    Remove accidental spaces / zero-width characters from
    environment variables.
    """
    return re.sub(
        r"[\s\u200B\u200C\u200D\uFEFF]",
        "",
        value or ""
    )


TELEGRAM_BOT_TOKEN = clean_secret(
    os.environ.get("TELEGRAM_BOT_TOKEN", "")
)

TELEGRAM_CHAT_ID = clean_secret(
    os.environ.get("TELEGRAM_CHAT_ID", "")
)


# ------------------------------------------------------------
# NORMAL PUMP / CRASH THRESHOLDS
# ------------------------------------------------------------

WINDOWS = {
    5: 10.0,     # ±10% in 5 minutes
    10: 15.0,    # ±15% in 10 minutes
    15: 20.0,    # ±20% in 15 minutes
}


# ------------------------------------------------------------
# 24-HOUR THRESHOLD
# ------------------------------------------------------------

DAILY_THRESHOLD = 30.0


# ------------------------------------------------------------
# RE-ARM
# ------------------------------------------------------------

REARM_MARGIN = 2.0


# ------------------------------------------------------------
# HISTORY
#
# We now keep 3 hours because the staircase detector needs
# several completed 15-minute candles.
# ------------------------------------------------------------

HISTORY_KEEP_SECONDS = 3 * 60 * 60

REFERENCE_TOLERANCE_SECONDS = 90


# ------------------------------------------------------------
# STAIRCASE DETECTOR
# ------------------------------------------------------------

STAIRCASE_TIMEFRAME_MINUTES = 15

# Analyse 5 completed 15-minute candles = about 75 minutes.
STAIRCASE_CANDLES = 5

# At least 4 of the 5 candles should close green.
STAIRCASE_MIN_GREEN_CANDLES = 4

# With 5 candles there are 4 close-to-close comparisons.
STAIRCASE_MIN_HIGHER_CLOSES = 4

# At least 3 of the 4 low comparisons should be higher.
STAIRCASE_MIN_HIGHER_LOWS = 3

# The gradual move should already show some strength...
STAIRCASE_MIN_TOTAL_RISE = 2.5

# ...but should not already be a huge pump.
STAIRCASE_MAX_TOTAL_RISE = 12.0

# Prevent a large breakout candle from being mistaken
# for the pre-breakout staircase.
STAIRCASE_MAX_CANDLE_BODY = 4.0

# Avoid unusually violent candles.
STAIRCASE_MAX_CANDLE_RANGE = 6.0

# Allow one small red candle, but not a large dump candle.
STAIRCASE_MAX_RED_BODY = 1.5

# Each synthetic 15-minute candle should contain
# enough 1-minute observations.
STAIRCASE_MIN_SAMPLES_PER_CANDLE = 8

# If a pattern disappears then forms again immediately,
# avoid repeatedly alerting.
STAIRCASE_COOLDOWN_SECONDS = 2 * 60 * 60


UK_TIMEZONE = ZoneInfo("Europe/London")


# ============================================================
# IN-MEMORY STATE
# ============================================================

history = []

# Used for normal and 24h anti-spam logic.
#
# Example:
# signal_state["ABCUSDT:5"] = "PUMP"
# signal_state["ABCUSDT:24h"] = "CRASH"
signal_state = {}

# Staircase anti-spam state.
staircase_active = set()
staircase_last_alert = {}

lock = threading.Lock()

first_run = True
last_success = None
last_market_count = 0


# ============================================================
# SYMBOL HELPERS
# ============================================================

def coin_name(symbol):
    """
    BTCUSDT -> BTC
    ABCUSDT -> ABC
    """
    if symbol.endswith("USDT"):
        return symbol[:-4]

    return symbol


def bybit_spot_url(symbol):
    """
    Create the Bybit UK Spot trading URL.
    """
    coin = quote(
        coin_name(symbol),
        safe=""
    )

    return (
        "https://www.bybit.com/"
        f"en-GB/trade/spot/{coin}/USDT"
    )


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

            daily_data = {}


            for ticker in data.get(
                "result",
                {}
            ).get(
                "list",
                []
            ):

                symbol = ticker.get(
                    "symbol",
                    ""
                )


                if not symbol.endswith("USDT"):
                    continue


                # ----------------------------
                # CURRENT PRICE
                # ----------------------------

                try:

                    price = float(
                        ticker.get("lastPrice")
                    )

                except (TypeError, ValueError):

                    continue


                if price <= 0:
                    continue


                prices[symbol] = price


                # ----------------------------
                # 24-HOUR DATA
                # ----------------------------

                try:

                    change_24h = (
                        float(
                            ticker.get(
                                "price24hPcnt"
                            )
                        )
                        * 100
                    )


                    previous_24h = float(
                        ticker.get(
                            "prevPrice24h"
                        )
                    )


                    if previous_24h > 0:

                        daily_data[symbol] = {
                            "change": change_24h,
                            "reference_price": previous_24h,
                            "current_price": price,
                        }


                except (
                    TypeError,
                    ValueError
                ):

                    pass


            if len(prices) < 10:

                errors.append(
                    f"{url}: only "
                    f"{len(prices)} "
                    "USDT pairs returned"
                )

                continue


            return (
                prices,
                daily_data,
                url
            )


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

    cutoff = (
        now
        - HISTORY_KEEP_SECONDS
    )


    while (
        history
        and history[0]["timestamp"] < cutoff
    ):

        history.pop(0)


def save_snapshot(
    timestamp,
    prices
):

    history.append({
        "timestamp": timestamp,
        "prices": prices,
    })


    clean_old_history(
        timestamp
    )


def find_reference_snapshot(
    now,
    minutes
):

    target = (
        now
        - minutes * 60
    )


    best = None


    for snapshot in history:

        timestamp = snapshot[
            "timestamp"
        ]


        if timestamp <= target:

            if (
                best is None
                or timestamp
                > best["timestamp"]
            ):

                best = snapshot


    if best is None:
        return None


    difference = (
        target
        - best["timestamp"]
    )


    if (
        difference
        > REFERENCE_TOLERANCE_SECONDS
    ):

        return None


    return best


# ============================================================
# NORMAL SIGNAL DETECTION
# ============================================================

def detect_signals(
    now,
    current_prices
):

    alerts = []


    for (
        minutes,
        threshold
    ) in WINDOWS.items():


        reference = (
            find_reference_snapshot(
                now,
                minutes
            )
        )


        if reference is None:
            continue


        reference_prices = (
            reference["prices"]
        )


        for (
            symbol,
            current_price
        ) in current_prices.items():


            reference_price = (
                reference_prices.get(
                    symbol
                )
            )


            if (
                not reference_price
                or reference_price <= 0
            ):

                continue


            change = (
                (
                    current_price
                    - reference_price
                )
                / reference_price
            ) * 100


            state_key = (
                f"{symbol}:{minutes}"
            )


            previous_state = (
                signal_state.get(
                    state_key
                )
            )


            # ----------------------------
            # PUMP
            # ----------------------------

            if change >= threshold:

                if (
                    previous_state
                    != "PUMP"
                ):

                    alerts.append({
                        "symbol": symbol,
                        "direction": "PUMP",
                        "window_label":
                            f"{minutes} minutes",
                        "change": change,
                        "reference_price":
                            reference_price,
                        "current_price":
                            current_price,
                    })


                signal_state[
                    state_key
                ] = "PUMP"


            # ----------------------------
            # CRASH
            # ----------------------------

            elif change <= -threshold:

                if (
                    previous_state
                    != "CRASH"
                ):

                    alerts.append({
                        "symbol": symbol,
                        "direction": "CRASH",
                        "window_label":
                            f"{minutes} minutes",
                        "change": change,
                        "reference_price":
                            reference_price,
                        "current_price":
                            current_price,
                    })


                signal_state[
                    state_key
                ] = "CRASH"


            # ----------------------------
            # RE-ARM
            # ----------------------------

            elif (
                abs(change)
                <= max(
                    0,
                    threshold
                    - REARM_MARGIN
                )
            ):

                signal_state.pop(
                    state_key,
                    None
                )


    return alerts


# ============================================================
# 24-HOUR SIGNAL DETECTION
# ============================================================

def detect_24h_signals(
    daily_data
):

    alerts = []


    for (
        symbol,
        data
    ) in daily_data.items():


        change = data[
            "change"
        ]


        state_key = (
            f"{symbol}:24h"
        )


        previous_state = (
            signal_state.get(
                state_key
            )
        )


        # ----------------------------
        # 24H PUMP
        # ----------------------------

        if (
            change
            >= DAILY_THRESHOLD
        ):

            if (
                previous_state
                != "PUMP"
            ):

                alerts.append({
                    "symbol": symbol,
                    "direction": "PUMP",
                    "window_label":
                        "24 hours",
                    "change": change,
                    "reference_price":
                        data[
                            "reference_price"
                        ],
                    "current_price":
                        data[
                            "current_price"
                        ],
                })


            signal_state[
                state_key
            ] = "PUMP"


        # ----------------------------
        # 24H CRASH
        # ----------------------------

        elif (
            change
            <= -DAILY_THRESHOLD
        ):

            if (
                previous_state
                != "CRASH"
            ):

                alerts.append({
                    "symbol": symbol,
                    "direction": "CRASH",
                    "window_label":
                        "24 hours",
                    "change": change,
                    "reference_price":
                        data[
                            "reference_price"
                        ],
                    "current_price":
                        data[
                            "current_price"
                        ],
                })


            signal_state[
                state_key
            ] = "CRASH"


        # ----------------------------
        # RE-ARM
        # ----------------------------

        elif (
            abs(change)
            <= (
                DAILY_THRESHOLD
                - REARM_MARGIN
            )
        ):

            signal_state.pop(
                state_key,
                None
            )


    return alerts


# ============================================================
# BUILD SYNTHETIC 15-MINUTE CANDLES
# ============================================================

def build_completed_15m_candles(
    now
):

    timeframe_seconds = (
        STAIRCASE_TIMEFRAME_MINUTES
        * 60
    )


    # Beginning of the current
    # incomplete 15-minute bucket.
    current_bucket = (
        int(
            now
            // timeframe_seconds
        )
        * timeframe_seconds
    )


    first_bucket = (
        current_bucket
        - (
            STAIRCASE_CANDLES
            * timeframe_seconds
        )
    )


    required_buckets = [
        first_bucket
        + (
            i
            * timeframe_seconds
        )
        for i in range(
            STAIRCASE_CANDLES
        )
    ]


    bucket_data = {
        bucket: {}
        for bucket
        in required_buckets
    }


    for snapshot in history:

        timestamp = snapshot[
            "timestamp"
        ]


        if (
            timestamp < first_bucket
            or timestamp
            >= current_bucket
        ):

            continue


        bucket = (
            int(
                timestamp
                // timeframe_seconds
            )
            * timeframe_seconds
        )


        if bucket not in bucket_data:
            continue


        prices = snapshot[
            "prices"
        ]


        for (
            symbol,
            price
        ) in prices.items():


            candle = (
                bucket_data[
                    bucket
                ].get(symbol)
            )


            if candle is None:

                bucket_data[
                    bucket
                ][symbol] = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "samples": 1,
                }


            else:

                candle[
                    "high"
                ] = max(
                    candle["high"],
                    price
                )


                candle[
                    "low"
                ] = min(
                    candle["low"],
                    price
                )


                candle[
                    "close"
                ] = price


                candle[
                    "samples"
                ] += 1


    candles_by_symbol = {}


    symbols = set()


    for bucket in required_buckets:

        symbols.update(
            bucket_data[
                bucket
            ].keys()
        )


    for symbol in symbols:

        candles = []

        valid = True


        for bucket in required_buckets:

            candle = (
                bucket_data[
                    bucket
                ].get(symbol)
            )


            if (
                candle is None
                or candle[
                    "samples"
                ]
                < STAIRCASE_MIN_SAMPLES_PER_CANDLE
            ):

                valid = False

                break


            candles.append(
                candle
            )


        if valid:

            candles_by_symbol[
                symbol
            ] = candles


    return candles_by_symbol


# ============================================================
# STAIRCASE PATTERN
# ============================================================

def staircase_metrics(
    candles
):

    green_candles = sum(
        1
        for candle in candles
        if (
            candle["close"]
            > candle["open"]
        )
    )


    higher_closes = sum(
        1
        for index
        in range(
            1,
            len(candles)
        )
        if (
            candles[index][
                "close"
            ]
            >
            candles[index - 1][
                "close"
            ]
        )
    )


    higher_lows = sum(
        1
        for index
        in range(
            1,
            len(candles)
        )
        if (
            candles[index][
                "low"
            ]
            >
            candles[index - 1][
                "low"
            ]
        )
    )


    first_open = (
        candles[0]["open"]
    )


    last_close = (
        candles[-1]["close"]
    )


    total_rise = (
        (
            last_close
            - first_open
        )
        / first_open
    ) * 100


    body_changes = []


    candle_ranges = []


    for candle in candles:

        body = (
            (
                candle["close"]
                - candle["open"]
            )
            / candle["open"]
        ) * 100


        candle_range = (
            (
                candle["high"]
                - candle["low"]
            )
            / candle["low"]
        ) * 100


        body_changes.append(
            body
        )


        candle_ranges.append(
            candle_range
        )


    return {
        "green_candles":
            green_candles,

        "higher_closes":
            higher_closes,

        "higher_lows":
            higher_lows,

        "total_rise":
            total_rise,

        "max_body":
            max(
                abs(x)
                for x
                in body_changes
            ),

        "largest_red_body":
            abs(
                min(
                    min(
                        body_changes
                    ),
                    0
                )
            ),

        "max_range":
            max(
                candle_ranges
            ),

        "current_price":
            last_close,
    }


def qualifies_as_staircase(
    metrics
):

    return (
        metrics[
            "green_candles"
        ]
        >=
        STAIRCASE_MIN_GREEN_CANDLES

        and

        metrics[
            "higher_closes"
        ]
        >=
        STAIRCASE_MIN_HIGHER_CLOSES

        and

        metrics[
            "higher_lows"
        ]
        >=
        STAIRCASE_MIN_HIGHER_LOWS

        and

        STAIRCASE_MIN_TOTAL_RISE
        <=
        metrics[
            "total_rise"
        ]
        <=
        STAIRCASE_MAX_TOTAL_RISE

        and

        metrics[
            "max_body"
        ]
        <=
        STAIRCASE_MAX_CANDLE_BODY

        and

        metrics[
            "max_range"
        ]
        <=
        STAIRCASE_MAX_CANDLE_RANGE

        and

        metrics[
            "largest_red_body"
        ]
        <=
        STAIRCASE_MAX_RED_BODY
    )


def detect_staircases(
    now
):

    global staircase_active


    candle_sets = (
        build_completed_15m_candles(
            now
        )
    )


    alerts = []


    matched_symbols = set()


    for (
        symbol,
        candles
    ) in candle_sets.items():


        metrics = (
            staircase_metrics(
                candles
            )
        )


        if not qualifies_as_staircase(
            metrics
        ):

            continue


        matched_symbols.add(
            symbol
        )


        # Already alerted while this
        # particular pattern remains active.
        if (
            symbol
            in staircase_active
        ):

            continue


        previous_alert_time = (
            staircase_last_alert.get(
                symbol,
                0
            )
        )


        if (
            now
            - previous_alert_time
            <
            STAIRCASE_COOLDOWN_SECONDS
        ):

            continue


        alerts.append({
            "symbol": symbol,
            "direction":
                "STAIRCASE",
            "timeframe":
                "15m",
            "candles":
                STAIRCASE_CANDLES,
            "green_candles":
                metrics[
                    "green_candles"
                ],
            "higher_closes":
                metrics[
                    "higher_closes"
                ],
            "higher_lows":
                metrics[
                    "higher_lows"
                ],
            "total_rise":
                metrics[
                    "total_rise"
                ],
            "current_price":
                metrics[
                    "current_price"
                ],
        })


        staircase_last_alert[
            symbol
        ] = now


        staircase_active.add(
            symbol
        )


    # Rearm a coin after the staircase
    # pattern disappears.
    staircase_active.intersection_update(
        matched_symbols
    )


    return alerts


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    message,
    button_url=None,
    button_text=None
):

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing"
        )


    if not TELEGRAM_CHAT_ID:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing"
        )


    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/"
        "sendMessage"
    )


    payload = {
        "chat_id":
            TELEGRAM_CHAT_ID,

        "text":
            message,

        "disable_web_page_preview":
            True,
    }


    if (
        button_url
        and button_text
    ):

        payload[
            "reply_markup"
        ] = {
            "inline_keyboard": [
                [
                    {
                        "text":
                            button_text,

                        "url":
                            button_url,
                    }
                ]
            ]
        }


    response = requests.post(
        url,
        json=payload,
        timeout=15
    )


    if response.status_code != 200:

        raise RuntimeError(
            "Telegram error: "
            + response.text[:500]
        )


# ============================================================
# FORMATTING
# ============================================================

def format_price(
    price
):

    if price >= 1000:

        return (
            f"${price:,.2f}"
        )


    if price >= 1:

        return (
            f"${price:.4f}"
        )


    if price >= 0.01:

        return (
            f"${price:.6f}"
        )


    return (
        f"${price:.8f}"
    )


def current_uk_time():

    return datetime.now(
        UK_TIMEZONE
    ).strftime(
        "%d %b %Y %H:%M:%S"
    )


# ============================================================
# NORMAL / 24H TELEGRAM SIGNAL
# ============================================================

def send_signal(
    alert
):

    direction = alert[
        "direction"
    ]


    symbol = alert[
        "symbol"
    ]


    coin = coin_name(
        symbol
    )


    icon = (
        "🚀"
        if direction == "PUMP"
        else "🔻"
    )


    change = alert[
        "change"
    ]


    change_text = (
        f"+{change:.2f}%"
        if change >= 0
        else f"{change:.2f}%"
    )


    title = (
        f"{icon} "
        f"{coin} "
        f"{direction}"
    )


    message = (
        f"{title}\n\n"

        f"Move: "
        f"{change_text}\n"

        f"Window: "
        f"{alert['window_label']}\n\n"

        f"Reference: "
        f"{format_price(alert['reference_price'])}\n"

        f"Current: "
        f"{format_price(alert['current_price'])}\n\n"

        f"Detected: "
        f"{current_uk_time()} UK"
    )


    send_telegram(
        message,
        button_url=
            bybit_spot_url(
                symbol
            ),
        button_text=
            f"📈 Open {coin} on Bybit"
    )


# ============================================================
# STAIRCASE TELEGRAM SIGNAL
# ============================================================

def send_staircase_signal(
    alert
):

    symbol = alert[
        "symbol"
    ]


    coin = coin_name(
        symbol
    )


    message = (
        f"🪜 {coin} STAIRCASE\n\n"

        f"Timeframe: "
        f"{alert['timeframe']}\n"

        f"Candles analysed: "
        f"{alert['candles']}\n"

        f"Rise: "
        f"+{alert['total_rise']:.2f}%\n"

        f"Green candles: "
        f"{alert['green_candles']}/"
        f"{alert['candles']}\n"

        f"Higher closes: "
        f"{alert['higher_closes']}/"
        f"{alert['candles'] - 1}\n"

        f"Higher lows: "
        f"{alert['higher_lows']}/"
        f"{alert['candles'] - 1}\n\n"

        f"Current: "
        f"{format_price(alert['current_price'])}\n\n"

        f"Detected: "
        f"{current_uk_time()} UK\n\n"

        f"Early staircase pattern detected."
    )


    send_telegram(
        message,
        button_url=
            bybit_spot_url(
                symbol
            ),
        button_text=
            f"📈 Open {coin} on Bybit"
    )


# ============================================================
# MAIN DETECTOR
# ============================================================

def run_detector():

    global first_run
    global last_success
    global last_market_count


    with lock:

        now = time.time()


        (
            current_prices,
            daily_data,
            endpoint
        ) = fetch_bybit_prices()


        # ----------------------------
        # 5m / 10m / 15m
        # ----------------------------

        alerts = detect_signals(
            now,
            current_prices
        )


        # ----------------------------
        # 24 HOURS
        # ----------------------------

        alerts.extend(
            detect_24h_signals(
                daily_data
            )
        )


        # ----------------------------
        # SAVE CURRENT SNAPSHOT
        # ----------------------------

        save_snapshot(
            now,
            current_prices
        )


        # ----------------------------
        # STAIRCASE
        # ----------------------------

        staircase_alerts = (
            detect_staircases(
                now
            )
        )


        last_success = now

        last_market_count = (
            len(current_prices)
        )


        # ----------------------------
        # STARTUP MESSAGE
        # ----------------------------

        if first_run:

            send_telegram(
                "✅ SIGNAL WATCH ONLINE\n\n"

                f"Bybit connection: OK\n"

                f"USDT Spot markets: "
                f"{len(current_prices)}\n\n"

                "Pump / Crash:\n"

                "5 min: ±10%\n"
                "10 min: ±15%\n"
                "15 min: ±20%\n"
                "24 hours: ±30%\n\n"

                "🪜 15m Staircase detector enabled.\n\n"

                "5/10/15 minute monitoring "
                "warms up within about "
                "15 minutes.\n"

                "The staircase detector needs "
                "roughly 75–90 minutes of "
                "history after a restart."
            )


            first_run = False


        # ----------------------------
        # SEND NORMAL SIGNALS
        # ----------------------------

        for alert in alerts:

            try:

                send_signal(
                    alert
                )

            except Exception as exc:

                print(
                    "Telegram signal error:",
                    exc
                )


        # ----------------------------
        # SEND STAIRCASE SIGNALS
        # ----------------------------

        for alert in staircase_alerts:

            try:

                send_staircase_signal(
                    alert
                )

            except Exception as exc:

                print(
                    "Telegram staircase error:",
                    exc
                )


        return {
            "success":
                True,

            "markets":
                len(current_prices),

            "endpoint":
                endpoint,

            "history_snapshots":
                len(history),

            "pump_crash_signals":
                len(alerts),

            "staircase_signals":
                len(
                    staircase_alerts
                ),

            "warmup":
                get_warmup_status(
                    now
                ),
        }


# ============================================================
# WARMUP STATUS
# ============================================================

def get_warmup_status(
    now=None
):

    if now is None:

        now = time.time()


    if not history:

        return {
            "5m": False,
            "10m": False,
            "15m": False,
            "24h": True,
            "staircase_15m":
                False,
        }


    oldest_age = (
        now
        - history[0][
            "timestamp"
        ]
    )


    return {
        "5m":
            oldest_age
            >= 5 * 60,

        "10m":
            oldest_age
            >= 10 * 60,

        "15m":
            oldest_age
            >= 15 * 60,

        # Uses Bybit's own 24h data,
        # so it needs no local warmup.
        "24h":
            True,

        # 90 minutes is a conservative
        # indication that five completed
        # 15m candles are available.
        "staircase_15m":
            oldest_age
            >= 90 * 60,
    }


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def home():

    warmup = (
        get_warmup_status()
    )


    last_run_text = None


    if last_success:

        last_run_text = (
            datetime.fromtimestamp(
                last_success,
                UK_TIMEZONE
            ).strftime(
                "%d %b %Y %H:%M:%S"
            )
        )


    return jsonify({
        "service":
            "Signal Watch",

        "status":
            "online",

        "markets_last_seen":
            last_market_count,

        "history_snapshots":
            len(history),

        "last_success_uk":
            last_run_text,

        "warmup":
            warmup,

        "thresholds": {
            "5_min":
                "±10%",

            "10_min":
                "±15%",

            "15_min":
                "±20%",

            "24_hours":
                "±30%",
        },

        "staircase": {
            "timeframe":
                "15m",

            "candles":
                STAIRCASE_CANDLES,

            "minimum_green":
                STAIRCASE_MIN_GREEN_CANDLES,

            "minimum_higher_closes":
                STAIRCASE_MIN_HIGHER_CLOSES,

            "minimum_higher_lows":
                STAIRCASE_MIN_HIGHER_LOWS,

            "rise_range":
                "2.5% to 12%",
        },
    })


@app.route("/run")
def run():

    try:

        result = (
            run_detector()
        )


        return jsonify(
            result
        ), 200


    except Exception as exc:

        print(
            "Detector error:",
            exc
        )


        return jsonify({
            "success":
                False,

            "error":
                str(exc),
        }), 500


@app.route("/health")
def health():

    return jsonify({
        "status":
            "OK"
    }), 200


@app.route("/telegram-test")
def telegram_test():

    try:

        send_telegram(
            "🧪 TELEGRAM TEST\n\n"
            "Signal Watch is connected "
            "correctly.\n\n"
            "5 min: ±10%\n"
            "10 min: ±15%\n"
            "15 min: ±20%\n"
            "24 hours: ±30%\n\n"
            "🪜 Staircase: 15m",

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
                str(exc),
        }), 500

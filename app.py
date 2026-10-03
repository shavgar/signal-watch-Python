import os
import re
import time
import threading
import statistics
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import requests
from flask import Flask, jsonify


app = Flask(__name__)


# ============================================================
# CONFIG
# ============================================================

BYBIT_API_BASES = [
    "https://api.bybit.com",
    "https://api.bytick.com",
]

TELEGRAM_API_BASE = "https://api.telegram.org"

UK_TIMEZONE = ZoneInfo("Europe/London")


def clean_secret(value):
    return re.sub(
        r"[\s\u200B\u200C\u200D\uFEFF]",
        "",
        value or ""
    )


TELEGRAM_BOT_TOKEN = clean_secret(
    os.environ.get(
        "TELEGRAM_BOT_TOKEN",
        ""
    )
)

TELEGRAM_CHAT_ID = clean_secret(
    os.environ.get(
        "TELEGRAM_CHAT_ID",
        ""
    )
)


# ============================================================
# EXISTING PUMP / CRASH RULES
# ============================================================

WINDOWS = {
    5: 10.0,
    10: 15.0,
    15: 20.0,
}

DAILY_THRESHOLD = 30.0

REARM_MARGIN = 2.0


# ============================================================
# HISTORY
# ============================================================

# Keep 3 hours of one-minute snapshots.
HISTORY_KEEP_SECONDS = 3 * 60 * 60

REFERENCE_TOLERANCE_SECONDS = 90


# ------------------------------------------------------------
# STAIRCASE + BREAKOUT DETECTOR
# ------------------------------------------------------------

STAIRCASE_TIMEFRAME_MINUTES = 15

# We will analyse 5 completed 15m candles:
# first 4 = staircase
# last 1 = breakout
STAIRCASE_CANDLES = 5

# First 4 candles should mostly be green
STAIR_STEP_GREEN_MIN = 3

# First 4 candles should show higher closes / higher lows
STAIR_STEP_HIGHER_CLOSES_MIN = 3
STAIR_STEP_HIGHER_LOWS_MIN = 3

# Total rise across the first 4 stair candles
STAIR_STEP_MIN_TOTAL_RISE = 1.0
STAIR_STEP_MAX_TOTAL_RISE = 8.0

# Stair candles should stay relatively small
STAIR_STEP_MAX_BODY = 2.2
STAIR_STEP_MAX_RANGE = 3.5

# Breakout candle should be clearly bigger
BREAKOUT_MIN_BODY = 3.0
BREAKOUT_MAX_BODY = 12.0
BREAKOUT_MIN_MULTIPLIER = 1.8   # breakout body vs average stair body

# The breakout candle should close above all previous stair highs
BREAKOUT_MUST_CLOSE_ABOVE_PREVIOUS_HIGHS = True

# Enough 1-minute samples per synthetic 15m candle
STAIRCASE_MIN_SAMPLES_PER_CANDLE = 8

# Avoid repeated alerts
STAIRCASE_COOLDOWN_SECONDS = 2 * 60 * 60

# ============================================================
# EARLY PUMP
# ============================================================

# Cheap first-stage filter.
#
# A coin only gets the more expensive
# volume / trades / order-book check
# if it first starts moving unusually.

EARLY_PREFILTER_1M = 1.5

EARLY_PREFILTER_3M = 3.0

EARLY_PREFILTER_5M = 4.5


# Score needed for Telegram alert.

EARLY_ALERT_SCORE = 70

EARLY_STRONG_SCORE = 85


# If the score falls below this,
# the coin becomes eligible again later.

EARLY_REARM_SCORE = 55


# Prevent repeated alerts for same coin.

EARLY_COOLDOWN_SECONDS = (
    30 * 60
)


# Maximum number of suspicious coins
# we deeply analyse in one cron run.

MAX_DEEP_SCANS_PER_RUN = 15


# ============================================================
# U-TURN AFTER CRASH
# ============================================================

# Watch a crashed coin for up to 3 hours.

UTURN_WATCH_SECONDS = (
    3 * 60 * 60
)


# Must recover at least 5% from
# its post-crash low.

UTURN_MIN_RECOVERY = 5.0


# Short-term upward movement needed.

UTURN_MIN_1M_CHANGE = 1.0

UTURN_MIN_3M_CHANGE = 2.0


# Minimum U-turn confidence score.

UTURN_ALERT_SCORE = 65


# ============================================================
# IN-MEMORY STATE
# ============================================================

history = []


# Existing Pump / Crash anti-spam.

signal_state = {}


# Staircase anti-spam.

staircase_active = set()

staircase_last_alert = {}


# Early Pump anti-spam.

early_active = set()

early_last_alert = {}


# U-turn watches.
#
# Example:
#
# uturn_watch["ABCUSDT"] = {
#     ...
# }

uturn_watch = {}


# ============================================================
# SIGNAL JOURNAL
#
# Tracks what happened after alerts.
#
# IMPORTANT:
# Free Render may restart.
# If it restarts, this memory resets.
# ============================================================

signal_log = []

MAX_SIGNAL_LOG = 500


# ============================================================
# GENERAL STATE
# ============================================================

lock = threading.Lock()

first_run = True

last_success = None

last_market_count = 0


# ============================================================
# SYMBOL HELPERS
# ============================================================

def coin_name(symbol):

    if symbol.endswith("USDT"):

        return symbol[:-4]

    return symbol


def bybit_spot_url(symbol):

    """
    Telegram button -> Bybit app
    -> coin candle / market page.
    """

    native_route = (
        "bybitapp://open/home?"
        "tab=3"
        "&tabName=margin"
        "&page=market"
        f"&symbol={symbol}"
    )

    fallback_url = (
        "https://www.bybit.com/"
        f"en-GB/trade/spot/"
        f"{coin_name(symbol)}/USDT"
    )

    return (
        "https://app.bybit.com/inapp"
        f"?by_dp="
        f"{quote(native_route, safe='')}"
        f"&by_web_link="
        f"{quote(fallback_url, safe='')}"
    )


# ============================================================
# FORMATTING
# ============================================================

def format_price(price):

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


def percent_change(
    new_price,
    old_price
):

    if (
        not old_price
        or old_price <= 0
    ):

        return None

    return (
        (
            new_price
            - old_price
        )
        / old_price
    ) * 100


def fmt_optional_percent(value):

    if value is None:

        return "warming up"

    return (
        f"{value:+.2f}%"
    )


def fmt_optional_ratio(value):

    if value is None:

        return "n/a"

    return (
        f"{value:.2f}x"
    )


def fmt_optional_pressure(value):

    if value is None:

        return "n/a"

    return (
        f"{value:.1f}%"
    )


# ============================================================
# BYBIT HTTP
# ============================================================

def bybit_get(
    path,
    params=None,
    timeout=15
):

    headers = {
        "User-Agent":
            "Signal-Watch/2.0"
    }

    errors = []

    for base in BYBIT_API_BASES:

        url = (
            base
            + path
        )

        try:

            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout
            )

            if (
                response.status_code
                != 200
            ):

                errors.append(
                    f"{url}: "
                    f"HTTP "
                    f"{response.status_code}"
                )

                continue

            data = response.json()

            if (
                data.get("retCode")
                != 0
            ):

                errors.append(
                    f"{url}: "
                    f"{data.get('retMsg')}"
                )

                continue

            return (
                data,
                url
            )

        except Exception as exc:

            errors.append(
                f"{url}: "
                f"{str(exc)}"
            )

    raise RuntimeError(
        "All Bybit endpoints failed: "
        + " | ".join(errors)
    )


# ============================================================
# FETCH ALL SPOT PRICES
# ============================================================

def fetch_bybit_prices():

    data, endpoint = bybit_get(
        "/v5/market/tickers",
        {
            "category":
                "spot"
        }
    )

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

        if not symbol.endswith(
            "USDT"
        ):

            continue

        try:

            price = float(
                ticker.get(
                    "lastPrice"
                )
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if price <= 0:

            continue

        prices[
            symbol
        ] = price

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

                daily_data[
                    symbol
                ] = {

                    "change":
                        change_24h,

                    "reference_price":
                        previous_24h,

                    "current_price":
                        price,
                }

        except (
            TypeError,
            ValueError
        ):

            pass

    if len(prices) < 10:

        raise RuntimeError(
            f"Only "
            f"{len(prices)} "
            f"USDT Spot markets "
            f"returned."
        )

    return (
        prices,
        daily_data,
        endpoint
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
        and
        history[0][
            "timestamp"
        ] < cutoff
    ):

        history.pop(0)


def save_snapshot(
    timestamp,
    prices
):

    history.append({

        "timestamp":
            timestamp,

        "prices":
            prices,
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
                or
                timestamp
                > best["timestamp"]
            ):

                best = snapshot

    if best is None:

        return None

    difference = (
        target
        - best[
            "timestamp"
        ]
    )

    if (
        difference
        > REFERENCE_TOLERANCE_SECONDS
    ):

        return None

    return best


def change_for_symbol(
    now,
    symbol,
    current_price,
    minutes
):

    reference = (
        find_reference_snapshot(
            now,
            minutes
        )
    )

    if reference is None:

        return None

    reference_price = (
        reference[
            "prices"
        ].get(
            symbol
        )
    )

    return percent_change(
        current_price,
        reference_price
    )


# ============================================================
# NORMAL 5m / 10m / 15m PUMP + CRASH
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
            reference[
                "prices"
            ]
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
                or
                reference_price <= 0
            ):

                continue

            change = (
                percent_change(
                    current_price,
                    reference_price
                )
            )

            state_key = (
                f"{symbol}:"
                f"{minutes}"
            )

            previous_state = (
                signal_state.get(
                    state_key
                )
            )

            # ------------------------
            # PUMP
            # ------------------------

            if (
                change
                >= threshold
            ):

                if (
                    previous_state
                    != "PUMP"
                ):

                    alerts.append({

                        "symbol":
                            symbol,

                        "direction":
                            "PUMP",

                        "window_label":
                            f"{minutes} "
                            f"minutes",

                        "change":
                            change,

                        "reference_price":
                            reference_price,

                        "current_price":
                            current_price,
                    })

                signal_state[
                    state_key
                ] = "PUMP"

            # ------------------------
            # CRASH
            # ------------------------

            elif (
                change
                <= -threshold
            ):

                if (
                    previous_state
                    != "CRASH"
                ):

                    alerts.append({

                        "symbol":
                            symbol,

                        "direction":
                            "CRASH",

                        "window_label":
                            f"{minutes} "
                            f"minutes",

                        "change":
                            change,

                        "reference_price":
                            reference_price,

                        "current_price":
                            current_price,
                    })

                signal_state[
                    state_key
                ] = "CRASH"

            # ------------------------
            # RE-ARM
            # ------------------------

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
# 24-HOUR PUMP + CRASH
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

        # ------------------------
        # 24H PUMP
        # ------------------------

        if (
            change
            >= DAILY_THRESHOLD
        ):

            if (
                previous_state
                != "PUMP"
            ):

                alerts.append({

                    "symbol":
                        symbol,

                    "direction":
                        "PUMP",

                    "window_label":
                        "24 hours",

                    "change":
                        change,

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

        # ------------------------
        # 24H CRASH
        # ------------------------

        elif (
            change
            <= -DAILY_THRESHOLD
        ):

            if (
                previous_state
                != "CRASH"
            ):

                alerts.append({

                    "symbol":
                        symbol,

                    "direction":
                        "CRASH",

                    "window_label":
                        "24 hours",

                    "change":
                        change,

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

        # ------------------------
        # RE-ARM
        # ------------------------

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
            or
            timestamp >= current_bucket
        ):

            continue

        bucket = (
            int(
                timestamp
                // timeframe_seconds
            )
            * timeframe_seconds
        )

        if (
            bucket
            not in bucket_data
        ):

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
                ].get(
                    symbol
                )
            )

            if candle is None:

                bucket_data[
                    bucket
                ][
                    symbol
                ] = {

                    "open":
                        price,

                    "high":
                        price,

                    "low":
                        price,

                    "close":
                        price,

                    "samples":
                        1,
                }

            else:

                candle[
                    "high"
                ] = max(
                    candle[
                        "high"
                    ],
                    price
                )

                candle[
                    "low"
                ] = min(
                    candle[
                        "low"
                    ],
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
                ].get(
                    symbol
                )
            )

            if (
                candle is None
                or
                candle[
                    "samples"
                ]
                <
                STAIRCASE_MIN_SAMPLES_PER_CANDLE
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
# STAIRCASE METRICS
# ============================================================

def staircase_metrics(candles):
    """
    candles = 5 completed 15m candles
    first 4 = staircase
    last 1 = breakout
    """

    stair = candles[:-1]
    breakout = candles[-1]

    stair_green = sum(
        1 for candle in stair
        if candle["close"] > candle["open"]
    )

    stair_higher_closes = sum(
        1 for i in range(1, len(stair))
        if stair[i]["close"] > stair[i - 1]["close"]
    )

    stair_higher_lows = sum(
        1 for i in range(1, len(stair))
        if stair[i]["low"] > stair[i - 1]["low"]
    )

    stair_first_open = stair[0]["open"]
    stair_last_close = stair[-1]["close"]

    stair_total_rise = (
        (stair_last_close - stair_first_open) / stair_first_open
    ) * 100

    stair_bodies = []
    stair_ranges = []

    for candle in stair:
        body = (
            (candle["close"] - candle["open"]) / candle["open"]
        ) * 100

        rng = (
            (candle["high"] - candle["low"]) / candle["low"]
        ) * 100

        stair_bodies.append(abs(body))
        stair_ranges.append(rng)

    avg_stair_body = (
        sum(stair_bodies) / len(stair_bodies)
        if stair_bodies else 0
    )

    breakout_body = (
        (breakout["close"] - breakout["open"]) / breakout["open"]
    ) * 100

    breakout_range = (
        (breakout["high"] - breakout["low"]) / breakout["low"]
    ) * 100

    previous_highs = [c["high"] for c in stair]
    max_previous_high = max(previous_highs) if previous_highs else 0

    breakout_closes_above_previous_high = (
        breakout["close"] > max_previous_high
    )

    breakout_multiplier = (
        abs(breakout_body) / avg_stair_body
        if avg_stair_body > 0 else 0
    )

    return {
        "stair_green": stair_green,
        "stair_higher_closes": stair_higher_closes,
        "stair_higher_lows": stair_higher_lows,
        "stair_total_rise": stair_total_rise,
        "stair_avg_body": avg_stair_body,
        "stair_max_body": max(stair_bodies) if stair_bodies else 0,
        "stair_max_range": max(stair_ranges) if stair_ranges else 0,
        "breakout_body": breakout_body,
        "breakout_range": breakout_range,
        "breakout_multiplier": breakout_multiplier,
        "breakout_closes_above_previous_high":
            breakout_closes_above_previous_high,
        "current_price": breakout["close"],
    }


def qualifies_as_staircase(metrics):
    # 1) First 4 candles = staircase
    staircase_ok = (
        metrics["stair_green"] >= STAIR_STEP_GREEN_MIN
        and metrics["stair_higher_closes"] >= STAIR_STEP_HIGHER_CLOSES_MIN
        and metrics["stair_higher_lows"] >= STAIR_STEP_HIGHER_LOWS_MIN
        and STAIR_STEP_MIN_TOTAL_RISE <= metrics["stair_total_rise"] <= STAIR_STEP_MAX_TOTAL_RISE
        and metrics["stair_max_body"] <= STAIR_STEP_MAX_BODY
        and metrics["stair_max_range"] <= STAIR_STEP_MAX_RANGE
    )

    # 2) Last candle = breakout
    breakout_ok = (
        metrics["breakout_body"] >= BREAKOUT_MIN_BODY
        and metrics["breakout_body"] <= BREAKOUT_MAX_BODY
        and metrics["breakout_multiplier"] >= BREAKOUT_MIN_MULTIPLIER
    )

    if BREAKOUT_MUST_CLOSE_ABOVE_PREVIOUS_HIGHS:
        breakout_ok = (
            breakout_ok
            and metrics["breakout_closes_above_previous_high"]
        )

    return staircase_ok and breakout_ok


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
            "direction": "STAIRCASE",
            "timeframe": "15m",
            "candles": STAIRCASE_CANDLES,
            "stair_green": metrics["stair_green"],
            "stair_higher_closes": metrics["stair_higher_closes"],
            "stair_higher_lows": metrics["stair_higher_lows"],
            "stair_total_rise": metrics["stair_total_rise"],
            "breakout_body": metrics["breakout_body"],
            "breakout_multiplier": metrics["breakout_multiplier"],
            "current_price": metrics["current_price"],
        })

        staircase_last_alert[
            symbol
        ] = now

        staircase_active.add(
            symbol
        )

    staircase_active.intersection_update(
        matched_symbols
    )

    return alerts


# ============================================================
# REAL BYBIT 1-MINUTE CANDLES
# ============================================================

def fetch_1m_candles(
    symbol
):

    data, _ = bybit_get(
        "/v5/market/kline",
        {
            "category":
                "spot",

            "symbol":
                symbol,

            "interval":
                "1",

            "limit":
                21,
        }
    )

    rows = (
        data.get(
            "result",
            {}
        ).get(
            "list",
            []
        )
    )

    candles = []

    for row in rows:

        try:

            candles.append({

                "start":
                    int(
                        row[0]
                    ),

                "open":
                    float(
                        row[1]
                    ),

                "high":
                    float(
                        row[2]
                    ),

                "low":
                    float(
                        row[3]
                    ),

                "close":
                    float(
                        row[4]
                    ),

                "volume":
                    float(
                        row[5]
                    ),

                "turnover":
                    float(
                        row[6]
                    ),
            })

        except (
            TypeError,
            ValueError,
            IndexError
        ):

            continue

    # Bybit returns newest first.

    return candles


# ============================================================
# VOLUME ANALYSIS
# ============================================================

def volume_metrics(
    candles
):

    """
    Compare latest completed
    1-minute candle against
    previous 10 completed minutes.
    """

    if len(candles) < 12:

        return {

            "volume_ratio":
                None,

            "turnover_ratio":
                None,

            "latest_volume":
                None,

            "latest_turnover":
                None,
        }

    latest_completed = (
        candles[1]
    )

    baseline = (
        candles[
            2:12
        ]
    )

    baseline_volumes = [

        candle[
            "volume"
        ]

        for candle
        in baseline

        if (
            candle[
                "volume"
            ]
            > 0
        )
    ]

    baseline_turnover = [

        candle[
            "turnover"
        ]

        for candle
        in baseline

        if (
            candle[
                "turnover"
            ]
            > 0
        )
    ]

    volume_ratio = None

    turnover_ratio = None

    if baseline_volumes:

        normal_volume = (
            statistics.median(
                baseline_volumes
            )
        )

        if normal_volume > 0:

            volume_ratio = (
                latest_completed[
                    "volume"
                ]
                /
                normal_volume
            )

    if baseline_turnover:

        normal_turnover = (
            statistics.median(
                baseline_turnover
            )
        )

        if normal_turnover > 0:

            turnover_ratio = (
                latest_completed[
                    "turnover"
                ]
                /
                normal_turnover
            )

    return {

        "volume_ratio":
            volume_ratio,

        "turnover_ratio":
            turnover_ratio,

        "latest_volume":
            latest_completed[
                "volume"
            ],

        "latest_turnover":
            latest_completed[
                "turnover"
            ],
    }


# ============================================================
# BUY / SELL TRADE PRESSURE
# ============================================================

def fetch_trade_pressure(
    symbol
):

    data, _ = bybit_get(
        "/v5/market/recent-trade",
        {
            "category":
                "spot",

            "symbol":
                symbol,

            "limit":
                60,
        }
    )

    buy_notional = 0.0

    sell_notional = 0.0


    # Ignore trades older than 2 minutes
    # for slower coins.

    cutoff_ms = (
        int(
            time.time()
            * 1000
        )
        -
        (
            2
            * 60
            * 1000
        )
    )

    for trade in (
        data.get(
            "result",
            {}
        ).get(
            "list",
            []
        )
    ):

        try:

            price = float(
                trade.get(
                    "price"
                )
            )

            size = float(
                trade.get(
                    "size"
                )
            )

            side = trade.get(
                "side",
                ""
            )

            trade_time = int(
                trade.get(
                    "time"
                )
            )

        except (
            TypeError,
            ValueError
        ):

            continue

        if (
            trade_time
            < cutoff_ms
        ):

            continue

        notional = (
            price
            * size
        )

        if side == "Buy":

            buy_notional += (
                notional
            )

        elif side == "Sell":

            sell_notional += (
                notional
            )

    total = (
        buy_notional
        + sell_notional
    )

    if total > 0:

        buy_pressure = (
            buy_notional
            / total
        ) * 100

    else:

        buy_pressure = None

    return {

        "buy_pressure":
            buy_pressure,

        "buy_notional":
            buy_notional,

        "sell_notional":
            sell_notional,
    }


# ============================================================
# ORDER BOOK PRESSURE
# ============================================================

def fetch_orderbook_pressure(
    symbol
):

    data, _ = bybit_get(
        "/v5/market/orderbook",
        {
            "category":
                "spot",

            "symbol":
                symbol,

            "limit":
                50,
        }
    )

    result = data.get(
        "result",
        {}
    )

    bid_notional = 0.0

    ask_notional = 0.0

    for row in result.get(
        "b",
        []
    ):

        try:

            price = float(
                row[0]
            )

            size = float(
                row[1]
            )

            bid_notional += (
                price
                * size
            )

        except (
            TypeError,
            ValueError,
            IndexError
        ):

            continue

    for row in result.get(
        "a",
        []
    ):

        try:

            price = float(
                row[0]
            )

            size = float(
                row[1]
            )

            ask_notional += (
                price
                * size
            )

        except (
            TypeError,
            ValueError,
            IndexError
        ):

            continue

    total = (
        bid_notional
        + ask_notional
    )

    if total > 0:

        bid_pressure = (
            bid_notional
            / total
        ) * 100

    else:

        bid_pressure = None

    return {

        "bid_pressure":
            bid_pressure,

        "bid_notional":
            bid_notional,

        "ask_notional":
            ask_notional,
    }


# ============================================================
# DEEP ANALYSIS
# ============================================================

def deep_market_analysis(
    symbol
):

    candles = (
        fetch_1m_candles(
            symbol
        )
    )

    volume = (
        volume_metrics(
            candles
        )
    )

    trades = (
        fetch_trade_pressure(
            symbol
        )
    )

    orderbook = (
        fetch_orderbook_pressure(
            symbol
        )
    )

    return {

        **volume,

        **trades,

        **orderbook,
    }


# ============================================================
# EARLY PUMP SCORE
# ============================================================

def momentum_score(
    change_1m,
    change_3m,
    change_5m
):

    score = 0

    # ------------------------
    # 1 minute
    # ------------------------

    if (
        change_1m
        is not None
    ):

        if change_1m >= 3.0:

            score += 14

        elif change_1m >= 2.0:

            score += 11

        elif change_1m >= 1.5:

            score += 8


    # ------------------------
    # 3 minutes
    # ------------------------

    if (
        change_3m
        is not None
    ):

        if change_3m >= 6.0:

            score += 10

        elif change_3m >= 4.0:

            score += 8

        elif change_3m >= 3.0:

            score += 6


    # ------------------------
    # 5 minutes
    # ------------------------

    if (
        change_5m
        is not None
    ):

        if change_5m >= 8.0:

            score += 7

        elif change_5m >= 6.0:

            score += 6

        elif change_5m >= 4.5:

            score += 4


    # Bonus if all are positive.

    if (
        change_1m is not None
        and
        change_3m is not None
        and
        change_5m is not None
        and
        change_1m > 0
        and
        change_3m > 0
        and
        change_5m > 0
    ):

        score += 4

    return min(
        score,
        35
    )


def volume_score(
    volume_ratio
):

    if volume_ratio is None:

        return 0

    if volume_ratio >= 6:

        return 30

    if volume_ratio >= 4:

        return 25

    if volume_ratio >= 3:

        return 18

    if volume_ratio >= 2:

        return 10

    return 0


def buy_pressure_score(
    buy_pressure
):

    if buy_pressure is None:

        return 0

    if buy_pressure >= 75:

        return 20

    if buy_pressure >= 70:

        return 16

    if buy_pressure >= 65:

        return 12

    if buy_pressure >= 60:

        return 8

    return 0


def orderbook_score(
    bid_pressure
):

    if bid_pressure is None:

        return 0

    if bid_pressure >= 70:

        return 15

    if bid_pressure >= 65:

        return 12

    if bid_pressure >= 60:

        return 9

    if bid_pressure >= 55:

        return 5

    return 0


def calculate_early_score(
    change_1m,
    change_3m,
    change_5m,
    analysis
):

    parts = {

        "momentum":
            momentum_score(
                change_1m,
                change_3m,
                change_5m
            ),

        "volume":
            volume_score(
                analysis.get(
                    "volume_ratio"
                )
            ),

        "buy_pressure":
            buy_pressure_score(
                analysis.get(
                    "buy_pressure"
                )
            ),

        "orderbook":
            orderbook_score(
                analysis.get(
                    "bid_pressure"
                )
            ),
    }

    parts[
        "total"
    ] = sum(
        parts.values()
    )

    return parts


# ============================================================
# EARLY PUMP DETECTOR
# ============================================================

def detect_early_pumps(
    now,
    current_prices,
    deep_cache
):

    global early_active

    candidates = []


    # ========================================================
    # CHEAP FIRST SCAN
    # ========================================================

    for (
        symbol,
        current_price
    ) in current_prices.items():

        change_1m = (
            change_for_symbol(
                now,
                symbol,
                current_price,
                1
            )
        )

        change_3m = (
            change_for_symbol(
                now,
                symbol,
                current_price,
                3
            )
        )

        change_5m = (
            change_for_symbol(
                now,
                symbol,
                current_price,
                5
            )
        )


        suspicious = (

            (
                change_1m is not None
                and
                change_1m
                >=
                EARLY_PREFILTER_1M
            )

            or

            (
                change_3m is not None
                and
                change_3m
                >=
                EARLY_PREFILTER_3M
            )

            or

            (
                change_5m is not None
                and
                change_5m
                >=
                EARLY_PREFILTER_5M
            )
        )


        if not suspicious:

            continue


        # If the normal +10% / 5m
        # PUMP threshold has already
        # been reached, don't send
        # another EARLY alert.

        if (
            change_5m
            is not None
            and
            change_5m
            >= WINDOWS[5]
        ):

            continue


        rank = max(

            change_1m
            or 0,

            change_3m
            or 0,

            change_5m
            or 0
        )


        candidates.append({

            "symbol":
                symbol,

            "current_price":
                current_price,

            "change_1m":
                change_1m,

            "change_3m":
                change_3m,

            "change_5m":
                change_5m,

            "rank":
                rank,
        })


    # Strongest suspicious movers first.

    candidates.sort(
        key=lambda item:
            item["rank"],
        reverse=True
    )


    # Safety cap.

    candidates = (
        candidates[
            :MAX_DEEP_SCANS_PER_RUN
        ]
    )


    alerts = []

    matched_symbols = set()


    # ========================================================
    # DEEP ANALYSIS
    # ========================================================

    for candidate in candidates:

        symbol = candidate[
            "symbol"
        ]

        try:

            if (
                symbol
                not in deep_cache
            ):

                deep_cache[
                    symbol
                ] = (
                    deep_market_analysis(
                        symbol
                    )
                )

            analysis = (
                deep_cache[
                    symbol
                ]
            )

        except Exception as exc:

            print(
                f"Deep analysis failed "
                f"for {symbol}:",
                exc
            )

            continue


        score_parts = (
            calculate_early_score(

                candidate[
                    "change_1m"
                ],

                candidate[
                    "change_3m"
                ],

                candidate[
                    "change_5m"
                ],

                analysis
            )
        )


        total_score = (
            score_parts[
                "total"
            ]
        )


        # ====================================================
        # ALERT
        # ====================================================

        if (
            total_score
            >= EARLY_ALERT_SCORE
        ):

            matched_symbols.add(
                symbol
            )

            previous_alert_time = (
                early_last_alert.get(
                    symbol,
                    0
                )
            )

            if (
                symbol
                not in early_active
                and
                (
                    now
                    - previous_alert_time
                    >=
                    EARLY_COOLDOWN_SECONDS
                )
            ):

                alerts.append({

                    **candidate,

                    **analysis,

                    "score":
                        total_score,

                    "score_parts":
                        score_parts,

                    "strong":
                        (
                            total_score
                            >=
                            EARLY_STRONG_SCORE
                        ),
                })


                early_last_alert[
                    symbol
                ] = now

                early_active.add(
                    symbol
                )


        # ====================================================
        # RE-ARM
        # ====================================================

        elif (
            total_score
            <= EARLY_REARM_SCORE
        ):

            early_active.discard(
                symbol
            )


    candidate_symbols = {

        item[
            "symbol"
        ]

        for item
        in candidates
    }


    for symbol in list(
        early_active
    ):

        if (
            symbol
            not in candidate_symbols
            and
            symbol
            not in matched_symbols
        ):

            early_active.discard(
                symbol
            )


    return alerts


# ============================================================
# START U-TURN WATCH AFTER CRASH
# ============================================================

def register_crash_for_uturn(
    alert,
    now
):

    symbol = alert[
        "symbol"
    ]

    existing = (
        uturn_watch.get(
            symbol
        )
    )


    # First crash alert.

    if existing is None:

        uturn_watch[
            symbol
        ] = {

            "started_at":
                now,

            "reference_price":
                alert[
                    "reference_price"
                ],

            "crash_price":
                alert[
                    "current_price"
                ],

            "lowest_price":
                alert[
                    "current_price"
                ],

            "crash_change":
                alert[
                    "change"
                ],

            "window_label":
                alert[
                    "window_label"
                ],
        }

        return


    # A second crash signal for the
    # same coin can update the episode.

    existing[
        "started_at"
    ] = now


    existing[
        "lowest_price"
    ] = min(

        existing[
            "lowest_price"
        ],

        alert[
            "current_price"
        ]
    )


    if (
        abs(
            alert[
                "change"
            ]
        )
        >
        abs(
            existing[
                "crash_change"
            ]
        )
    ):

        existing[
            "crash_change"
        ] = alert[
            "change"
        ]

        existing[
            "window_label"
        ] = alert[
            "window_label"
        ]

        existing[
            "reference_price"
        ] = alert[
            "reference_price"
        ]


# ============================================================
# U-TURN SCORE
# ============================================================

def uturn_score(
    recovery,
    change_1m,
    change_3m,
    analysis
):

    score = 0


    # ========================================================
    # RECOVERY FROM BOTTOM
    # Maximum = 25
    # ========================================================

    if recovery >= 10:

        score += 25

    elif recovery >= 8:

        score += 22

    elif recovery >= 6:

        score += 18

    elif recovery >= 5:

        score += 15


    # ========================================================
    # 1M MOMENTUM
    # ========================================================

    if (
        change_1m
        is not None
    ):

        if change_1m >= 2:

            score += 7

        elif change_1m >= 1:

            score += 4


    # ========================================================
    # 3M MOMENTUM
    # ========================================================

    if (
        change_3m
        is not None
    ):

        if change_3m >= 4:

            score += 8

        elif change_3m >= 2:

            score += 5


    # ========================================================
    # VOLUME
    # Maximum contribution here = 25
    # ========================================================

    score += min(

        volume_score(
            analysis.get(
                "volume_ratio"
            )
        ),

        25
    )


    # ========================================================
    # BUY PRESSURE
    # ========================================================

    score += (
        buy_pressure_score(
            analysis.get(
                "buy_pressure"
            )
        )
    )


    # ========================================================
    # ORDER BOOK
    # ========================================================

    score += (
        orderbook_score(
            analysis.get(
                "bid_pressure"
            )
        )
    )


    return min(
        score,
        100
    )


# ============================================================
# U-TURN DETECTOR
# ============================================================

def detect_uturns(
    now,
    current_prices,
    deep_cache
):

    alerts = []

    finished = []


    for (
        symbol,
        watch
    ) in list(
        uturn_watch.items()
    ):


        # Stop watching after 3 hours.

        if (
            now
            - watch[
                "started_at"
            ]
            >
            UTURN_WATCH_SECONDS
        ):

            finished.append(
                symbol
            )

            continue


        current_price = (
            current_prices.get(
                symbol
            )
        )


        if not current_price:

            continue


        # Keep updating the crash bottom.

        if (
            current_price
            <
            watch[
                "lowest_price"
            ]
        ):

            watch[
                "lowest_price"
            ] = current_price


        recovery = (
            percent_change(

                current_price,

                watch[
                    "lowest_price"
                ]
            )
        )


        if recovery is None:

            continue


        change_1m = (
            change_for_symbol(
                now,
                symbol,
                current_price,
                1
            )
        )


        change_3m = (
            change_for_symbol(
                now,
                symbol,
                current_price,
                3
            )
        )


        # ====================================================
        # CHEAP U-TURN FILTER
        # ====================================================

        if (

            recovery
            < UTURN_MIN_RECOVERY

            or

            change_1m
            is None

            or

            change_3m
            is None

            or

            change_1m
            <
            UTURN_MIN_1M_CHANGE

            or

            change_3m
            <
            UTURN_MIN_3M_CHANGE
        ):

            continue


        # ====================================================
        # DEEP ANALYSIS
        # ====================================================

        try:

            if (
                symbol
                not in deep_cache
            ):

                deep_cache[
                    symbol
                ] = (
                    deep_market_analysis(
                        symbol
                    )
                )

            analysis = (
                deep_cache[
                    symbol
                ]
            )

        except Exception as exc:

            print(
                f"U-turn analysis "
                f"failed for {symbol}:",
                exc
            )

            continue


        score = (
            uturn_score(
                recovery,
                change_1m,
                change_3m,
                analysis
            )
        )


        if (
            score
            <
            UTURN_ALERT_SCORE
        ):

            continue


        alerts.append({

            "symbol":
                symbol,

            "current_price":
                current_price,

            "lowest_price":
                watch[
                    "lowest_price"
                ],

            "reference_price":
                watch[
                    "reference_price"
                ],

            "original_crash":
                watch[
                    "crash_change"
                ],

            "crash_window":
                watch[
                    "window_label"
                ],

            "recovery":
                recovery,

            "change_1m":
                change_1m,

            "change_3m":
                change_3m,

            "volume_ratio":
                analysis.get(
                    "volume_ratio"
                ),

            "buy_pressure":
                analysis.get(
                    "buy_pressure"
                ),

            "bid_pressure":
                analysis.get(
                    "bid_pressure"
                ),

            "score":
                score,
        })


        # Only one U-turn alert
        # for this crash episode.

        finished.append(
            symbol
        )


    for symbol in finished:

        uturn_watch.pop(
            symbol,
            None
        )


    return alerts


# ============================================================
# SIGNAL PERFORMANCE JOURNAL
# ============================================================

def record_signal(
    now,
    kind,
    symbol,
    price,
    extra=None
):

    entry = {

        "timestamp":
            now,

        "time_uk":
            datetime.fromtimestamp(
                now,
                UK_TIMEZONE
            ).strftime(
                "%d %b %Y %H:%M:%S"
            ),

        "kind":
            kind,

        "symbol":
            symbol,

        "price":
            price,

        "extra":
            extra or {},

        "outcomes":
            {},

        "max_price_seen":
            price,

        "min_price_seen":
            price,

        "max_change_seen":
            0.0,

        "min_change_seen":
            0.0,
    }


    signal_log.append(
        entry
    )


    if (
        len(signal_log)
        >
        MAX_SIGNAL_LOG
    ):

        remove_count = (
            len(signal_log)
            -
            MAX_SIGNAL_LOG
        )

        del signal_log[
            :remove_count
        ]


def update_signal_outcomes(
    now,
    current_prices
):

    horizons = [
        5,
        15,
        30,
        60
    ]


    for entry in signal_log:

        current_price = (
            current_prices.get(
                entry[
                    "symbol"
                ]
            )
        )


        if not current_price:

            continue


        live_change = (
            percent_change(

                current_price,

                entry[
                    "price"
                ]
            )
        )


        entry[
            "max_price_seen"
        ] = max(

            entry[
                "max_price_seen"
            ],

            current_price
        )


        entry[
            "min_price_seen"
        ] = min(

            entry[
                "min_price_seen"
            ],

            current_price
        )


        if live_change is not None:

            entry[
                "max_change_seen"
            ] = max(

                entry[
                    "max_change_seen"
                ],

                live_change
            )


            entry[
                "min_change_seen"
            ] = min(

                entry[
                    "min_change_seen"
                ],

                live_change
            )


        # Record exact checkpoints.

        for minutes in horizons:

            key = (
                f"{minutes}m"
            )


            if (
                key
                in entry[
                    "outcomes"
                ]
            ):

                continue


            if (
                now
                - entry[
                    "timestamp"
                ]
                >=
                minutes * 60
            ):

                change = (
                    percent_change(

                        current_price,

                        entry[
                            "price"
                        ]
                    )
                )


                entry[
                    "outcomes"
                ][
                    key
                ] = {

                    "change":
                        (
                            round(
                                change,
                                4
                            )

                            if (
                                change
                                is not None
                            )

                            else None
                        ),

                    "price":
                        current_price,

                    "captured_at":
                        now,
                }


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
            "TELEGRAM_BOT_TOKEN "
            "is missing"
        )


    if not TELEGRAM_CHAT_ID:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID "
            "is missing"
        )


    url = (

        f"{TELEGRAM_API_BASE}/"

        f"bot"
        f"{TELEGRAM_BOT_TOKEN}/"

        f"sendMessage"
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
        and
        button_text
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


    if (
        response.status_code
        != 200
    ):

        raise RuntimeError(
            "Telegram error: "
            +
            response.text[
                :500
            ]
        )


# ============================================================
# NORMAL PUMP / CRASH TELEGRAM
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

        if (
            direction
            ==
            "PUMP"
        )

        else

        "🔻"
    )


    change = alert[
        "change"
    ]


    change_text = (

        f"+{change:.2f}%"

        if (
            change
            >= 0
        )

        else

        f"{change:.2f}%"
    )


    message = (

        f"{icon} "
        f"{coin} "
        f"{direction}\n\n"

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
            f"📈 Open "
            f"{coin} "
            f"on Bybit"
    )


# ============================================================
# STAIRCASE TELEGRAM
# ============================================================

def send_staircase_signal(alert):
    symbol = alert["symbol"]
    coin = coin_name(symbol)

    message = (
        f"🪜 {coin} STAIR + BREAKOUT\n\n"
        f"Timeframe: {alert['timeframe']}\n"
        f"Pattern: 4 stair candles + 1 breakout candle\n\n"
        f"Stair rise: +{alert['stair_total_rise']:.2f}%\n"
        f"Green stair candles: {alert['stair_green']}/4\n"
        f"Higher closes: {alert['stair_higher_closes']}/3\n"
        f"Higher lows: {alert['stair_higher_lows']}/3\n\n"
        f"Breakout candle body: +{alert['breakout_body']:.2f}%\n"
        f"Breakout strength: {alert['breakout_multiplier']:.2f}x stair average\n"
        f"Current: {format_price(alert['current_price'])}\n\n"
        f"Detected: {current_uk_time()} UK"
    )

    send_telegram(
        message,
        button_url=bybit_spot_url(symbol),
        button_text=f"📈 Open {coin} on Bybit"
    )

# ============================================================
# EARLY PUMP TELEGRAM
# ============================================================

def send_early_pump_signal(
    alert
):

    symbol = alert[
        "symbol"
    ]

    coin = coin_name(
        symbol
    )


    if alert[
        "strong"
    ]:

        title_icon = "🔥"

        title_text = (
            "STRONG EARLY PUMP"
        )

    else:

        title_icon = "⚡"

        title_text = (
            "EARLY PUMP"
        )


    message = (

        f"{title_icon} "
        f"{coin} "
        f"{title_text}\n\n"

        f"Score: "
        f"{alert['score']}/100\n\n"

        f"1m: "
        f"{fmt_optional_percent(alert['change_1m'])}\n"

        f"3m: "
        f"{fmt_optional_percent(alert['change_3m'])}\n"

        f"5m: "
        f"{fmt_optional_percent(alert['change_5m'])}\n\n"

        f"Volume: "
        f"{fmt_optional_ratio(alert.get('volume_ratio'))} "
        f"normal\n"

        f"Buy pressure: "
        f"{fmt_optional_pressure(alert.get('buy_pressure'))}\n"

        f"Order-book bids: "
        f"{fmt_optional_pressure(alert.get('bid_pressure'))}\n\n"

        f"Current: "
        f"{format_price(alert['current_price'])}\n\n"

        f"Detected: "
        f"{current_uk_time()} UK\n\n"

        f"Early unusual activity detected."
    )


    send_telegram(

        message,

        button_url=
            bybit_spot_url(
                symbol
            ),

        button_text=
            f"📈 Open "
            f"{coin} "
            f"on Bybit"
    )


# ============================================================
# U-TURN TELEGRAM
# ============================================================

def send_uturn_signal(
    alert
):

    symbol = alert[
        "symbol"
    ]

    coin = coin_name(
        symbol
    )


    message = (

        f"🔄 "
        f"{coin} "
        f"U-TURN\n\n"

        f"Score: "
        f"{alert['score']}/100\n\n"

        f"Original crash: "
        f"{alert['original_crash']:.2f}% "
        f"({alert['crash_window']})\n"

        f"Bottom: "
        f"{format_price(alert['lowest_price'])}\n"

        f"Recovery from bottom: "
        f"+{alert['recovery']:.2f}%\n\n"

        f"1m: "
        f"{fmt_optional_percent(alert['change_1m'])}\n"

        f"3m: "
        f"{fmt_optional_percent(alert['change_3m'])}\n"

        f"Volume: "
        f"{fmt_optional_ratio(alert.get('volume_ratio'))} "
        f"normal\n"

        f"Buy pressure: "
        f"{fmt_optional_pressure(alert.get('buy_pressure'))}\n"

        f"Order-book bids: "
        f"{fmt_optional_pressure(alert.get('bid_pressure'))}\n\n"

        f"Current: "
        f"{format_price(alert['current_price'])}\n\n"

        f"Detected: "
        f"{current_uk_time()} UK\n\n"

        f"Possible crash reversal detected."
    )


    send_telegram(

        message,

        button_url=
            bybit_spot_url(
                symbol
            ),

        button_text=
            f"📈 Open "
            f"{coin} "
            f"on Bybit"
    )


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

            "1m":
                False,

            "3m":
                False,

            "5m":
                False,

            "10m":
                False,

            "15m":
                False,

            "24h":
                True,

            "early_pump":
                False,

            "uturn":
                False,

            "staircase_15m":
                False,
        }


    oldest_age = (
        now
        - history[
            0
        ][
            "timestamp"
        ]
    )


    return {

        "1m":
            (
                oldest_age
                >=
                1 * 60
            ),

        "3m":
            (
                oldest_age
                >=
                3 * 60
            ),

        "5m":
            (
                oldest_age
                >=
                5 * 60
            ),

        "10m":
            (
                oldest_age
                >=
                10 * 60
            ),

        "15m":
            (
                oldest_age
                >=
                15 * 60
            ),

        "24h":
            True,

        "early_pump":
            (
                oldest_age
                >=
                5 * 60
            ),

        "uturn":
            (
                oldest_age
                >=
                3 * 60
            ),

        "staircase_15m":
            (
                oldest_age
                >=
                90 * 60
            ),
    }


# ============================================================
# MAIN DETECTOR
# ============================================================

def run_detector():

    global first_run

    global last_success

    global last_market_count


    with lock:

        now = time.time()


        # Cache deeper analysis so
        # Early Pump and U-Turn do
        # not request the same coin twice.

        deep_cache = {}


        (
            current_prices,
            daily_data,
            endpoint
        ) = fetch_bybit_prices()


        # ====================================================
        # NORMAL PUMP / CRASH
        # ====================================================

        alerts = (
            detect_signals(
                now,
                current_prices
            )
        )


        alerts.extend(

            detect_24h_signals(
                daily_data
            )

        )


        # ====================================================
        # CRASH -> U-TURN WATCH
        # ====================================================

        for alert in alerts:

            if (
                alert[
                    "direction"
                ]
                ==
                "CRASH"
            ):

                register_crash_for_uturn(
                    alert,
                    now
                )


        # ====================================================
        # EARLY PUMP
        # ====================================================

        early_alerts = (
            detect_early_pumps(
                now,
                current_prices,
                deep_cache
            )
        )


        # ====================================================
        # U-TURN
        # ====================================================

        uturn_alerts = (
            detect_uturns(
                now,
                current_prices,
                deep_cache
            )
        )


        # ====================================================
        # SAVE CURRENT SNAPSHOT
        # ====================================================

        save_snapshot(
            now,
            current_prices
        )


        # ====================================================
        # STAIRCASE
        # ====================================================

        staircase_alerts = (
            detect_staircases(
                now
            )
        )


        # ====================================================
        # UPDATE OLD SIGNAL RESULTS
        # ====================================================

        update_signal_outcomes(
            now,
            current_prices
        )


        last_success = now

        last_market_count = (
            len(
                current_prices
            )
        )


        # ====================================================
        # STARTUP MESSAGE
        # ====================================================

        if first_run:

            send_telegram(

                "✅ SIGNAL WATCH V2 ONLINE\n\n"

                f"Bybit connection: OK\n"

                f"USDT Spot markets: "
                f"{len(current_prices)}\n\n"

                "Large Pump / Crash:\n"

                "5 min: ±10%\n"

                "10 min: ±15%\n"

                "15 min: ±20%\n"

                "24 hours: ±30%\n\n"

                "⚡ Early Pump: enabled\n"

                "🔄 U-Turn: enabled\n"

                "🪜 Staircase: enabled\n\n"

                "1/3/5 minute history "
                "warms up during the "
                "first few minutes.\n"

                "Staircase needs roughly "
                "75–90 minutes after "
                "a restart."
            )


            first_run = False


        # ====================================================
        # SEND NORMAL SIGNALS
        # ====================================================

        for alert in alerts:

            try:

                send_signal(
                    alert
                )


                record_signal(

                    now,

                    alert[
                        "direction"
                    ],

                    alert[
                        "symbol"
                    ],

                    alert[
                        "current_price"
                    ],

                    {

                        "window":
                            alert[
                                "window_label"
                            ],

                        "change":
                            alert[
                                "change"
                            ],
                    }
                )


            except Exception as exc:

                print(
                    "Telegram normal "
                    "signal error:",
                    exc
                )


        # ====================================================
        # SEND EARLY PUMP SIGNALS
        # ====================================================

        for alert in early_alerts:

            try:

                send_early_pump_signal(
                    alert
                )


                record_signal(

                    now,

                    "EARLY_PUMP",

                    alert[
                        "symbol"
                    ],

                    alert[
                        "current_price"
                    ],

                    {

                        "score":
                            alert[
                                "score"
                            ],

                        "change_1m":
                            alert[
                                "change_1m"
                            ],

                        "change_3m":
                            alert[
                                "change_3m"
                            ],

                        "change_5m":
                            alert[
                                "change_5m"
                            ],

                        "volume_ratio":
                            alert.get(
                                "volume_ratio"
                            ),

                        "buy_pressure":
                            alert.get(
                                "buy_pressure"
                            ),

                        "bid_pressure":
                            alert.get(
                                "bid_pressure"
                            ),
                    }
                )


            except Exception as exc:

                print(
                    "Telegram early-pump "
                    "error:",
                    exc
                )


        # ====================================================
        # SEND U-TURN SIGNALS
        # ====================================================

        for alert in uturn_alerts:

            try:

                send_uturn_signal(
                    alert
                )


                record_signal(

                    now,

                    "UTURN",

                    alert[
                        "symbol"
                    ],

                    alert[
                        "current_price"
                    ],

                    {

                        "score":
                            alert[
                                "score"
                            ],

                        "recovery":
                            alert[
                                "recovery"
                            ],

                        "original_crash":
                            alert[
                                "original_crash"
                            ],
                    }
                )


            except Exception as exc:

                print(
                    "Telegram U-turn "
                    "error:",
                    exc
                )


        # ====================================================
        # SEND STAIRCASE SIGNALS
        # ====================================================

        for alert in staircase_alerts:

            try:

                send_staircase_signal(
                    alert
                )


                record_signal(

                    now,

                    "STAIRCASE",

                    alert[
                        "symbol"
                    ],

                    alert[
                        "current_price"
                    ],

                    {

                        "rise":
                            alert[
                                "total_rise"
                            ],
                    }
                )


            except Exception as exc:

                print(
                    "Telegram staircase "
                    "error:",
                    exc
                )


        # ====================================================
        # RETURN STATUS
        # ====================================================

        return {

            "success":
                True,

            "markets":
                len(
                    current_prices
                ),

            "endpoint":
                endpoint,

            "history_snapshots":
                len(
                    history
                ),

            "pump_crash_signals":
                len(
                    alerts
                ),

            "early_pump_signals":
                len(
                    early_alerts
                ),

            "uturn_signals":
                len(
                    uturn_alerts
                ),

            "staircase_signals":
                len(
                    staircase_alerts
                ),

            "uturn_watches":
                len(
                    uturn_watch
                ),

            "deep_scans":
                len(
                    deep_cache
                ),

            "warmup":
                get_warmup_status(
                    now
                ),
        }


# ============================================================
# HOME / STATUS PAGE
# ============================================================

@app.route("/")
def home():

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
            "Signal Watch V2",

        "status":
            "online",

        "markets_last_seen":
            last_market_count,

        "history_snapshots":
            len(
                history
            ),

        "last_success_uk":
            last_run_text,

        "warmup":
            get_warmup_status(),


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


        "early_pump": {

            "status":
                "enabled",

            "prefilter_1m":
                (
                    f"+"
                    f"{EARLY_PREFILTER_1M}%"
                ),

            "prefilter_3m":
                (
                    f"+"
                    f"{EARLY_PREFILTER_3M}%"
                ),

            "prefilter_5m":
                (
                    f"+"
                    f"{EARLY_PREFILTER_5M}%"
                ),

            "alert_score":
                EARLY_ALERT_SCORE,

            "strong_score":
                EARLY_STRONG_SCORE,

            "max_deep_scans_per_run":
                MAX_DEEP_SCANS_PER_RUN,
        },


        "uturn": {

            "status":
                "enabled",

            "watch_hours":
                (
                    UTURN_WATCH_SECONDS
                    / 3600
                ),

            "minimum_recovery":
                (
                    f"{UTURN_MIN_RECOVERY}%"
                ),

            "alert_score":
                UTURN_ALERT_SCORE,

            "active_watches":
                len(
                    uturn_watch
                ),
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


        "signal_log_entries":
            len(
                signal_log
            ),
    })


# ============================================================
# RUN
# ============================================================

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

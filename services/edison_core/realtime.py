"""
Real-Time Data Service for EDISON
Provides current time, date, weather, and news capabilities.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class RealTimeDataService:
    """Provides real-time data: time, date, weather, news."""

    def __init__(self):
        self._weather_cache: Dict[str, Any] = {}
        self._cache_ttl = 600  # 10 min cache for weather
        logger.info("✓ Real-time data service initialized")

    # ── Time & Date ──────────────────────────────────────────────────────

    def get_current_datetime(self, timezone_name: str = "local") -> Dict[str, Any]:
        """Return current date, time, day-of-week, and unix timestamp."""
        try:
            import zoneinfo
        except ImportError:
            zoneinfo = None

        now = datetime.now()
        utc_now = datetime.now(timezone.utc)

        # Try to resolve named timezone
        tz_label = "local"
        if timezone_name and timezone_name != "local" and zoneinfo:
            try:
                tz = zoneinfo.ZoneInfo(timezone_name)
                now = datetime.now(tz)
                tz_label = timezone_name
            except Exception:
                pass  # fall back to local

        return {
            "ok": True,
            "data": {
                "date": now.strftime("%B %d, %Y"),
                "time": now.strftime("%I:%M %p"),
                "time_24h": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "timezone": tz_label,
                "unix_timestamp": int(utc_now.timestamp()),
                "iso": now.isoformat(),
                "year": now.year,
                "month": now.month,
                "day": now.day,
            },
        }

    # ── Weather ──────────────────────────────────────────────────────────

    def get_weather(self, location: str) -> Dict[str, Any]:
        """
        Fetch current weather for a location using wttr.in (free, no API key).
        Falls back to OpenWeatherMap-compatible free tier if available.
        """
        import time as _time

        cache_key = location.lower().strip()
        cached = self._weather_cache.get(cache_key)
        if cached and (_time.time() - cached["_ts"]) < self._cache_ttl:
            return {"ok": True, "data": cached["data"]}

        # Try wttr.in first (free, no API key)
        data = self._fetch_wttr(location)
        if data:
            self._weather_cache[cache_key] = {"data": data, "_ts": _time.time()}
            return {"ok": True, "data": data}

        return {"ok": False, "error": f"Could not fetch weather for '{location}'"}

    def _fetch_wttr(self, location: str) -> Optional[Dict[str, Any]]:
        """Fetch weather from wttr.in JSON API."""
        import requests

        try:
            url = f"https://wttr.in/{requests.utils.quote(location)}?format=j1"
            resp = requests.get(url, timeout=8, headers={"User-Agent": "EDISON-AI/1.0"})
            resp.raise_for_status()
            raw = resp.json()

            current = raw.get("current_condition", [{}])[0]
            area = raw.get("nearest_area", [{}])[0]

            area_name = area.get("areaName", [{}])[0].get("value", location)
            region = area.get("region", [{}])[0].get("value", "")
            country = area.get("country", [{}])[0].get("value", "")

            weather_desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")

            # Build forecast summary (next 3 days)
            forecast = []
            for day in raw.get("weather", [])[:3]:
                forecast.append({
                    "date": day.get("date", ""),
                    "high_f": day.get("maxtempF", ""),
                    "low_f": day.get("mintempF", ""),
                    "high_c": day.get("maxtempC", ""),
                    "low_c": day.get("mintempC", ""),
                    "description": day.get("hourly", [{}])[4].get("weatherDesc", [{}])[0].get("value", "")
                    if len(day.get("hourly", [])) > 4 else "",
                })

            return {
                "location": f"{area_name}, {region}, {country}".strip(", "),
                "temperature_f": f"{current.get('temp_F', '?')}°F",
                "temperature_c": f"{current.get('temp_C', '?')}°C",
                "feels_like_f": f"{current.get('FeelsLikeF', '?')}°F",
                "feels_like_c": f"{current.get('FeelsLikeC', '?')}°C",
                "condition": weather_desc,
                "humidity": f"{current.get('humidity', '?')}%",
                "wind_mph": current.get("windspeedMiles", "?"),
                "wind_dir": current.get("winddir16Point", "?"),
                "visibility_miles": current.get("visibilityMiles", "?"),
                "uv_index": current.get("uvIndex", "?"),
                "pressure_mb": current.get("pressure", "?"),
                "forecast": forecast,
            }

        except Exception as e:
            logger.warning(f"wttr.in fetch failed for '{location}': {e}")
            return None

    # ── News ─────────────────────────────────────────────────────────────

    def get_news(self, topic: str = "top news", max_results: int = 8) -> Dict[str, Any]:
        """
        Fetch current news headlines using DuckDuckGo News API.
        """
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            ddgs = DDGS()
            results = list(ddgs.news(topic, max_results=max_results))

            articles = []
            for r in results:
                articles.append({
                    "title": r.get("title", ""),
                    "source": r.get("source", ""),
                    "url": r.get("url", r.get("link", "")),
                    "date": r.get("date", r.get("published", "")),
                    "snippet": r.get("body", r.get("excerpt", "")),
                })

            return {
                "ok": True,
                "data": {
                    "topic": topic,
                    "count": len(articles),
                    "articles": articles,
                },
            }

        except Exception as e:
            logger.error(f"News fetch failed for '{topic}': {e}")
            return {"ok": False, "error": f"Could not fetch news: {str(e)}"}

    # ── Convenience: detect & answer real-time queries inline ───────────

    @staticmethod
    def is_realtime_query(message: str) -> Optional[str]:
        """
        Detect if message is a simple real-time query.
        Returns the query type or None.
        """
        msg = message.lower().strip()

        time_patterns = [
            "what time is it", "what's the time", "whats the time",
            "current time", "tell me the time", "what time",
            "do you know the time", "do you know what time",
        ]
        date_patterns = [
            "what's today's date", "whats todays date", "what is today's date",
            "what day is it", "what's the date", "whats the date",
            "today's date", "todays date", "current date", "what date is it",
            "what is the date", "what day is today", "what is today",
        ]
        weather_patterns = [
            "weather in", "weather for", "what's the weather",
            "whats the weather", "how's the weather", "hows the weather",
            "forecast for", "forecast in", "temperature in",
            "is it raining", "is it going to rain", "will it rain",
            "is it cold", "is it hot", "is it snowing",
        ]
        news_patterns = [
            "today's news", "todays news", "latest news", "current news",
            "top news", "news today", "what's in the news",
            "what's happening", "whats happening", "breaking news",
            "news about", "headlines",
        ]

        for p in time_patterns:
            if p in msg:
                return "time"
        for p in date_patterns:
            if p in msg:
                return "date"
        for p in weather_patterns:
            if p in msg:
                return "weather"
        for p in news_patterns:
            if p in msg:
                return "news"

        return None

    @staticmethod
    def extract_location(message: str) -> str:
        """Extract location from a weather query."""
        msg = message.lower()
        # Try common patterns
        for prefix in [
            "weather in ", "weather for ", "forecast for ", "forecast in ",
            "temperature in ", "is it raining in ", "is it cold in ",
            "is it hot in ", "is it snowing in ", "will it rain in ",
        ]:
            if prefix in msg:
                loc = msg.split(prefix, 1)[1]
                # Clean up trailing question marks, periods, etc
                loc = loc.rstrip("?.! ")
                # Remove trailing filler words
                for suffix in [" today", " right now", " currently", " tomorrow",
                               " this week", " tonight", " this weekend"]:
                    if loc.endswith(suffix):
                        loc = loc[: -len(suffix)].strip()
                return loc.title() if loc else "New York"

        return "New York"  # default if unable to parse

    @staticmethod
    def extract_news_topic(message: str) -> str:
        """Extract topic from a news query."""
        msg = message.lower()
        for prefix in ["news about ", "news on ", "news regarding ",
                        "headlines about ", "headlines on "]:
            if prefix in msg:
                topic = msg.split(prefix, 1)[1].rstrip("?.! ")
                return topic if topic else "top news today"
        return "top news today"

    def build_realtime_context(self, message: str) -> Optional[str]:
        """
        If the message is a real-time query, return a context string
        that should be injected into the system prompt so the LLM can
        answer accurately. Returns None if not a real-time query.
        """
        qtype = self.is_realtime_query(message)
        if not qtype:
            return None

        if qtype == "time":
            info = self.get_current_datetime()
            d = info["data"]
            return (
                f"REAL-TIME DATA — The current date and time is: "
                f"{d['day_of_week']}, {d['date']} at {d['time']} ({d['time_24h']}) {d['timezone']} time. "
                f"Use this exact information to answer the user's question."
            )

        if qtype == "date":
            info = self.get_current_datetime()
            d = info["data"]
            return (
                f"REAL-TIME DATA — Today is {d['day_of_week']}, {d['date']}. "
                f"The year is {d['year']}. Use this exact information."
            )

        if qtype == "weather":
            location = self.extract_location(message)
            info = self.get_weather(location)
            if info.get("ok"):
                w = info["data"]
                forecast_str = ""
                if w.get("forecast"):
                    parts = []
                    for f in w["forecast"][:3]:
                        parts.append(f"{f['date']}: {f.get('description', '?')}, High {f['high_f']}, Low {f['low_f']}")
                    forecast_str = " | 3-Day Forecast: " + "; ".join(parts)
                return (
                    f"REAL-TIME WEATHER DATA for {w['location']} — "
                    f"Temperature: {w['temperature_f']} ({w['temperature_c']}), "
                    f"Feels like: {w['feels_like_f']}, "
                    f"Condition: {w['condition']}, "
                    f"Humidity: {w['humidity']}, "
                    f"Wind: {w['wind_mph']} mph {w['wind_dir']}, "
                    f"UV Index: {w['uv_index']}"
                    f"{forecast_str}. "
                    f"Use this REAL-TIME data to answer accurately."
                )
            return f"Weather data unavailable for {location}. Let the user know."

        if qtype == "news":
            topic = self.extract_news_topic(message)
            info = self.get_news(topic)
            if info.get("ok"):
                articles = info["data"]["articles"]
                parts = []
                for a in articles[:6]:
                    parts.append(f"• {a['title']} ({a['source']}, {a.get('date', 'recent')}): {a.get('snippet', '')[:150]}")
                return (
                    f"REAL-TIME NEWS ({topic}) — Latest headlines:\n"
                    + "\n".join(parts)
                    + "\nUse these CURRENT headlines to answer. Cite sources."
                )
            return f"News data unavailable for {topic}."

        return None

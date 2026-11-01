"""FastAPI-сервіс: HTTP-ендпоінти + планувальник кожні 5 хвилин."""
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from data_fetcher import fetch_klines
from features import build_features
from model import train_model
from scheduler import run_cycle, STATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("app")

SYMBOL = "BTCUSDT"
INTERVAL = "5m"

scheduler = BackgroundScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_cycle(SYMBOL)
    scheduler.add_job(
        lambda: run_cycle(SYMBOL),
        "interval",
        minutes=5,
        id="cycle",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    log.info("Scheduler started: кожні 5 хвилин для %s", SYMBOL)
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(title="5m Candle Predictor", lifespan=lifespan)


@app.get("/")
def root():
    return {
        "service": "5m Candle Predictor",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "endpoints": ["/health", "/signal", "/cycle", "/train"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if STATE["last_error"] is None else "degraded",
        "last_update": STATE["last_update"],
        "last_error": STATE["last_error"],
        "model_loaded": STATE["last_signal"]["source"] == "model"
        if STATE["last_signal"] else False,
    }


@app.get("/signal")
def signal():
    if STATE["last_signal"] is None:
        result = run_cycle(SYMBOL)
        return result or {"error": "Сигнал ще не готовий, спробуйте за хвилину"}
    return STATE["last_signal"]


@app.post("/cycle")
def trigger_cycle():
    result = run_cycle(SYMBOL)
    return result or {"error": STATE["last_error"] or "Невідома помилка"}


@app.post("/train")
def train():
    df = fetch_klines(SYMBOL, INTERVAL, limit=1000)
    feats = build_features(df)
    return train_model(feats)

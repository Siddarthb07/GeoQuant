from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.database import init_db
from app.routers.api import router as api_router
from app.services.self_learning import start_self_learning_loop
from app.services.train_service import launch_training, models_exist


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(api_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    start_self_learning_loop()
    if not models_exist():
        launch_training(force=False)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    context = {"app_name": settings.app_name}
    # Compatible with both newer and older Starlette/FastAPI template signatures.
    try:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context,
        )
    except TypeError:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                **context,
            },
        )

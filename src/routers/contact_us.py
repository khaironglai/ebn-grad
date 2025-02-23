from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="src/templates")

@router.get("/contact_us", response_class=HTMLResponse)
async def render_contact_us(request: Request):
    return templates.TemplateResponse("contact_us.html", {"request": request})
import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from uvicorn import run as app_run

from us_visa.constants import APP_HOST, APP_PORT
from us_visa.pipeline.prediction_pipeline import USvisaData, USvisaClassifier


app = FastAPI(
    title="US Visa Approval Prediction API",
    version="1.0.0",
    description="Production API for US visa approval prediction.",
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

TEMPLATE_PATH = os.path.join("templates", "usvisa.html")

def render_html() -> HTMLResponse:
    """Read and return the main HTML template as a direct HTMLResponse."""
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# For production, replace "*" with your actual frontend/domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# API REQUEST SCHEMA
# ---------------------------------------------------------------------

class VisaApplication(BaseModel):
    continent: str
    education_of_employee: str
    has_job_experience: str
    requires_job_training: str

    no_of_employees: int = Field(..., ge=0)

    region_of_employment: str

    prevailing_wage: float = Field(..., ge=0)

    unit_of_wage: str
    full_time_position: str

    company_age: int = Field(..., ge=0, le=500)


# ---------------------------------------------------------------------
# HTML FORM DATA
# ---------------------------------------------------------------------

class DataForm:

    def __init__(self, request: Request):

        self.request = request

        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None
        self.company_age: Optional[str] = None

    async def get_usvisa_data(self):

        form = await self.request.form()

        self.continent = form.get("continent")
        self.education_of_employee = form.get("education_of_employee")
        self.has_job_experience = form.get("has_job_experience")
        self.requires_job_training = form.get("requires_job_training")
        self.no_of_employees = form.get("no_of_employees")
        self.region_of_employment = form.get("region_of_employment")
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get("full_time_position")
        self.company_age = form.get("company_age")


# ---------------------------------------------------------------------
# COMMON PREDICTION FUNCTION
# ---------------------------------------------------------------------

def predict_application(data: VisaApplication):

    usvisa_data = USvisaData(
        continent=data.continent,
        education_of_employee=data.education_of_employee,
        has_job_experience=data.has_job_experience,
        requires_job_training=data.requires_job_training,
        no_of_employees=data.no_of_employees,
        region_of_employment=data.region_of_employment,
        prevailing_wage=data.prevailing_wage,
        unit_of_wage=data.unit_of_wage,
        full_time_position=data.full_time_position,
        company_age=data.company_age,
    )

    dataframe = usvisa_data.get_usvisa_input_data_frame()

    predictor = USvisaClassifier()

    prediction = predictor.predict(
        dataframe=dataframe
    )[0]

    prediction_value = int(prediction)

    if prediction_value == 1:
        prediction_text = "Visa-Approved"
    else:
        prediction_text = "Visa Not-Approved"

    return {
        "prediction": prediction_text,
        "prediction_value": prediction_value,
    }


# ---------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------

@app.get(
    "/health",
    tags=["system"],
)
async def health_check():

    return {
        "status": "ok"
    }


# ---------------------------------------------------------------------
# WEB UI
# ---------------------------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["web"],
)
async def index(request: Request):
    return render_html()


# ---------------------------------------------------------------------
# JSON PREDICTION API
# ---------------------------------------------------------------------

@app.post(
    "/predict",
    tags=["prediction"],
)
async def predict_api(data: VisaApplication):

    try:

        result = predict_application(data)

        return {
            "status": True,
            **result,
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "error": str(e),
            },
        )


# ---------------------------------------------------------------------
# HTML FORM PREDICTION
# ---------------------------------------------------------------------

@app.post(
    "/",
    tags=["web"],
)
async def predict_route(request: Request):

    try:

        form = DataForm(request)

        await form.get_usvisa_data()

        data = VisaApplication(
            continent=form.continent,
            education_of_employee=form.education_of_employee,
            has_job_experience=form.has_job_experience,
            requires_job_training=form.requires_job_training,

            no_of_employees=int(
                form.no_of_employees
            ),

            region_of_employment=form.region_of_employment,

            prevailing_wage=float(
                form.prevailing_wage
            ),

            unit_of_wage=form.unit_of_wage,
            full_time_position=form.full_time_position,

            company_age=int(
                form.company_age
            ),
        )

        result = predict_application(data)

        return render_html()

    except Exception as e:

        return JSONResponse(
            status_code=400,
            content={
                "status": False,
                "error": str(e),
            },
        )


@app.get(
    "/favicon.ico",
    include_in_schema=False,
)
async def favicon():

    return RedirectResponse(
        url="/static/favicon.ico"
    )


# ---------------------------------------------------------------------
# LOCAL DEVELOPMENT
# ---------------------------------------------------------------------

if __name__ == "__main__":

    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT,
    )
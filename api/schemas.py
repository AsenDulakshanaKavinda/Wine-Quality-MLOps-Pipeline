from pydantic import BaseModel, Field
from api.constants import *

class InputSchema(BaseModel):
    fixed_acidity: float = Field(..., ge=min_fixed_acidity, le=max_fixed_acidity)
    volatile_acidity: float = Field(..., ge=min_volatile_acidity, le=max_volatile_acidity)
    citric_acid: float = Field(..., ge=min_citric_acid, le=max_citric_acid)
    residual_sugar: float = Field(..., ge=min_residual_sugar, le=max_residual_sugar)
    chlorides: float = Field(..., ge=min_chlorides, le=max_chlorides)
    free_sulfur_dioxide: float = Field(..., ge=min_free_sulfur_dioxide, le=max_free_sulfur_dioxide)
    total_sulfur_dioxide: float = Field(..., ge=min_total_sulfur_dioxide, le=max_total_sulfur_dioxide)
    density: float = Field(..., ge=min_density, le=max_density)
    pH: float = Field(..., ge=min_pH, le=max_pH)
    sulphates: float = Field(..., ge=min_sulphates, le=max_sulphates)
    alcohol: float = Field(..., ge=min_alcohol, le=max_alcohol)

class OutputSchema(BaseModel):
    quality: float = Field(..., ge=min_quality, le=max_quality)
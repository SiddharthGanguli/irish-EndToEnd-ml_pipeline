from pydantic import BaseModel, Field

class InputFeatures(BaseModel):
    model_name: str  
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Output(BaseModel):
    species: int = Field(
        ...,
        description="0 for Iris-setosa, 1 for Iris-versicolor, 2 for Iris-virginica"
    )

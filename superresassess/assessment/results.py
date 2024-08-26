from pathlib import Path
from pydantic import BaseModel, field_serializer


class AssessResult(BaseModel):
    best_model_path: Path
    log_versions: list[str]

    @field_serializer("best_model_path")
    def serialize_best_model_path(self, best_model_path: Path, _info):
        return str(best_model_path)


class InternalTestResult(BaseModel):
    internal_testing_loss: float
    num_epochs_trained: int


class ExternalTestResult(BaseModel):
    external_testing_loss: float

from fastapi.responses import JSONResponse


def error_response(message: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "gateway_error", "code": None}},
        status_code=status,
    )

import time
from typing import Annotated

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from src.v1.correct_text import main
from src.v1.schemas import GECRequest, GECResponse
from src.v1.utils.depends import make_dependable

print(GECRequest)

class ResponseErrorMessage(BaseModel):
    detail: str

router_v1 = APIRouter(
    prefix="/v1",
    tags=["GEC"],
    responses={
        status.HTTP_401_UNAUTHORIZED: {"model": ResponseErrorMessage},
        status.HTTP_403_FORBIDDEN: {"model": ResponseErrorMessage},
    }
)

@router_v1.post(
    "/gec", response_model=GECResponse, summary="Grammatical Error Correction (문법교정)"
)
async def _gec(request: Annotated[GECRequest, Depends(make_dependable(GECRequest))]):
    # 시작 시간 날짜 로그
    start_time = time.time()
    start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print(f"Request started at {start_time_formatted}")
    
    # response 로그
    result = await main(request)
    
    # result 로그
    end_time = time.time()
    end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    duration = end_time - start_time
    print(f"Request ended at {end_time_formatted}")
    print(f"Request processed in {duration} seconds")

    return result

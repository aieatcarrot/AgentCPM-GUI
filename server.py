import uvicorn
from agent_cpm import AgentCPM
from typing import Any, Callable, Union, Optional, Literal
import json
from fastapi import Body, APIRouter, Request, UploadFile, File, Form, FastAPI
from pydantic import BaseModel, Field
from typing import List
from fastapi.concurrency import run_in_threadpool
import argparse

agentCPM = AgentCPM()

class BaseResponse(BaseModel):
    code: int = Field(200, description="API status code")
    msg: str = Field("success", description="API status message")
    data: Any = Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }

class BaseRequest(BaseModel):
    image_path: Optional[str] = ""
    image_io: Optional[UploadFile] = None

class BaseOutput:
    def __init__(
        self,
        data: Any,
        format: str | Callable = None,
        data_alias: str = "",
        **extras: Any,
    ) -> None:
        self.data = data
        self.format = format
        self.extras = extras
        if data_alias:
            setattr(self, data_alias, property(lambda obj: obj.data))

    def __str__(self) -> str:
        if self.format == "json":
            return json.dumps(self.data, ensure_ascii=False, indent=2)
        elif callable(self.format):
            return self.format(self)
        else:
            return str(self.data)

at_router = APIRouter(prefix="/gui", tags=["自动化遍历"])

@at_router.post("/agent", response_model=BaseResponse, summary="页面元素检测")
async def elements_detect(image_data: Optional[UploadFile] =  File(None, description="上传图片"),
                          image_url: Optional[str] =  Form("", description="图片url地址"),
                          instruction: str = Form(description="指令"),
                          cache_id: Optional[int] = Form(None, description="历史信息缓存id"),
                          multi_turn: Optional[bool] = Form(False, description="连续")):
    
    try:
        image = image_data.file.read()
        result = await run_in_threadpool(
            agentCPM, 
            image, 
            instruction=instruction,
            multi_turn=multi_turn,
            cache_id=cache_id
        )

    except Exception as e:
        return {"code": 500, "msg": f"{e}"} 
    return BaseOutput(result)

def create_app():
    app = FastAPI(title="AI API Server", version= "0.0.1")
    # api_middleware(app)
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
    app.include_router(at_router)
    return app

def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
        )
    else:
        uvicorn.run(app, host=host, port=port)

app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ijiami",
        description="AI 自动化遍历服务",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6759)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    run_api(
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
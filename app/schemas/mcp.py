from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MCPToolDescriptor(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPResourceDescriptor(BaseModel):
    id: str
    name: str
    description: str


class MCPCapabilitiesResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    server_name: str = "telecom-support-mcp-server"
    server_version: str = "1.0.0"
    tools: list[MCPToolDescriptor]
    resources: list[MCPResourceDescriptor]


class ToolExecuteRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str = Field(..., examples=["req-1"])
    tool: str = Field(..., examples=["predict_churn"])
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPContentItem(BaseModel):
    type: Literal["text", "json"] = "json"
    text: str | None = None
    data: Any | None = None


class MCPResult(BaseModel):
    content: list[MCPContentItem]
    isError: bool = False


class MCPResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str
    result: MCPResult

from typing import AsyncGenerator
# Standard library imports
import json
import base64
import os
import logging
import ast
import subprocess
import sys
import asyncio
import resource
from functools import lru_cache

# Third-party imports
from pydantic import BaseModel
import openai

# Local app imports
from app.configs import settings
from app.oai_models import (
    ChatCompletionStreamResponse, 
    ChatCompletionResponse,
    random_uuid
)
from app.oai_streaming import create_streaming_response, ChatCompletionResponseBuilder
from app.utils import (
    strip_markers,
    refine_mcp_response,
    AgentResourceManager, 
    refine_chat_history,
    get_user_messages,
    refine_assistant_message,
    wrap_chunk
)

logger = logging.getLogger(__name__)

import re
def _has_required_citations(text: str) -> bool:
    if not text: 
        return False
    has_link = bool(re.search(r'https?://(www\.)?uscis\.gov/policy-manual', text))
    has_label = bool(re.search(r'Volume\s+\w+\s+Chapter\s+\w+(?:\s+Section\s+\w+)?', text, re.I))
    return has_link and has_label

@lru_cache(maxsize=1)
def get_avatar() -> str:
    if not os.path.exists('assets/images/robot.png'):
        return '🤖'
    
    with open('assets/images/robot.png', 'rb') as image_file:
        return f'data:image/png;base64,{base64.b64encode(image_file.read()).decode("utf-8")}'

class Executor:
    def __init__(self):
        self.history = []

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": "Run python code for data analysis, numerical calculation, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "The reasoning for the python code"
                            },
                            "code": {
                                "type": "string",
                                "description": "The python code to run"
                            }
                        },
                        "required": ["reasoning", "code"]
                    }
                }
            }
        ]




    async def python(self, code: str) -> str:
        variables = []
        tree = ast.parse(code)

        # Only get assignments at the global/module level
        for node in tree.body:
            if isinstance(node, ast.Assign):
                first_target = node.targets[0]
                if isinstance(first_target, ast.Name):
                    variables.append(first_target.id)
                    
                if isinstance(first_target, ast.Attribute):
                    variables.append(first_target.attr)
                
                if isinstance(first_target, ast.Subscript):
                    variables.append(first_target.value.id)

                if isinstance(first_target, ast.Tuple):
                    for target in first_target.elts:
                        if isinstance(target, ast.Name):
                            variables.append(target.id)
                            
                if isinstance(first_target, ast.List):
                    for target in first_target.elts:
                        if isinstance(target, ast.Name):
                            variables.append(target.id)

        return_variables = set(variables)

        for var in variables:
            if var in return_variables:
                code += f'\nprint("{var} =", {var})'

        code += '\n'
        
        max_memory = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') * 0.5)
        max_cpu = int(os.sysconf('SC_CLK_TCK') * 0.5)
        
        def limit_resource(memory_limit: int, cpu_limit: int):
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))


        result: subprocess.CompletedProcess[str] = await asyncio.to_thread(
            subprocess.run, 
            [sys.executable, "-c", code],
            preexec_fn=lambda: limit_resource(max_memory, max_cpu),
            capture_output=True,
            text=True,
            timeout=30
        )

        out, err = result.stdout.strip(), result.stderr.strip()
        return_code = result.returncode
        
        if err:
            return f"{out} (error: {err!r}; return code: {return_code})"

        return out

    async def search_federal_regulations(self, tool_args: dict[str, str]) -> str:
        """Search federal regulations using MCP functions"""
        try:
            from mcps.regulations.main import search_documents
            
            # Convert tool_args to the format expected by the MCP function
            search_term = tool_args.get("search_term", "")
            agency_id = tool_args.get("agency_id")
            document_type = tool_args.get("document_type")
            docket_id = tool_args.get("docket_id")
            posted_date = tool_args.get("posted_date")
            page_size = int(tool_args.get("page_size", 20))
            page_number = int(tool_args.get("page_number", 1))
            
            result = await search_documents(
                search_term=search_term,
                agency_id=agency_id,
                document_type=document_type,
                docket_id=docket_id,
                posted_date=posted_date,
                page_size=page_size,
                page_number=page_number
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error searching federal regulations: {e}")
            return f"Error searching federal regulations: {str(e)}"

    async def get_document_details(self, tool_args: dict[str, str]) -> str:
        """Get detailed information for a specific federal regulatory document"""
        try:
            from mcps.regulations.main import get_document_details as mcp_get_document_details
            
            document_id = tool_args.get("document_id", "")
            include_attachments = tool_args.get("include_attachments", False)
            
            result = await mcp_get_document_details(
                document_id=document_id,
                include_attachments=include_attachments
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error getting document details: {e}")
            return f"Error getting document details: {str(e)}"

    async def search_comments(self, tool_args: dict[str, str]) -> str:
        """Search for public comments on federal regulatory documents"""
        try:
            from mcps.regulations.main import search_comments
            
            # Convert tool_args to the format expected by the MCP function
            search_term = tool_args.get("search_term")
            agency_id = tool_args.get("agency_id")
            comment_on_id = tool_args.get("comment_on_id")
            posted_date = tool_args.get("posted_date")
            last_modified_date = tool_args.get("last_modified_date")
            page_size = int(tool_args.get("page_size", 20))
            page_number = int(tool_args.get("page_number", 1))
            sort = tool_args.get("sort")
            
            result = await search_comments(
                search_term=search_term,
                agency_id=agency_id,
                comment_on_id=comment_on_id,
                posted_date=posted_date,
                last_modified_date=last_modified_date,
                page_size=page_size,
                page_number=page_number,
                sort=sort
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error searching comments: {e}")
            return f"Error searching comments: {str(e)}"

    async def get_comment_details(self, tool_args: dict[str, str]) -> str:
        """Get detailed information for a specific public comment"""
        try:
            from mcps.regulations.main import get_comment_details
            
            comment_id = tool_args.get("comment_id", "")
            include_attachments = tool_args.get("include_attachments", False)
            
            result = await get_comment_details(
                comment_id=comment_id,
                include_attachments=include_attachments
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error getting comment details: {e}")
            return f"Error getting comment details: {str(e)}"

    async def search_dockets(self, tool_args: dict[str, str]) -> str:
        """Search for regulatory dockets"""
        try:
            from mcps.regulations.main import search_dockets
            
            # Convert tool_args to the format expected by the MCP function
            search_term = tool_args.get("search_term")
            agency_id = tool_args.get("agency_id")
            docket_type = tool_args.get("docket_type")
            last_modified_date = tool_args.get("last_modified_date")
            page_size = int(tool_args.get("page_size", 20))
            page_number = int(tool_args.get("page_number", 1))
            sort = tool_args.get("sort")
            
            result = await search_dockets(
                search_term=search_term,
                agency_id=agency_id,
                docket_type=docket_type,
                last_modified_date=last_modified_date,
                page_size=page_size,
                page_number=page_number,
                sort=sort
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error searching dockets: {e}")
            return f"Error searching dockets: {str(e)}"

    async def get_docket_details(self, tool_args: dict[str, str]) -> str:
        """Get detailed information for a specific regulatory docket"""
        try:
            from mcps.regulations.main import get_docket_details
            
            docket_id = tool_args.get("docket_id", "")
            
            result = await get_docket_details(
                docket_id=docket_id
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error getting docket details: {e}")
            return f"Error getting docket details: {str(e)}"


        
    async def execute_tool(self, tool_name: str, tool_args: dict[str, str]) -> str:
        if tool_name == "search_documents":
            search_term = tool_args.get("search_term", "")
            if not search_term:
                return "No search term provided"
            logger.info(f'Executing federal regulations search: {search_term} ({tool_args.get("reasoning", "")})')
            return await self.search_federal_regulations(tool_args)

        if tool_name == "get_document_details":
            document_id = tool_args.get("document_id", "")
            if not document_id:
                return "No document ID provided"
            logger.info(f'Getting document details: {document_id} ({tool_args.get("reasoning", "")})')
            return await self.get_document_details(tool_args)

        if tool_name == "search_comments":
            logger.info(f'Searching comments: {tool_args.get("search_term", "all")} ({tool_args.get("reasoning", "")})')
            return await self.search_comments(tool_args)

        if tool_name == "get_comment_details":
            comment_id = tool_args.get("comment_id", "")
            if not comment_id:
                return "No comment ID provided"
            logger.info(f'Getting comment details: {comment_id} ({tool_args.get("reasoning", "")})')
            return await self.get_comment_details(tool_args)

        if tool_name == "search_dockets":
            logger.info(f'Searching dockets: {tool_args.get("search_term", "all")} ({tool_args.get("reasoning", "")})')
            return await self.search_dockets(tool_args)

        if tool_name == "get_docket_details":
            docket_id = tool_args.get("docket_id", "")
            if not docket_id:
                return "No docket ID provided"
            logger.info(f'Getting docket details: {docket_id} ({tool_args.get("reasoning", "")})')
            return await self.get_docket_details(tool_args)



        if tool_name == "python":
            code = tool_args.get("code", "")
            if not code:
                return "No code provided"
            logger.info(f'Executing python code: {code} ({tool_args.get("reasoning", "")})')
            return await self.python(code)

        raise ValueError(f"Unknown tool: {tool_name}")

    def compact_history(self):
        self.history = [
            (e.model_dump() if isinstance(e, BaseModel) else e) 
            for e in self.history
        ]
        
        self.history = [
            {
                'role': e['role'],
                'content': strip_markers(
                    e['content'], 
                    (
                        ('action', False),
                        ('details', False), 
                        ('agent_message', False), 
                        ('think', False),
                    )
                )
            }
            for e in self.history
            if e['role'] != 'tool'
        ]

    async def execute(self, expectations: str, steps: list[str], output: str | None = None) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        tools = self.list_tools()
        prompt_template = "Expectations: {expectations}\nSteps: {steps}\nFinal output guideline: {output}"

        self.history.append(
            {
                "role": "user",
                "content": prompt_template.format(
                    expectations=expectations,
                    steps=steps,
                    output=output or "whatever"
                )
            }
        )

        while True:
            generator = create_streaming_response(
                base_url=settings.llm_base_url,
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}"
                },
                model=settings.llm_model_id,
                messages=self.history,
                tools=tools
            )

            response_builder = ChatCompletionResponseBuilder()

            async for chunk in generator:
                response_builder.add_chunk(chunk)
                
                if chunk.choices[0].delta.content:
                    yield chunk

            completion = await response_builder.build()
            self.history.append(refine_assistant_message(completion.choices[0].message))

            if not (completion.choices[0].message.tool_calls or []):
                break

            for call in (completion.choices[0].message.tool_calls or []):
                tool_call_id = call.id
                tool_name = call.function.name
                tool_args = json.loads(call.function.arguments)

                yield wrap_chunk(random_uuid(), f"<action>Executing {tool_name}</action>", role='assistant')
                yield wrap_chunk(random_uuid(), f"<details><summary>Arguments</summary>\n```json\n{call.function.arguments}\n```\n</details>", role='assistant')
                result = await self.execute_tool(tool_name, tool_args)
                yield wrap_chunk(random_uuid(), f"<details><summary>Result</summary>\n```\n{result}\n```\n</details>", role='assistant')

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result
                })
        
        self.compact_history()

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cot_planning",
            "description": "Use Chain-of-Thought reasoning to break down complex legal requests into logical steps before creating the legal analysis plan",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The user's original legal question or request"
                    },
                    "reasoning_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The logical reasoning steps identified"
                    },
                    "planning_summary": {
                        "type": "string",
                        "description": "Summary of the planning approach"
                    }
                },
                "required": ["user_request", "reasoning_steps", "planning_summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the federal regulations database (Regulations.gov) for current federal regulatory documents, notices, and rules. Use this for real-time federal regulatory information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Search term to filter documents (e.g. 'water', 'healthcare')"},
                    "agency_id": {"type": "string", "description": "Agency acronym to filter by (e.g. 'EPA', 'FDA')"},
                    "document_type": {"type": "string", "description": "Document type filter (Notice, Rule, Proposed Rule, Supporting & Related Material, Other)"},
                    "docket_id": {"type": "string", "description": "Docket ID to filter by"},
                    "posted_date": {"type": "string", "description": "Posted date filter (format: yyyy-MM-dd)"},
                    "page_size": {"type": "integer", "description": "Number of results per page (5-250, default: 20)"},
                    "page_number": {"type": "integer", "description": "Page number (1-20, default: 1)"},
                    "reasoning": {"type": "string", "description": "Why this search is needed"}
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_details",
            "description": "Get detailed information for a specific federal regulatory document from Regulations.gov",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "description": "The document ID (e.g. 'FDA-2009-N-0501-0012')"},
                    "include_attachments": {"type": "boolean", "description": "Whether to include attachments in the response"},
                    "reasoning": {"type": "string", "description": "Why this document is needed"}
                },
                "required": ["document_id"]
            }
        }
    },


    {
        "type": "function",
        "function": {
            "name": "search_comments",
            "description": "Search for public comments on federal regulatory documents from Regulations.gov",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Search term to filter comments (e.g. 'water', 'healthcare')"},
                    "agency_id": {"type": "string", "description": "Agency acronym to filter by (e.g. 'EPA', 'FDA')"},
                    "comment_on_id": {"type": "string", "description": "Object ID to filter comments for a specific document"},
                    "posted_date": {"type": "string", "description": "Posted date filter (format: yyyy-MM-dd)"},
                    "last_modified_date": {"type": "string", "description": "Last modified date filter (format: yyyy-MM-dd HH:mm:ss)"},
                    "page_size": {"type": "integer", "description": "Number of results per page (5-250, default: 20)"},
                    "page_number": {"type": "integer", "description": "Page number (1-20, default: 1)"},
                    "sort": {"type": "string", "description": "Sort field (postedDate, lastModifiedDate, documentId) with optional - prefix for desc"},
                    "reasoning": {"type": "string", "description": "Why this search is needed"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_comment_details",
            "description": "Get detailed information for a specific public comment from Regulations.gov",
            "parameters": {
                "type": "object",
                "properties": {
                    "comment_id": {"type": "string", "description": "The comment ID (e.g. 'HHS-OCR-2018-0002-5313')"},
                    "include_attachments": {"type": "boolean", "description": "Whether to include attachments in the response"},
                    "reasoning": {"type": "string", "description": "Why this comment is needed"}
                },
                "required": ["comment_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_dockets",
            "description": "Search for regulatory dockets (organizational folders containing multiple documents) from Regulations.gov",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "Search term to filter dockets (e.g. 'water', 'healthcare')"},
                    "agency_id": {"type": "string", "description": "Agency acronym to filter by (e.g. 'EPA', 'FDA')"},
                    "docket_type": {"type": "string", "description": "Docket type filter (Rulemaking, Nonrulemaking)"},
                    "last_modified_date": {"type": "string", "description": "Last modified date filter (format: yyyy-MM-dd HH:mm:ss)"},
                    "page_size": {"type": "integer", "description": "Number of results per page (5-250, default: 20)"},
                    "page_number": {"type": "integer", "description": "Page number (1-20, default: 1)"},
                    "sort": {"type": "string", "description": "Sort field (title, docketId, lastModifiedDate) with optional - prefix for desc"},
                    "reasoning": {"type": "string", "description": "Why this search is needed"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_docket_details",
            "description": "Get detailed information for a specific regulatory docket from Regulations.gov",
            "parameters": {
                "type": "object",
                "properties": {
                    "docket_id": {"type": "string", "description": "The docket ID (e.g. 'EPA-HQ-OAR-2003-0129')"},
                    "reasoning": {"type": "string", "description": "Why this docket is needed"}
                },
                "required": ["docket_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Run python code for data analysis, calculations, or legal document processing",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    }
]


def get_executor_ability(executor: Executor) -> str:
    return "\n".join([
        f"- {tool['function']['name']}: {tool['function']['description']}" 
        for tool in executor.list_tools()
    ])
    
async def get_system_prompt(chat_history: list[dict[str, str]], executor: Executor) -> str:
    # Base legal assistant prompt
    system_prompt = """You are a comprehensive legal research assistant with expertise in:
- Federal regulations and administrative law
- Constitutional law and civil rights
- Business and corporate law
- Employment law and labor relations
- Environmental law and regulations
- Healthcare law and compliance
- Tax law and regulations
- Intellectual property law
- Criminal law and procedure
- Family law and domestic relations
- Real estate and property law
- Contract law and commercial transactions

Your responsibilities:
1. **Legal Research**: Search federal regulations and regulatory databases
2. **Regulatory Analysis**: Explain regulatory requirements and compliance procedures
3. **Legal Guidance**: Provide step-by-step guidance on legal processes and procedures
4. **Document Review**: Help analyze legal documents and identify key requirements
5. **Compliance Assessment**: Assist with understanding regulatory compliance obligations
6. **Timeline Information**: Provide realistic processing time expectations for legal matters

CITATION REQUIREMENTS:
- **ALWAYS include direct URLs** to relevant legal sources when available
- **Provide clickable links** using Markdown link format `[Title](URL)`
- **Reference specific regulations, statutes, or case citations** with links
- **Include links to official government websites** and regulatory agencies
- **Add links to relevant legal databases** and resources

Use this structured approach in your answers:
1) Summary
2) Legal Framework
3) Requirements & Eligibility
4) Procedures & Process
5) Compliance Considerations
6) Common Issues & Pitfalls
7) Sources & References

IMPORTANT: Every piece of legal information should be backed by authoritative sources and links when possible.

PLANNER RULE: You MUST call the `cot_planning` tool BEFORE calling other tools, then `search_documents` and `python` as needed. Always plan first, then search for relevant federal regulatory information, then provide a structured response with citations.
"""


    # Add executor capabilities
    executor_ability = get_executor_ability(executor)
    system_prompt += f"\nExecutor ability:\n{executor_ability}"
    
    return system_prompt



async def handle_prompt(messages: list[dict[str, str]]) -> AsyncGenerator[ChatCompletionStreamResponse | ChatCompletionResponse, None]:
    executor = Executor()
    system_prompt = await get_system_prompt(messages, executor=executor)

    arm = AgentResourceManager()
    messages = refine_chat_history(messages, system_prompt, arm)

    cot_done = False

    async_client = openai.AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    while True:
        completion = await async_client.chat.completions.create(
            model=settings.llm_model_id,
            messages=messages,
            tools=PLANNER_TOOLS,
            stream=False,
        )
        assistant_message = completion.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        message = {"role": "assistant", "content": assistant_message.content or ""}

        if not tool_calls:
            messages.append(message)
            if assistant_message.content:
                yield wrap_chunk(random_uuid(), assistant_message.content, role="assistant")
            break

        message["tool_calls"] = [{
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in tool_calls]
        messages.append(message)

        for call in tool_calls:
            id = call.id
            name = call.function.name
            args = call.function.arguments
            args_json = json.loads(args or "{}")
            
            print(f"call: {call}; name: {name}; args: {args}; args_json: {args_json};")

            if name == "cot_planning":
                cot_done = True
                steps = args_json.get("reasoning_steps", [])
                expectations = args_json.get("user_request", "")
                
                output = ''

                try:
                    logger.info(f"Executing executor with expectations: {expectations} and steps: {steps}")

                    async for item in executor.execute(expectations, steps, "Should be short and concise"):
                        if isinstance(item, ChatCompletionStreamResponse) and item.choices[0].delta.content:
                            output += item.choices[0].delta.content

                except Exception as e:
                    logger.error(f"Error while running executor: {str(e)}", exc_info=True)
                    output = f"Error while running executor: {str(e)}"
                    
                finally:
                    output = refine_mcp_response(output, arm)

                messages.append({
                    'role': 'tool',
                    'tool_call_id': id,
                    'content': output
                })
                continue

            elif name in ["search_documents", "get_document_details", "search_comments", "get_comment_details", "search_dockets", "get_docket_details", "python"]:
                if not cot_done:
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': id,
                        'content': 'You must start with `cot_planning` before any other tool.'
                    })
                    continue

                out = await executor.execute_tool(name, args_json)
                messages.append({"role": "tool", "tool_call_id": id, "content": out})
from typing import AsyncGenerator
from app.oai_models import ChatCompletionStreamResponse, ChatCompletionResponse
from app.configs import settings
from app.utils import (
    wrap_chunk, 
    random_uuid, 
    strip_markers,
    refine_mcp_response,
    AgentResourceManager, 
    refine_chat_history,
    get_user_messages,
    refine_assistant_message
)
from app.oai_streaming import create_streaming_response, ChatCompletionResponseBuilder
import json
from functools import lru_cache
import base64
import os
import logging
import ast
import subprocess
import sys
import asyncio
import resource
from app.engine import get_db_info, search as db_search, get_active_collection
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
        info = get_db_info()
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": f"Search for information in the {info.name} database. {info.description}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "The reasoning for the search query"
                            },
                            "query": {
                                "type": "string",
                                "description": "The query to search for"
                            }
                        },
                        "required": ["reasoning", "query"]
                    }
                }
            },
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

    async def search(self, query: str) -> str: # TODO: add keyword search later
        logger.info(f"Searching for {query}")
        results = await db_search(query)
        logger.info(f"Got {len(results)} results")

        return "\n".join([f"- {result.content} (distance: {result.distance})" for result in results])


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
        
    async def execute_tool(self, tool_name: str, tool_args: dict[str, str]) -> str:
        if tool_name == "search":
            logger.info(f'Executing search: {tool_args["query"]} ({tool_args["reasoning"]})')
            return await self.search(tool_args["query"])

        if tool_name == "python":
            logger.info(f'Executing python code: {tool_args["code"]} ({tool_args["reasoning"]})')
            return await self.python(tool_args["code"])

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
            "name": "legal_analysis_plan",
            "description": "Create a comprehensive legal analysis plan with specific steps for immigration law questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "legal_issue": {
                        "type": "string",
                        "description": "The specific legal question or issue being addressed"
                    },
                    "legal_basis": {
                        "type": "string",
                        "description": "Relevant laws, policies, or regulations that apply"
                    },
                    "analysis_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Step-by-step legal analysis and research process"
                    },
                    "required_documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Legal forms and documents that may be needed"
                    },
                    "timeline": {
                        "type": "string",
                        "description": "Expected processing times and important deadlines"
                    }
                },
                "required": ["legal_issue", "legal_basis", "analysis_steps"]
            }
        }
    }
]


def get_executor_ability(executor: Executor) -> str:
    return "\n".join([
        f"- {tool['function']['name']}: {tool['function']['description']}" 
        for tool in executor.list_tools()
    ])
    
from app.lite_keybert import KeyBERT

async def get_system_prompt(chat_history: list[dict[str, str]], executor: Executor) -> str:
    # Base legal assistant prompt
    system_prompt = """You are an immigration law specialist with expertise in:
- U.S. Citizenship and Immigration Services (USCIS) policies
- Family-based immigration (spouse, parent, child petitions)
- Employment-based immigration (work visas, green cards)
- Naturalization and citizenship processes
- Immigration court procedures and appeals

Your responsibilities:
1. **Legal Research**: Search the USCIS Policy Manual and related legal databases
2. **Process Guidance**: Explain step-by-step procedures for immigration applications
3. **Eligibility Assessment**: Help users understand qualification requirements
4. **Document Preparation**: Guide users on required forms and documentation
5. **Timeline Information**: Provide realistic processing time expectations

CITATION REQUIREMENTS:
- **ALWAYS include direct URLs** to USCIS Policy Manual sections when available
- **Provide clickable links** using Markdown link format `[Title](URL)`
- **Reference specific volume, chapter, and section numbers** with links
- **Include form download links** from USCIS website
- **Add links to relevant USCIS pages** for additional information

Response Format (Markdown):
- Start with a **Summary** section
- Provide a **Step-by-Step Guidance** section with numbered lists
- Add **Policy Citations with Markdown links**
- Add **Warnings / Important Notes**
- End with a **Sources & Links** section in bullet point format with clickable Markdown links
- Always remind users to seek legal counsel for specific cases

IMPORTANT: Every piece of legal information should be backed by a Markdown link when possible.
"""


    # Add executor capabilities
    executor_ability = get_executor_ability(executor)
    system_prompt += f"\nExecutor ability:\n{executor_ability}"

    # Add legal context from database
    user_messages = get_user_messages(chat_history, 3)
    kw_thresold = 0.5
    top_k = 3

    try:
        kb = KeyBERT()
        keywords = await kb.extract_keywords(
            user_messages, 
            keyphrase_ngram_range=(2, 4),
            stop_words="english", 
            use_maxsum=False,
            use_mmr=True,
            diversity=0.7,
            threshold=kw_thresold,
            merge=True
        )

        flatterned_keywords = []
        
        for kw in keywords:
            flatterned_keywords.extend(kw)

        # sort by score
        flatterned_keywords.sort(key=lambda x: x[1], reverse=True)
        flatterned_keywords = [kw for kw, score in flatterned_keywords[:top_k]]
        
        logger.info(f'Detected flatterned_keywords: {flatterned_keywords}')
        hits = await db_search(flatterned_keywords)

        if hits:
            system_prompt += f"\nReferences:"

            for hit in hits:
                system_prompt += f"\n- {hit.content}"

    except Exception as err:
        logger.error(f'Error while extracting flatterned_keywords and searching for relevant information: {err}', exc_info=True)
    
    return system_prompt

async def handle_prompt(messages: list[dict[str, str]]) -> AsyncGenerator[ChatCompletionStreamResponse | ChatCompletionResponse, None]:
    executor = Executor()
    system_prompt = await get_system_prompt(messages, executor=executor)

    arm = AgentResourceManager()
    messages = refine_chat_history(messages, system_prompt, arm)

    while True:
        generator = create_streaming_response(
            base_url=settings.llm_base_url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}"
            },
            model=settings.llm_model_id,
            messages=messages,
            tools=PLANNER_TOOLS
        )

        response_builder = ChatCompletionResponseBuilder()

        async for chunk in generator:
            response_builder.add_chunk(chunk)

            if chunk.choices[0].delta.content:
                yield chunk

        completion = await response_builder.build()

        if not (completion.choices[0].message.tool_calls or []):
            break

        expectations = ''
        steps = []

        for call in (completion.choices[0].message.tool_calls or []):
            tool_name = call.function.name
            tool_args = json.loads(call.function.arguments)
            
            if tool_name == 'legal_analysis_plan':
                expectations += tool_args['legal_issue'] + '\n'
                steps.extend(tool_args['analysis_steps'])

        call_id = random_uuid()

        messages.append({
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': call_id,
                    'type': 'function',
                    'function': {
                        'name': 'legal_analysis_plan',
                        'arguments': json.dumps({
                            'legal_issue': expectations,
                            'legal_basis': tool_args['legal_basis'],
                            'analysis_steps': steps,
                            'required_documents': tool_args['required_documents'],
                            'timeline': tool_args['timeline']
                        })
                    }
                }
            ]
        })

        if len(steps) > 0:
            output = ''

            try:
                yield wrap_chunk(random_uuid(), f'<agent_message avatar="{get_avatar()}" notification="Executor is working">', role='assistant')
                logger.info(f"Executing executor with expectations: {expectations} and steps: {steps}")

                async for item in executor.execute(expectations, steps, "Should be short and concise"):
                    if isinstance(item, ChatCompletionStreamResponse) and item.choices[0].delta.content:
                        output += item.choices[0].delta.content
                        yield item

            except Exception as e:
                logger.error(f"Error while running executor: {str(e)}", exc_info=True)
                output = f"Error while running executor: {str(e)}"
                
            finally:
                yield wrap_chunk(random_uuid(), "</agent_message>", role='assistant')
                output = refine_mcp_response(output, arm)

        else:
            output = 'Please specify at least one step for the executor to run.'

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": output
        })

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
import re
import openai

logger = logging.getLogger(__name__)

class Step(BaseModel):
    reason: str
    task: str
    expectation: str

COT_TEMPLATE = """
You are an analytical assistant. Your task is to break the user request into a list of steps, each step should clearly describe a single action with expectation output. At each step, it should be a solid link with the previous. The user asked:
"{user_request}"

So far, these are the steps completed:
{context}

What is the next step should we do?
Respond in JSON format: {{ "reason": "...", "task": "...", "expectation": "..." }}
If no more are needed, just return: <done/>.
"""

def make_plan(user_request: str, max_steps: int = 15) -> list[Step]:
    list_of_steps: list[Step] = []
    
    client = openai.OpenAI(
        api_key=settings.llm_api_key, 
        base_url=settings.llm_base_url
    )
    
    for _ in range(max_steps):
        context = "\n".join([f"{i+1}. {step.task}: {step.expectation}" for i, step in enumerate(list_of_steps)])
        prompt = COT_TEMPLATE.format(
            user_request=user_request,
            context=context
        )

        response = client.chat.completions.create(
            model=settings.llm_model_id,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content
        
        if "<done/>" in response_text.strip().lower():
            reasoning = None

            if 'reason' in response_text.lower():
                try:
                    l, r = response_text.find('{'), response_text.rfind('}')+1
                    resp_json: dict = json.loads(response_text[l:r])
                    reasoning = resp_json.get('reason')

                except Exception as err:
                    logger.error(f"Error parsing JSON: {err}; Response: {response_text}")

            if not reasoning:
                reasoning = response_text.strip()

            break

        try:
            l, r = response_text.find('{'), response_text.rfind('}')+1
            step_data: dict = json.loads(response_text[l:r])
            step = Step(**step_data)
            list_of_steps.append(step)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")

    return list_of_steps


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
                            "query": {
                                "type": "string",
                                "description": "The query to search for"
                            }
                        },
                        "required": ["query"]
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
                            "code": {
                                "type": "string",
                                "description": "The python code to run"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

    async def search(self, query: str) -> str: # TODO: add keyword search later
        logger.info(f"Searching for {query}")
        
        # Extract legal intent for faster search
        legal_intent = await extract_legal_intent(query)
        
        # Use enhanced search with legal context
        results = await db_search(query, legal_context=legal_intent)
        logger.info(f"Got {len(results)} results")

        # Format results efficiently
        formatted_results = []
        for result in results:
            citation = f" (Section: {result.metadata.section})" if result.metadata.section else ""
            formatted_results.append(f"- {result.content}{citation}")
        
        return "\n".join(formatted_results)

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

        return_variables = set(return_variables)

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
            logger.info(f'Executing search: {tool_args["query"]}')
            return await self.search(tool_args["query"])

        if tool_name == "python":
            logger.info(f'Executing python code: {tool_args["code"]}')
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
            "name": "make_plan",
            "description": "Create a plan with reasoning for the next step",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief reasoning for this plan"
                    },
                    "expectations": {
                        "type": "string",
                        "description": "The expectations and goals for this step"
                    },
                    "steps": {
                        "type": "array",
                        "description": "Step-by-step actions to meet the expectations",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["reasoning", "expectations", "steps"]
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
    system_prompt = """You are a USCIS legal information assistant. Use Chain of Thought reasoning for planning:

1. **Accurate**: Only provide information found in USCIS Policy Manual
2. **Current**: Prioritize the most recent policy information  
3. **Specific**: Include exact citations and section references
4. **Clear**: Explain complex legal concepts in simple terms

**Use systematic reasoning**: Think step-by-step, plan search strategy, consider challenges.

Your task is to make plan, review the result of the executor and response to the user accurately."""
    executor_ability = get_executor_ability(executor)
    system_prompt += f"\nExecutor ability:\n{executor_ability}"

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
                citation = f" (Section: {hit.metadata.section})" if hit.metadata.section else ""
                system_prompt += f"\n- {hit.content}{citation}"

    except Exception as err:
        logger.error(f'Error while extracting flatterned_keywords and searching for relevant information: {err}', exc_info=True)
    
    return system_prompt

async def handle_prompt(messages: list[dict[str, str]]) -> AsyncGenerator[ChatCompletionStreamResponse | ChatCompletionResponse, None]:
    executor = Executor()
    system_prompt = await get_system_prompt(messages, executor=executor)

    arm = AgentResourceManager()
    messages = refine_chat_history(messages, system_prompt, arm)

    use_simple_cot = os.getenv("USE_SIMPLE_COT_PLANNER", "0").lower() in ("1", "true", "yes")

    if use_simple_cot:
        # Single-pass fast CoT planner: derive steps and execute once
        user_request = messages[-1].get("content", "")
        steps = make_plan(user_request, max_steps=5)
        if steps:
            expectations = "\n".join([step.expectation for step in steps])
            plan_steps = [step.task for step in steps]

            yield wrap_chunk(random_uuid(), f"<think>Planning {len(plan_steps)} steps</think>", role='assistant')

            output = ''
            try:
                yield wrap_chunk(random_uuid(), f'<agent_message avatar="{get_avatar()}" notification="Executor is working">', role='assistant')
                async for item in executor.execute(expectations, plan_steps, "Be concise"):
                    if isinstance(item, ChatCompletionStreamResponse) and item.choices[0].delta.content:
                        output += item.choices[0].delta.content
                        yield item
            except Exception as e:
                logger.error(f"Error while running executor: {str(e)}", exc_info=True)
                output = f"Error while running executor: {str(e)}"
            finally:
                yield wrap_chunk(random_uuid(), "</agent_message>", role='assistant')
                output = refine_mcp_response(output, arm)
                output += "\n\n*This is general guidance, not legal advice.*"

            # Return once
            yield ChatCompletionResponse(
                id=random_uuid(),
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": output},
                    "finish_reason": "stop"
                }],
                created=int(asyncio.get_event_loop().time()),
                model=settings.llm_model_id,
                object="chat.completion"
            )
            return
        # If no steps, fall through to default planner

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
            
            if tool_name == 'make_plan':
                # Extract Chain of Thought reasoning
                reasoning = tool_args.get('reasoning', '')
                expectations += tool_args['expectations'] + '\n'
                steps.extend(tool_args['steps'])
                
                # Log the reasoning for transparency
                logger.info(f"Planner reasoning: {reasoning}")
                
                # Add reasoning to the response for user transparency
                if reasoning:
                    yield wrap_chunk(random_uuid(), f"<think>{reasoning}</think>", role='assistant')

        call_id = random_uuid()

        messages.append({
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': call_id,
                    'type': 'function',
                    'function': {
                        'name': 'make_plan',
                        'arguments': json.dumps({
                            'expectations': expectations,
                            'steps': steps
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
                
                # Show concise execution reasoning
                execution_reasoning = f"Executing {len(steps)} steps: {', '.join(steps[:2])}{'...' if len(steps) > 2 else ''}"
                yield wrap_chunk(random_uuid(), f"<think>{execution_reasoning}</think>", role='assistant')

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
                
                # Add concise Chain of Thought summary
                if 'reasoning' in locals():
                    summary = f"CoT: {len(steps)} steps executed successfully"
                    yield wrap_chunk(random_uuid(), f"<summary>{summary}</summary>", role='assistant')
                
                # Add brief legal disclaimer
                output += "\n\n*This is general guidance, not legal advice.*"

        else:
            output = 'Please specify at least one step for the executor to run.'
            output += "\n\n*This is general guidance, not legal advice.*"

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": output
        })

async def extract_legal_intent(query: str) -> dict:
    """Extract legal intent from user query - optimized for speed"""
    legal_patterns = {
        "eligibility": r"(eligible|qualify|requirements|criteria|qualification)",
        "process": r"(how to|process|steps|procedure|apply|application)",
        "timeline": r"(how long|time|duration|processing time|wait|schedule)",
        "documents": r"(documents|forms|evidence|proof|required|submit)",
        "appeals": r"(appeal|denied|rejected|challenge|reconsideration)",
        "fees": r"(cost|fee|payment|money|price)",
        "status": r"(status|check|track|current|pending)",
        "renewal": r"(renew|extension|continue|maintain)",
        "change": r"(change|modify|update|correct|amend)"
    }
    
    intent = {}
    for category, pattern in legal_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            intent[category] = True
    
    return intent

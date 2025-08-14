from typing import AsyncGenerator
from app.oai_models import ChatCompletionStreamResponse
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
import time
from typing import Optional

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

def _parse_step_obj(text: str) -> dict | None:
    """Parse step object from LLM response text."""
    try:
        candidate = text
        if "```" in text:
            start = text.find("```")
            end = text.find("```", start + 3)
            if end != -1:
                candidate = text[start + 3:end]
        l, r = candidate.find('{'), candidate.rfind('}') + 1
        if l != -1 and r > l:
            candidate = candidate[l:r]
        try:
            data = json.loads(candidate)
        except Exception:
            try:
                from json_repair import repair_json
                repaired = repair_json(candidate)
                if isinstance(repaired, tuple) and len(repaired) > 0:
                    repaired = repaired[0]
                data = json.loads(repaired)
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                return None
        if isinstance(data, dict) and isinstance(data.get('steps'), list) and len(data['steps']) > 0:
            data = data['steps'][0]
        if isinstance(data, dict):
            if 'task' not in data:
                data['task'] = "Legal research action based on reasoning"
            if 'expectation' not in data:
                data['expectation'] = "Expected legal information to retrieve"
            if 'reason' not in data:
                data['reason'] = ""
            return data
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing step: {e}")
        return None

async def make_plan(user_request: str, max_steps: int = 15) -> list[Step]:
    """Generate a plan for legal research using Chain of Thought reasoning."""
    if not user_request or not user_request.strip():
        logger.warning("Empty user request provided to make_plan")
        return []
    
    if max_steps < 1 or max_steps > 20:
        logger.warning(f"Invalid max_steps: {max_steps}, clamping to [1, 20]")
        max_steps = max(1, min(max_steps, 20))
    
    list_of_steps: list[Step] = []
    
    def _make_plan_sync(): # Inner synchronous function
        try:
            client = openai.OpenAI(
                api_key=settings.llm_api_key, 
                base_url=settings.llm_base_url
            )
            
            # Enhanced legal-focused prompt for better step generation
            enhanced_prompt = """You are a USCIS legal research specialist with expertise in immigration law. Break down this legal query into focused, actionable research steps:

User Request: "{user_request}"

Generate legal research steps that:
1. Are specific and actionable with concrete search terms (e.g., "Search for 'continuous residence requirements naturalization INA 316' or '5 year permanent resident naturalization eligibility'")
2. Build upon each other logically for legal analysis
3. Cover all aspects of the legal question comprehensively
4. Follow USCIS policy manual structure and legal precedents
5. Will provide legally accurate and complete guidance
6. Include specific legal citations, form numbers, or section references when possible

IMPORTANT: Each step should contain specific legal terms, section numbers, or form names that can be searched for in the USCIS Policy Manual database.

Current steps: {current_steps}/{max_steps}

Respond in JSON format:
{{ "reason": "Legal reasoning for this step", "task": "Specific legal research action", "expectation": "Legal information we expect to find" }}

If no more steps needed for comprehensive legal coverage, return: <done/>"""

            for step_count in range(max_steps):
                context = "\n".join([f"{i+1}. {step.task}: {step.expectation}" for i, step in enumerate(list_of_steps)])
                prompt = enhanced_prompt.format(
                    user_request=user_request,
                    current_steps=len(list_of_steps),
                    max_steps=max_steps
                )

                try:
                    response = client.chat.completions.create(
                        model=settings.llm_model_id,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=30  # Add timeout for API calls
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
                        step_data = _parse_step_obj(response_text)
                        if step_data is None:
                            logger.warning(f"Failed to parse step {step_count + 1}, skipping")
                            continue
                        
                        step = Step(**step_data)
                        
                        # Validate legal step quality
                        if not validate_legal_step(step):
                            logger.warning(f"Generated step {step_count + 1} failed legal validation: {step.task}")
                            # Try to regenerate with more specific guidance
                            continue
                        
                        list_of_steps.append(step)
                        
                        # Check step dependencies
                        if len(list_of_steps) > 1 and not check_step_dependencies(list_of_steps):
                            logger.warning(f"Step dependencies check failed for step {step_count + 1}")
                            # Continue but log the issue
                        
                    except Exception as e:
                        logger.error(f"Failed to parse response for step {step_count + 1}: {e}")
                        continue
                        
                except Exception as api_error:
                    logger.error(f"API error on step {step_count + 1}: {api_error}")
                    break

            # Post-process validation
            if list_of_steps:
                # Ensure we have at least one valid legal step
                valid_steps = [step for step in list_of_steps if validate_legal_step(step)]
                if not valid_steps:
                    logger.error("No valid legal steps generated")
                    # Create a fallback step
                    fallback_step = Step(
                        reason="Fallback legal research step",
                        task=f"Research USCIS Policy Manual for {user_request[:100]}",
                        expectation="Find relevant immigration law information and requirements"
                    )
                    return [fallback_step]
                
                # Check overall plan quality
                if not check_step_dependencies(valid_steps):
                    logger.warning("Step dependencies check failed, but continuing with generated steps")
                
                return valid_steps
            else:
                logger.error("No steps generated")
                return []
            
        except Exception as e:
            logger.error(f"Critical error in _make_plan_sync: {e}")
            return []
    
    return await asyncio.to_thread(_make_plan_sync)


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

def env_true(var_name: str, default: bool = False) -> bool:
    """Check if environment variable is truthy."""
    value = os.getenv(var_name, "0")
    return value.lower() in ("1", "true", "yes", "on")

def bounded_env(min_val: int, max_val: int, var_name: str, default: int) -> int:
    """Get bounded environment variable value."""
    try:
        value = int(os.getenv(var_name, str(default)) or str(default))
        return max(min_val, min(max_val, value))
    except (ValueError, TypeError):
        return default

def is_complex(query: str) -> bool:
    """Determine if a legal query requires complex reasoning."""
    complexity_indicators = [
        r'\b(and|or|but|however|although|unless|if|when|while)\b',
        r'\b(traveled|abroad|outside|returned|left|came back)\b',
        r'\b(spouse|married|citizen|permanent resident|green card)\b',
        r'\b(years?|months?|days?|time|period|duration)\b',
        r'\b(eligible|qualify|requirements|criteria|exceptions?)\b',
        r'\b(continuous|residence|physical presence|absence)\b',
        r'\b(naturalization|citizenship|application|process)\b'
    ]
    
    # Count complexity indicators
    complexity_score = sum(1 for pattern in complexity_indicators 
                          if re.search(pattern, query, re.IGNORECASE))
    
    # Query is complex if it has multiple complexity factors or is long
    return complexity_score >= 3 or len(query.split()) >= 15

async def handle_prompt(messages: list[dict[str, str]]) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
    """Handle user prompt with enhanced legal response system."""
    executor = Executor()
    system_prompt = await get_system_prompt(messages, executor=executor)

    arm = AgentResourceManager()
    messages = refine_chat_history(messages, system_prompt, arm)

    use_cot = env_true("USE_SIMPLE_COT_PLANNER")
    user_request = messages[-1].get("content", "")
    
    logger.info(f"USE_SIMPLE_COT_PLANNER: {use_cot}, Query complexity: {is_complex(user_request)}")
    
    try:
        if use_cot and is_complex(user_request):
            logger.info("Using ENHANCED CoT mode for complex query")
            max_steps = bounded_env(1, 10, "SIMPLE_COT_MAX_STEPS", 5)
            steps = await make_plan(user_request, max_steps=max_steps)
            
            if steps:
                logger.info(f"Generated {len(steps)} research steps")
                answer = await execute_legal_steps_optimized(executor, steps, arm, time_budget=8)
                yield wrap_chunk(random_uuid(), answer, role='assistant')
                return
        
        # Either not complex, flag off, or no steps → fallback
        logger.info("Using QUICK retrieval mode")
        answer = await quick_retrieval_answer(user_request, arm, time_budget=3)
        yield wrap_chunk(random_uuid(), answer, role='assistant')
        return
        
    except Exception as e:
        logger.error(f"Error in enhanced mode, falling back: {e}")
        # Last-resort fallback
        try:
            answer = await quick_retrieval_answer(user_request, arm, time_budget=3)
            yield wrap_chunk(random_uuid(), answer, role='assistant')
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            basic_response = f"""**Legal Information Request: {user_request}**

I'm here to help with USCIS legal questions. Please try rephrasing your question or contact support if the issue persists.

**What to do next:**
- Rephrase your question more simply
- Check your internet connection
- Contact system administrator if the problem continues"""
            yield wrap_chunk(random_uuid(), basic_response, role='assistant')
        return

async def execute_legal_steps_optimized(executor: Executor, steps: list[Step], arm: AgentResourceManager, time_budget: int = 8) -> str:
    """Execute legal research steps with accuracy-focused optimizations."""
    output = ''
    legal_context = {}
    
    start_time = time.time()
    
    try:
        for i, step in enumerate(steps):
            # Check time budget
            if time.time() - start_time > time_budget:
                logger.warning(f"Time budget exceeded ({time_budget}s), stopping execution")
            break

            # Execute step with legal context from previous findings
            step_result = await execute_legal_step_optimized(executor, step, legal_context, arm)
            legal_context[f"step_{i+1}"] = step_result
            
            # Add step result to output with legal formatting
            output += f"\n\n**Legal Research Step {i+1}: {step.task}**\n{step_result}"
            
            # Legal quality check - stop when we have comprehensive legal coverage
            if await has_comprehensive_legal_coverage(output, steps[:i+1], legal_context):
                output += "\n\n*Legal research complete - comprehensive coverage achieved.*"
                break
        
        # Build coherent legal argument from research
        legal_argument = build_legal_argument(steps, legal_context)
        
        # Final legal synthesis with enhanced formatting
        output = await synthesize_legal_response(output, steps, legal_context, arm)
        
        # Add legal argument section
        output += "\n\n" + legal_argument
        
    except Exception as e:
        logger.error(f"Error while running legal executor: {str(e)}", exc_info=True)
        output = f"Error while running legal executor: {str(e)}"
    
    finally:
        output = refine_mcp_response(output, arm)
        output += "\n\n*This is general guidance, not legal advice.*"
    
    return output

async def execute_legal_step_optimized(executor: Executor, step: Step, legal_context: dict, arm: AgentResourceManager) -> str:
    """Execute a single legal research step with accuracy optimizations."""
    try:
        # Build legal context from previous steps for better search precision
        context = ""
        if legal_context:
            # Use last 2 legal findings to inform next search
            recent_findings = list(legal_context.values())[-2:]
            context = " ".join([str(finding) for finding in recent_findings])
        
        # Build more specific, diverse search queries to avoid duplicate results
        step_number = len(legal_context) + 1
        enhanced_query = build_specific_legal_query(step, legal_context, step_number)
        
        if context:
            enhanced_query += f" {context[:200]}"  # Add limited context
        
        logger.info(f"Searching for: {enhanced_query}")
        
        # Execute search with legal accuracy focus
        search_result = await asyncio.wait_for(
            executor.search(enhanced_query),  # Use default search behavior
            timeout=15.0  # Longer timeout for comprehensive legal research
        )
        
        # Format result for legal accuracy - handle different response structures
        if search_result:
            content = ""
            
            # Debug logging to see what we're getting
            logger.info(f"Search result type: {type(search_result)}")
            logger.info(f"Search result content: {str(search_result)[:200]}...")
            
            # Handle string response from executor.search()
            if isinstance(search_result, str):
                content = search_result
                logger.info(f"Extracted string content length: {len(content)}")
            # Handle ChatCompletion response structure (fallback)
            elif hasattr(search_result, 'choices') and search_result.choices:
                if hasattr(search_result.choices[0], 'delta') and search_result.choices[0].delta:
                    content = search_result.choices[0].delta.content or ""
                elif hasattr(search_result.choices[0], 'message') and search_result.choices[0].message:
                    content = search_result.choices[0].message.content or ""
                elif hasattr(search_result.choices[0], 'content'):
                    content = search_result.choices[0].content or ""
                logger.info(f"Extracted ChatCompletion content length: {len(content)}")
            
            # If we got content, execute legal analysis using LLM
            if content and len(content.strip()) > 10:
                # Clean and format the content
                cleaned_content = content.strip()
                
                # Score legal relevance
                relevance_score = score_legal_relevance(cleaned_content, enhanced_query)
                
                # Execute legal analysis using LLM instead of just returning raw content
                legal_analysis = await execute_legal_analysis(cleaned_content, step.task, enhanced_query)
                
                # Add legal relevance indicator
                relevance_indicator = "High" if relevance_score > 0.7 else "Medium" if relevance_score > 0.4 else "Low"
                
                return f"**Legal Research Results (Relevance: {relevance_indicator})**\n{legal_analysis}"
            else:
                # No meaningful content found, provide a more helpful message
                return f"**Legal Research Completed:** {step.task}\n\n*Note: Search completed but no specific legal content was found. This may indicate the topic requires more specific search terms or the information is not in the current database.*"
        else:
            return f"**Legal Research Completed:** {step.task}\n\n*Note: Search completed but no results were returned.*"
            
    except asyncio.TimeoutError:
        return f"**Legal Research Timeout:** {step.task}\n\n*Note: Search timed out after 15 seconds. This may indicate the query is too complex or the system is under heavy load.*"
    except Exception as e:
        logger.error(f"Error executing legal step {step.task}: {e}")
        return f"**Legal Research Error:** {step.task}\n\n*Note: An error occurred during search execution. Please try rephrasing your question.*"

async def execute_legal_analysis(legal_content: str, step_task: str, search_query: str) -> str:
    """Execute legal analysis using LLM to provide comprehensive legal guidance."""
    try:
        # Create a focused prompt for legal analysis
        analysis_prompt = f"""You are a legal expert specializing in USCIS immigration law. 

Analyze the following legal content and provide comprehensive, actionable legal guidance:

**Research Task:** {step_task}
**Search Query:** {search_query}

**Legal Content to Analyze:**
{legal_content[:4000]}  # Limit content length for LLM processing

**Your Task:**
1. Extract key legal requirements and procedures
2. Identify specific eligibility criteria
3. Provide actionable steps and documentation needs
4. Cite relevant legal sections and forms
5. Give practical advice and timelines

Provide a structured, comprehensive legal analysis that a person can actually use to take action.

**Legal Analysis:**"""

        # Use OpenAI to analyze the legal content
        client = openai.OpenAI(api_key=settings.llm_api_key)
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You are a USCIS legal expert providing comprehensive, actionable legal guidance."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=2000,
            temperature=0.1  # Low temperature for consistent legal analysis
        )
        
        if response.choices and response.choices[0].message:
            analysis = response.choices[0].message.content
            return analysis.strip()
        else:
            # Fallback to structured content if LLM fails
            return f"**Legal Analysis:**\n{legal_content[:2000]}..."
            
    except Exception as e:
        logger.error(f"Error executing legal analysis: {e}")
        # Fallback to structured content if LLM analysis fails
        return f"**Legal Content:**\n{legal_content[:2000]}..."

def score_legal_relevance(content: str, query: str) -> float:
    """Score the legal relevance of search results."""
    if not content or not query:
        return 0.0
    
    score = 0.0
    content_lower = content.lower()
    query_lower = query.lower()
    
    # Extract legal terms from query
    query_terms = extract_legal_terms(query)
    
    # Score based on legal term matches
    for term in query_terms:
        if term.lower() in content_lower:
            score += 0.2  # Each legal term match adds 0.2
    
    # Score based on USCIS-specific content
    uscis_indicators = ['uscis', 'policy manual', 'immigration', 'naturalization', 'citizenship']
    for indicator in uscis_indicators:
        if indicator in content_lower:
            score += 0.1
    
    # Score based on legal citation presence
    citation_patterns = [r'INA\s+\d+', r'Section\s+\d+', r'Chapter\s+\d+', r'[A-Z]-\d+']
    for pattern in citation_patterns:
        import re
        if re.search(pattern, content, re.IGNORECASE):
            score += 0.15
    
    # Score based on content length (prefer substantial content)
    if len(content) > 500:
        score += 0.1
    elif len(content) > 200:
        score += 0.05
    
    # Normalize score to 0.0-1.0 range
    return min(1.0, score)

async def has_comprehensive_legal_coverage(output: str, completed_steps: list[Step], legal_context: dict) -> bool:
    """Check if we have comprehensive legal coverage for the query."""
    try:
        # Check if we have substantial legal information
        if len(output) < 500:  # Reduced minimum for legal coverage
            return False
        
        # Check if we have at least 3 steps completed
        if len(completed_steps) < 3:
            return False
        
        # Extract legal terms from all steps and output
        all_legal_terms = set()
        for step in completed_steps:
            step_terms = extract_legal_terms(step.task)
            all_legal_terms.update(step_terms)
        
        output_terms = extract_legal_terms(output)
        all_legal_terms.update(output_terms)
        
        # Check if we have diverse legal information
        legal_keywords = ['eligibility', 'requirements', 'forms', 'process', 'documentation', 'timeline', 'fees']
        keyword_coverage = sum(1 for keyword in legal_keywords if keyword.lower() in output.lower())
        
        # Check for legal citations and forms
        has_citations = any('INA' in term for term in all_legal_terms)
        has_forms = any('-' in term and term[0].isalpha() for term in all_legal_terms)
        
        # Need at least 4 legal aspects covered and some citations/forms
        basic_coverage = keyword_coverage >= 4
        citation_coverage = has_citations or has_forms
        
        return basic_coverage and citation_coverage
        
    except Exception as e:
        logger.error(f"Error checking legal coverage: {e}")
        return False

async def synthesize_comprehensive_legal_guidance(research_output: str, completed_steps: list[Step], legal_context: dict) -> str:
    """Use LLM to synthesize comprehensive legal guidance from all research steps."""
    try:
        # Create a comprehensive synthesis prompt
        synthesis_prompt = f"""You are a senior USCIS legal expert. 

Synthesize all the legal research into comprehensive, actionable guidance for the user.

**Research Steps Completed:**
{chr(10).join([f"{i+1}. {step.task}" for i, step in enumerate(completed_steps)])}

**Legal Research Output:**
{research_output[:3000]}

**Your Task:**
Create a comprehensive legal analysis that:
1. **Directly answers the user's question** with specific legal guidance
2. **Provides actionable steps** they can take immediately
3. **Lists required documents and forms** with specific details
4. **Gives timelines and processing information** 
5. **Addresses their specific situation** based on the research
6. **Cites relevant legal authorities** (INA sections, forms, etc.)
7. **Provides practical advice** for their case

**Format your response as:**
- **Eligibility Assessment:** [Their specific eligibility]
- **Required Actions:** [Step-by-step what they need to do]
- **Documentation Needed:** [Specific forms and evidence]
- **Timeline:** [How long each step takes]
- **Legal Basis:** [Relevant laws and regulations]
- **Practical Tips:** [Real-world advice for their situation]

Make this guidance specific, actionable, and immediately useful for the user's legal situation.

**Comprehensive Legal Guidance:**"""

        # Use OpenAI for comprehensive synthesis
        client = openai.OpenAI(api_key=settings.llm_api_key)
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": "You are a senior USCIS legal expert providing comprehensive, actionable legal guidance."},
                {"role": "user", "content": synthesis_prompt}
            ],
            max_tokens=3000,
            temperature=0.1  # Low temperature for consistent legal guidance
        )
        
        if response.choices and response.choices[0].message:
            guidance = response.choices[0].message.content
            return guidance.strip()
        else:
            # Fallback to structured summary
            return f"**Legal Guidance Summary:**\n{research_output[:2000]}..."
            
    except Exception as e:
        logger.error(f"Error synthesizing comprehensive guidance: {e}")
        # Fallback to structured summary if LLM synthesis fails
        return f"**Legal Guidance Summary:**\n{research_output[:2000]}..."

async def synthesize_legal_response(output: str, steps: list[Step], legal_context: dict, arm: AgentResourceManager) -> str:
    """Synthesize final legal response from all research steps."""
    try:
        # Create a legal research summary
        summary = f"**Legal Research Summary**\n"
        summary += f"Successfully completed {len(steps)} legal research steps:\n"
        
        for i, step in enumerate(steps):
            summary += f"{i+1}. {step.task}\n"
        
        # Add legal insights
        summary += f"\n**Key Legal Insights:**\n"
        
        # Extract key legal points from steps
        key_insights = []
        for step in steps:
            step_lower = step.task.lower()
            if 'eligibility' in step_lower:
                key_insights.append("Eligibility requirements identified")
            if 'process' in step_lower or 'application' in step_lower:
                key_insights.append("Application process outlined")
            if 'requirements' in step_lower or 'documentation' in step_lower:
                key_insights.append("Documentation requirements specified")
            if 'timeline' in step_lower or 'time' in step_lower:
                key_insights.append("Processing timeline established")
            if 'fees' in step_lower or 'cost' in step_lower:
                key_insights.append("Fee structure documented")
        
        if key_insights:
            for insight in key_insights:
                summary += f"• {insight}\n"
        else:
            summary += "• Legal requirements and procedures identified\n"
        
        # Add legal recommendations
        summary += f"\n**Legal Recommendations:**\n"
        summary += "• Review all requirements carefully before proceeding\n"
        summary += "• Consult with an immigration attorney for complex cases\n"
        summary += "• Keep copies of all documentation and correspondence\n"
        summary += "• Monitor USCIS processing times and policy updates\n"
        
        # Use LLM to synthesize comprehensive legal guidance from all research
        comprehensive_guidance = await synthesize_comprehensive_legal_guidance(output, steps, legal_context)
        
        summary += f"\n**Comprehensive Legal Guidance**\n{comprehensive_guidance}"
        
        return summary
        
    except Exception as e:
        logger.error(f"Error synthesizing legal response: {e}")
        return output

async def extract_legal_intent(query: str) -> dict:
    """Extract legal intent from user query - optimized for speed and accuracy"""
    legal_patterns = {
        "eligibility": r"(eligible|qualify|requirements|criteria|qualification|must have|need to have)",
        "process": r"(how to|process|steps|procedure|apply|application|submit|file)",
        "timeline": r"(how long|time|duration|processing time|wait|schedule|when|deadline)",
        "documents": r"(documents|forms|evidence|proof|required|submit|provide|show)",
        "appeals": r"(appeal|denied|rejected|challenge|reconsideration|motion|review)",
        "fees": r"(cost|fee|payment|money|price|how much|total cost|expenses)",
        "status": r"(status|check|track|current|pending|approved|denied|processing)",
        "renewal": r"(renew|extension|continue|maintain|extend|continue|reapply)",
        "change": r"(change|modify|update|correct|amend|adjust|revise|edit)",
        "travel": r"(travel|leave|absence|return|reentry|permit|advance parole)",
        "family": r"(spouse|husband|wife|marriage|family|children|parents|siblings)",
        "employment": r"(work|job|employment|employer|sponsor|labor|occupation)",
        "criminal": r"(criminal|arrest|conviction|record|background|good moral character)",
        "education": r"(education|school|university|degree|student|academic|study)"
    }
    
    intent = {}
    query_lower = query.lower()
    
    # Check for legal topic areas
    legal_topics = {
        "naturalization": r"(naturalization|citizenship|become citizen|apply for citizenship)",
        "green_card": r"(green card|permanent resident|lpr|permanent residence)",
        "visa": r"(visa|temporary|nonimmigrant|immigrant visa)",
        "asylum": r"(asylum|refugee|protection|persecution)",
        "deportation": r"(deportation|removal|deport|leave|exit)"
    }
    
    # Extract basic intent patterns
    for category, pattern in legal_patterns.items():
        if re.search(pattern, query_lower, re.IGNORECASE):
            intent[category] = True
    
    # Extract legal topics
    for topic, pattern in legal_topics.items():
        if re.search(pattern, query_lower, re.IGNORECASE):
            intent["legal_topic"] = topic
            break
    
    # Extract urgency level
    urgency_patterns = {
        "urgent": r"(urgent|asap|immediately|right away|emergency|critical)",
        "timeline_sensitive": r"(deadline|due date|expiring|expires|soon|quickly)",
        "planning": r"(planning|future|later|eventually|someday|considering)"
    }
    
    for urgency, pattern in urgency_patterns.items():
        if re.search(pattern, query_lower, re.IGNORECASE):
            intent["urgency"] = urgency
            break
    
    # Extract complexity indicators
    complexity_indicators = {
        "simple": r"(simple|basic|general|overview|summary)",
        "complex": r"(complex|complicated|detailed|specific|technical|legal)",
        "case_specific": r"(my case|my situation|specific|particular|unique)"
    }
    
    for complexity, pattern in complexity_indicators.items():
        if re.search(pattern, query_lower, re.IGNORECASE):
            intent["complexity"] = complexity
            break

    return intent
async def quick_retrieval_answer(query: str, arm: AgentResourceManager, time_budget: int = 3) -> str:
    """Provide quick legal answer without complex planning."""
    try:
        # Simple search for immediate answer
        executor = Executor()
        search_result = await asyncio.wait_for(
            executor.search(query),
            timeout=time_budget
        )
        
        if search_result and hasattr(search_result, 'choices'):
            content = search_result.choices[0].delta.content or ""
            return f"""**Quick Legal Answer**

{content}

*This is a quick response. For comprehensive analysis, enable enhanced mode.*"""
        else:
            return f"""**Legal Information Request: {query}**

I'm here to help with USCIS legal questions. This is a basic response.

*For detailed legal research, please enable the enhanced response system.*"""
            
    except Exception as e:
        logger.error(f"Quick retrieval failed: {e}")
        return f"""**Legal Information Request: {query}**

I'm here to help with USCIS legal questions. Please try rephrasing your question.

*For detailed legal research, please enable the enhanced response system.*"""

def validate_legal_step(step: Step) -> bool:
    """Validate that a generated step is legally relevant and actionable."""
    if not step.task or not step.task.strip():
        return False
    
    # Check for legal relevance
    legal_keywords = [
        'uscis', 'policy', 'ina', 'naturalization', 'lpr', 'visa', 'citizenship',
        'green card', 'permanent resident', 'immigration', 'law', 'regulation',
        'form', 'application', 'eligibility', 'requirements', 'process'
    ]
    
    task_lower = step.task.lower()
    has_legal_keywords = any(keyword in task_lower for keyword in legal_keywords)
    
    # Check for actionable content
    action_verbs = ['search', 'find', 'research', 'identify', 'analyze', 'examine', 'review']
    has_action = any(verb in task_lower for verb in action_verbs)
    
    # Check for specific legal terms
    has_specific_terms = any(term in task_lower for term in ['section', 'chapter', 'volume', 'form', 'ina'])
    
    return has_legal_keywords and has_action and (has_specific_terms or len(step.task) > 20)

def extract_legal_terms(text: str) -> list[str]:
    """Extract specific legal terms, citations, and form numbers from text."""
    import re
    
    legal_terms = []
    
    # Extract INA citations (e.g., INA 316, INA 319)
    ina_citations = re.findall(r'INA\s+\d+[A-Z]?', text, re.IGNORECASE)
    legal_terms.extend(ina_citations)
    
    # Extract form numbers (e.g., N-400, I-485, I-751)
    form_numbers = re.findall(r'[A-Z]-\d+', text, re.IGNORECASE)
    legal_terms.extend(form_numbers)
    
    # Extract section numbers (e.g., Section 316, Chapter 3)
    section_refs = re.findall(r'(?:Section|Chapter|Volume)\s+\d+[A-Z]?', text, re.IGNORECASE)
    legal_terms.extend(section_refs)
    
    # Extract specific legal concepts
    legal_concepts = [
        'continuous residence', 'physical presence', 'good moral character',
        'naturalization', 'citizenship', 'permanent resident', 'green card',
        'spouse', 'marriage', 'travel', 'absence', 'eligibility', 'requirements'
    ]
    
    for concept in legal_concepts:
        if concept in text.lower():
            legal_terms.append(concept)
    
    return list(set(legal_terms))  # Remove duplicates

def build_specific_legal_query(step: Step, context: dict, step_number: int) -> str:
    """Build specific, diverse legal search queries to avoid duplicate results."""
    legal_terms = extract_legal_terms(step.task)
    
    # Create step-specific query variations to get diverse results
    step_variations = {
        1: "requirements eligibility criteria",
        2: "process procedures steps", 
        3: "documentation forms evidence",
        4: "timeline processing time",
        5: "case law precedents decisions",
        6: "policy updates recent changes",
        7: "appeals denials exceptions",
        8: "fees costs payment",
        9: "interview test preparation",
        10: "travel absence reentry"
    }
    
    # Use step-specific focus areas
    if step_number <= len(step_variations):
        focus_area = step_variations[step_number]
    else:
        focus_area = "requirements process"
    
    # Build diverse query
    if legal_terms:
        primary_terms = legal_terms[:2]  # Limit to avoid overly long queries
        query = f"{' '.join(primary_terms)} {focus_area}"
    else:
        # Extract key concepts from task
        task_lower = step.task.lower()
        if 'naturalization' in task_lower:
            query = f"naturalization {focus_area} USCIS Policy Manual"
        elif 'citizenship' in task_lower:
            query = f"citizenship {focus_area} USCIS Policy Manual"
        elif 'form' in task_lower:
            query = f"USCIS forms {focus_area} Policy Manual"
        else:
            query = f"immigration {focus_area} USCIS Policy Manual"
    
    # Add step-specific modifiers to ensure diversity
    step_modifiers = ["specific", "detailed", "comprehensive", "current", "recent", "practical"]
    modifier = step_modifiers[step_number % len(step_modifiers)]
    
    return f"{query} {modifier}"

def build_legal_search_query(step: Step, context: dict) -> str:
    """Build optimized legal search queries with specific legal terms."""
    legal_terms = extract_legal_terms(step.task)
    
    # Prioritize legal terms
    if legal_terms:
        # Use most relevant legal terms first
        primary_terms = legal_terms[:3]
        query = " ".join(primary_terms)
        
        # Add USCIS context
        query += " USCIS Policy Manual"
        
        # Add relevant context from previous steps
        if context:
            recent_context = list(context.values())[-1] if context else ""
            if recent_context and len(recent_context) > 50:
                # Extract key terms from recent context
                context_terms = extract_legal_terms(str(recent_context))
                if context_terms:
                    query += " " + " ".join(context_terms[:2])
    else:
        # Fallback to enhanced generic search
        query = f"USCIS Policy Manual {step.task}"
    
    return query

def build_legal_argument(steps: list[Step], results: dict) -> str:
    """Build a coherent legal argument from research steps and results."""
    if not steps or not results:
        return "Insufficient information to build legal argument."
    
    argument = "**Legal Analysis Summary**\n\n"
    
    # Build logical progression
    for i, step in enumerate(steps, 1):
        step_key = f"step_{i}"
        if step_key in results:
            result = results[step_key]
            argument += f"**Step {i}: {step.task}**\n"
            argument += f"*Finding: {result[:300]}{'...' if len(result) > 300 else ''}*\n\n"
    
    # Add legal reasoning
    argument += "**Legal Reasoning:**\n"
    argument += "Based on the above research, the legal requirements and processes are as follows:\n\n"
    
    # Extract key legal points
    key_points = []
    for step in steps:
        if 'eligibility' in step.task.lower():
            key_points.append("Eligibility requirements")
        if 'process' in step.task.lower() or 'application' in step.task.lower():
            key_points.append("Application process")
        if 'requirements' in step.task.lower():
            key_points.append("Documentation requirements")
        if 'timeline' in step.task.lower() or 'time' in step.task.lower():
            key_points.append("Processing timeline")
    
    if key_points:
        for point in key_points:
            argument += f"• {point}\n"
    else:
        argument += "• Legal requirements and procedures\n"
    
    return argument

def check_step_dependencies(steps: list[Step]) -> bool:
    """Check if steps build logically upon each other."""
    if len(steps) < 2:
        return True
    
    for i in range(1, len(steps)):
        current_step = steps[i]
        previous_step = steps[i-1]
        
        # Check if current step references previous step's output
        current_lower = current_step.task.lower()
        previous_lower = previous_step.task.lower()
        
        # Look for logical connections
        has_connection = False
        
        # Check for topic continuity
        if any(term in current_lower for term in ['then', 'next', 'following', 'based on', 'after']):
            has_connection = True
        
        # Check for related legal concepts
        current_terms = extract_legal_terms(current_step.task)
        previous_terms = extract_legal_terms(previous_step.task)
        
        if set(current_terms) & set(previous_terms):  # Intersection
            has_connection = True
        
        if not has_connection:
            logger.warning(f"Step {i+1} may not logically follow from step {i}")
            return False
    
    return True


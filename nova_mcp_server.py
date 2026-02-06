#!/usr/bin/env python3
"""
NOVA MCP Security Gateway
Author: Thomas Roccia (@fr0gger_)

THIS SERVER MUST RUN FIRST IN THE MCP CHAIN
It validates all prompts against security rules before they reach the LLM.
"""

import os
import sys
from dotenv import load_dotenv
import datetime
import logging
from pathlib import Path
from typing import Any, List, Dict, Optional

import uuid
import time
import json
import re
from collections import defaultdict

# Load environment variables from a .env file, so OPENAI_API_KEY is set
load_dotenv()

# Global session tracking
session_store = {}
SESSION_TIMEOUT = 1800  # 30 minutes in seconds

from mcp.server.fastmcp import FastMCP

# Import Nova components
try:
    from nova.core.parser import NovaParser
    from nova.core.rules import NovaRule, KeywordPattern, SemanticPattern, LLMPattern
    from nova.core.matcher import NovaMatcher
    from nova.evaluators.llm import (
        OpenAIEvaluator,
        AnthropicEvaluator,
        AzureOpenAIEvaluator,
        OllamaEvaluator,
        GroqEvaluator
    )
except ImportError:
    print("Error: Nova package not found in PYTHONPATH.")
    print("Make sure Nova is installed or set your PYTHONPATH correctly.")
    sys.exit(1)


# Suppress sentence-transformers logs
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Suppress huggingface/transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize FastMCP server
mcp = FastMCP("nova-security")

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_DIR = os.path.join(SCRIPT_DIR, "nova_rules")

# Get user home directory (should be writable)
#HOME_DIR = str(Path.home())
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

# Print configuration
print(f"NOVA MCP SECURITY GATEWAY INITIALIZING")
print(f"IMPORTANT: This server must be configured to run FIRST in the MCP chain")
print(f"Using rules directory: {RULES_DIR}")
print(f"Using logs directory: {LOG_DIR}")

# Setup logging
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "nova_matches.log")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stderr)
        ]
    )
    print(f"Logging to file: {LOG_FILE}")
    # Log to file that we are using this log path
    logging.getLogger("nova-mcp-server").info(f"Logging to file: {LOG_FILE}")
except Exception as e:
    # Fall back to stderr only if file handler cannot be created
    print(f"Warning: Could not create log directory: {e}")
    # Log warning to stderr only since file handler failed
    logging.getLogger("nova-mcp-server").warning(f"Could not create log directory: {e}")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )

logger = logging.getLogger("nova-mcp-server")

# Add these lines to fix the file logging:
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# Attach file handler to nova-mcp-server logger
logger.addHandler(file_handler)
# Log startup messages to file
logger.info("NOVA MCP SECURITY GATEWAY INITIALIZING")
logger.info("IMPORTANT: This server must be configured to run FIRST in the MCP chain")
logger.info(f"Using rules directory: {RULES_DIR}")
logger.info(f"Using logs directory: {LOG_DIR}")

# Determine default LLM evaluator based on available API keys
def _select_llm_evaluator():
    # Priority: OpenAI, Anthropic, Azure OpenAI, Ollama, Groq
    # Log available environment keys for debugging
    logger.debug(
        "LLM env keys: OPENAI={openai}, ANTHROPIC={anthropic}, AZURE_OPENAI={azure_key}, AZURE_ENDPOINT={azure_endpoint}, OLLAMA={ollama}, GROQ={groq}".format(
            openai='set' if os.getenv('OPENAI_API_KEY') else 'unset',
            anthropic='set' if os.getenv('ANTHROPIC_API_KEY') else 'unset',
            azure_key='set' if os.getenv('AZURE_OPENAI_API_KEY') else 'unset',
            azure_endpoint='set' if os.getenv('AZURE_OPENAI_ENDPOINT') else 'unset',
            ollama='set' if os.getenv('OLLAMA_HOST') else 'unset',
            groq='set' if os.getenv('GROQ_API_KEY') else 'unset'
        )
    )
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        model = os.getenv("OPENAI_MODEL")
        return OpenAIEvaluator(api_key=openai_key, model=model) if model else OpenAIEvaluator(api_key=openai_key)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        model = os.getenv("ANTHROPIC_MODEL")
        return AnthropicEvaluator(api_key=anthropic_key, model=model) if model else AnthropicEvaluator(api_key=anthropic_key)
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_key and azure_endpoint:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        if deployment:
            return AzureOpenAIEvaluator(api_key=azure_key, endpoint=azure_endpoint, deployment_name=deployment)
        return AzureOpenAIEvaluator(api_key=azure_key, endpoint=azure_endpoint, deployment_name="gpt-35-turbo")
    ollama_host = os.getenv("OLLAMA_HOST")
    if ollama_host:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        return OllamaEvaluator(host=ollama_host, model=model)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        model = os.getenv("GROQ_MODEL")
        return GroqEvaluator(api_key=groq_key, model=model) if model else GroqEvaluator(api_key=groq_key)
    return None

# Instantiate a global LLM evaluator (once per server)
default_llm_evaluator = _select_llm_evaluator()
if default_llm_evaluator:
    logger.info(f"Using LLM evaluator: {default_llm_evaluator.__class__.__name__}, model={getattr(default_llm_evaluator, 'model', None)}")
else:
    logger.warning("No LLM evaluator configured; LLM patterns will be skipped.")


def extract_rules(content: str) -> List[str]:
    """
    Extract individual rule blocks from a file containing multiple rules.

    Args:
        content: String containing multiple rule definitions

    Returns:
        List of strings, each containing a single rule
    """
    # Pattern to find rule declarations
    rule_start_pattern = r'rule\s+\w+\s*{?'
    rule_starts = [m.start() for m in re.finditer(rule_start_pattern, content)]

    if not rule_starts:
        return []

    # Extract each rule block
    rule_blocks = []

    for i in range(len(rule_starts)):
        start = rule_starts[i]

        # End is either the start of the next rule or the end of the content
        end = rule_starts[i+1] if i < len(rule_starts) - 1 else len(content)

        # Extract the rule text
        rule_text = content[start:end].strip()
        rule_blocks.append(rule_text)

    return rule_blocks


def init_rule_attributes(rule):
    """
    Initialize all required attributes on a rule to ensure they exist.
    This matches how novarun.py handles rules.

    Args:
        rule: NovaRule object to initialize
    """
    # Make sure rule has all required attributes to avoid NoneType errors
    if not hasattr(rule, 'keywords') or rule.keywords is None:
        rule.keywords = {}

    if not hasattr(rule, 'semantics') or rule.semantics is None:
        rule.semantics = {}

    if not hasattr(rule, 'llms') or rule.llms is None:
        rule.llms = {}

    # Also make sure the condition exists
    if not hasattr(rule, 'condition'):
        rule.condition = ""

    return rule


def find_matching_rule(prompt: str) -> List[Dict[str, Any]]:
    """Check all rules until a match is found.
    Returns the first matching rule or empty list if none matches.
    The function continues checking if no match is found but stops on first match."""
    logger.debug(f"Checking prompt against rules: {prompt[:200]}...")

    # Check if rules directory exists
    if not os.path.isdir(RULES_DIR):
        logger.error(f"Rules directory not found: {RULES_DIR}")
        return []

    # Get all rule files from the directory
    rule_files = []
    for root, _, files in os.walk(RULES_DIR):
        for file in files:
            if file.endswith('.nov'):
                rule_files.append(os.path.join(root, file))

    logger.debug(f"Found {len(rule_files)} rule files: {rule_files}")

    if not rule_files:
        logger.warning(f"No rule files found in {RULES_DIR}")
        return []

    # Use the pre-selected LLM evaluator (None => skip LLM patterns)
    evaluator = default_llm_evaluator

    # Process each rule file
    for rule_file in rule_files:
        try:
            logger.info(f"Processing rule file: {rule_file}")

            # Load file content directly
            with open(rule_file, 'r') as f:
                file_content = f.read()
                logger.debug(f"Rule file content loaded: {len(file_content)} bytes")

            # Extract individual rules if multiple rules in file
            if 'rule ' in file_content.lower() and file_content.count('rule ') > 1:
                # Extract all rules from the file
                rule_blocks = extract_rules(file_content)
                logger.debug(f"Extracted {len(rule_blocks)} rule blocks from {rule_file}")
            else:
                rule_blocks = [file_content]
                logger.debug(f"Single rule in file {rule_file}")

            # Process each rule block independently
            for rule_idx, rule_text in enumerate(rule_blocks):
                try:
                    # Parse the rule
                    logger.debug(f"Parsing rule #{rule_idx+1} from {rule_file}...")
                    parser = NovaParser()
                    rule = parser.parse(rule_text)

                    if rule is None:
                        logger.error(f"Parsed rule is None from file {rule_file} - skipping")
                        continue

                    # Initialize all required attributes
                    rule = init_rule_attributes(rule)

                    # Log rule details
                    rule_name = rule.name
                    logger.debug(f"Successfully parsed rule: {rule_name}")
                    logger.debug(f"Rule attributes: keywords={len(rule.keywords)}, semantics={len(rule.semantics)}, llms={len(rule.llms)}")
                    logger.debug(f"Rule condition: {rule.condition}")

                    # Create a matcher for this rule (do not auto-create new LLM evaluator)
                    matcher = NovaMatcher(rule, llm_evaluator=evaluator, create_llm_evaluator=False)

                    # Manually check the prompt against the rule
                    try:
                        logger.debug(f"Checking prompt against rule {rule_name}...")

                        # EXACT COPY OF BEHAVIOR FROM NOVARUN
                        # If this rule uses LLM and we have an evaluator, explicitly check the LLM patterns
                        if rule.llms and evaluator:
                            for key, pattern in rule.llms.items():
                                logger.debug(f"Evaluating LLM pattern {key} with threshold {pattern.threshold}")
                                try:
                                    matched, confidence, details = evaluator.evaluate_prompt(
                                        pattern.pattern,
                                        prompt,
                                        temperature=pattern.threshold
                                    )
                                    # Log errors from evaluator details if present
                                    if isinstance(details, dict) and details.get('error'):
                                        err = details.get('error')
                                        logger.error(
                                            f"LLM evaluation error for pattern {key}: {err}. details={details}"
                                        )
                                    else:
                                        logger.debug(
                                            f"LLM pattern {key} result: matched={matched}, confidence={confidence}"
                                        )
                                except Exception as e:
                                    logger.error(f"Exception during LLM evaluation for pattern {key}: {e}")

                        # Now check the entire rule
                        result = matcher.check_prompt(prompt)

                        # Log the result
                        matched = result.get('matched', False)
                        logger.debug(f"Rule {rule_name} matched: {matched}")

                        # Extra debugging
                        if 'debug' in result:
                            logger.debug(f"Result debug info: {json.dumps(result['debug'], default=str)}")

                        # If matched, return immediately with this match
                        if matched:
                            # Add source file information
                            result['rule_file'] = rule_file
                            logger.info(f"Match found for rule {rule_name} in file: {rule_file}")
                            return [result]  # Return as list with single match
                        else:
                            logger.debug(f"Rule {rule_name} did not match, continuing to next rule")

                    except Exception as e:
                        logger.error(f"Error checking prompt against rule {rule_name}: {str(e)}")
                        import traceback
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                        continue

                except Exception as e:
                    rule_idx_str = str(rule_idx + 1)
                    logger.error(f"Error processing rule #{rule_idx_str} in {rule_file}: {str(e)}")
                    import traceback
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    continue

        except Exception as e:
            logger.error(f"Error processing rule file {rule_file}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            continue

    # If we get here, no rule matched
    logger.debug("No matching rules found across all files and rules")
    return []


def get_or_create_session(prompt_context=None):
    """Generate or retrieve a session ID based on context clues"""
    # Clean up expired sessions
    current_time = time.time()
    expired = [sid for sid, data in session_store.items()
               if current_time - data['last_activity'] > SESSION_TIMEOUT]
    for sid in expired:
        del session_store[sid]

    # Try to identify existing session from context clues
    # This is placeholder logic - adapt based on your specific context
    session_id = None

    # If no existing session found, create a new one
    if not session_id:
        session_id = f"novamcp_{uuid.uuid4().hex[:8]}"
        session_store[session_id] = {
            'created': current_time,
            'last_activity': current_time,
            'prompt_count': 0
        }
    else:
        # Update existing session
        session_store[session_id]['last_activity'] = current_time
        session_store[session_id]['prompt_count'] += 1

    return session_id

@mcp.tool()
async def validate_prompt(prompt: str, user_id: str = "unknown") -> str:
    """
    SECURITY CHECKPOINT: Validate prompt against NOVA security rules.

    Args:
        prompt: The prompt to check
        user_id: The identifier of the user submitting the prompt

    Returns:
        A message indicating if the prompt is allowed or blocked
    """
    try:
        # Generate or retrieve session ID
        session_id = get_or_create_session()

        # Debug check rule files
        rule_files = []
        for root, _, files in os.walk(RULES_DIR):
            for file in files:
                if file.endswith('.nov'):
                    rule_files.append(os.path.join(root, file))

        # Check for empty prompt
        if not prompt or prompt.strip() == "":
            logger.warning("Empty prompt received, skipping validation")
            return "AUTHORIZED"

        # Find matching rule
        result = find_matching_rule(prompt)

        if result and any(r.get('matched', False) for r in result):
            # Get rule information
            rule_name = result[0].get('rule_name', 'Unknown')
            meta = result[0].get('meta', {})
            description = meta.get('description', 'No description provided')
            severity = meta.get('severity', 'unknown')

            # Log the match
            try:
                log_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "event": "rule_match",
                    "user_id": user_id,
                    "rule_name": rule_name,
                    "severity": severity,
                    "description": description,
                    "prompt": prompt,
                }

                # Prepare structured log for security alert
                truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
                log_data.update({
                    "session_id": session_id,
                    "event": "SECURITY_ALERT",
                    "truncated_prompt": truncated_prompt
                })
                # Emit JSON-formatted warning for analytics
                logger.warning(json.dumps(log_data))
            except Exception as e:
                logger.error(f"Error logging match: {e}")

            # Return formatted response according to requirements
            return f"NOT AUTHORIZED\n\nYour prompt is not authorized.\n\nSecurity rule matched: {rule_name}\nDescription: {description}\nSeverity: {severity}\n\nThis request has been blocked by the NOVA security gateway."

        # Log the non-match
        try:
            truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            logger.info(f"SECURITY CHECKPOINT PASSED: [Session: {session_id}] No rules matched: \"{truncated_prompt}\"")
        except Exception as e:
            logger.error(f"Error logging validation: {e}")

        return "AUTHORIZED"

    except Exception as e:
        # Catch any unexpected errors
        error_msg = f"Error during security validation: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.debug(f"Validation error traceback: {traceback.format_exc()}")

        # Return a generic message - FAIL CLOSED for safety
        return "NOT AUTHORIZED\n\nYour prompt is not authorized.\n\nUnable to complete security validation. For safety, this request has been blocked."


def main():
    """Main entry point for the MCP server"""
    try:
        # Log server startup
        logger.info("NOVA MCP SECURITY GATEWAY STARTING")
        logger.info("This server must be configured to run FIRST in the MCP chain")

        # Check for rules directory and files
        if not os.path.isdir(RULES_DIR):
            logger.critical(f"CRITICAL ERROR: Rules directory not found: {RULES_DIR}")
            logger.critical("Server will start but NO RULES will be enforced!")
        else:
            rule_files = [f for f in os.listdir(RULES_DIR) if f.endswith('.nov')]
            if not rule_files:
                logger.critical(f"CRITICAL ERROR: No .nov files found in {RULES_DIR}")
                logger.critical("Server will start but NO RULES will be enforced!")
            else:
                logger.info(f"Found {len(rule_files)} rule files in {RULES_DIR}")

    except Exception as e:
        logger.error(f"Error during startup: {e}")

    # Initialize and run the server
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()

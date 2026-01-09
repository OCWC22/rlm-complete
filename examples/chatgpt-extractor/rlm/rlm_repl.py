"""
Simple Recursive Language Model (RLM) with REPL environment.
Includes cost tracking and iteration monitoring.
"""

from typing import Dict, List, Optional, Any 

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.llm import (
    OpenAIClient, AnthropicClient, 
    CostTracker, get_cost_tracker, set_cost_tracker
)
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils

from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


def get_llm_client(provider: str, api_key: Optional[str], model: str, cost_tracker: CostTracker):
    """Factory function to create the appropriate LLM client with cost tracking."""
    if provider == "anthropic":
        return AnthropicClient(
            api_key=api_key, 
            model=model, 
            call_type="root",
            cost_tracker=cost_tracker
        )
    elif provider == "openai":
        return OpenAIClient(api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: 'openai', 'anthropic'")


class RLM_REPL(RLM):
    """
    LLM Client that can handle long contexts by recursively calling itself.
    
    Features:
    - Cost tracking: Monitor API spending in real-time
    - Budget limits: Set max spending per session
    - Iteration control: Limit number of thinking steps
    - Visible logging: See exactly what the RLM is doing
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "gpt-5",
                 recursive_model: str = "gpt-5",
                 max_iterations: int = 20,
                 depth: int = 0,
                 enable_logging: bool = False,
                 provider: str = "openai",  # "openai" or "anthropic"
                 budget_limit: Optional[float] = None,  # Max $ to spend
                 ):
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.provider = provider
        self._max_iterations = max_iterations
        self.depth = depth  # Unused in this version
        
        # Initialize cost tracker with optional budget limit
        self.cost_tracker = CostTracker(budget_limit=budget_limit)
        set_cost_tracker(self.cost_tracker)
        
        # Create LLM client with cost tracking
        self.llm = get_llm_client(provider, api_key, model, self.cost_tracker)
        
        # Track recursive call depth to prevent infinite loops
        self.repl_env = None
        
        # Initialize colorful logger
        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)
        
        self.messages = []  # Initialize messages list
        self.query = None
        
        # Iteration tracking for visibility
        self._current_iteration = 0
    
    def setup_context(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None):
        """
        Setup the context for the RLMClient.

        Args:
            context: The large context to analyze in the form of a list of messages, string, or Dict
            query: The user's question
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        # Initialize the conversation with the REPL prompt
        self.messages = build_system_prompt()
        self.logger.log_initial_messages(self.messages)
        
        # Initialize REPL environment with context data and cost tracking
        context_data, context_str = utils.convert_context_for_repl(context)
        
        self.repl_env = REPLEnv(
            context_json=context_data, 
            context_str=context_str, 
            recursive_model=self.recursive_model,
            provider=self.provider,
            cost_tracker=self.cost_tracker,  # Pass cost tracker for unified tracking
        )
        
        return self.messages

    def completion(self, context: List[str] | str | List[Dict[str, str]], query: Optional[str] = None) -> str:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.
        
        Includes real-time cost tracking and iteration monitoring.
        """
        self.messages = self.setup_context(context, query)
        self._current_iteration = 0
        
        # Main loop runs for fixed # of root LM iterations
        for iteration in range(self._max_iterations):
            self._current_iteration = iteration + 1
            
            # Check budget before making call
            if not self.cost_tracker.check_budget():
                budget_msg = (
                    f"⚠️  Budget limit reached at iteration {iteration}! "
                    f"Spent ${self.cost_tracker.total_cost:.4f} of "
                    f"${self.cost_tracker.budget_limit:.2f}"
                )
                print(budget_msg)
                return f"[Budget exceeded] Query incomplete after {iteration} iterations."
            
            # Log iteration progress
            if self.logger.enabled:
                summary = self.cost_tracker.get_summary()
                print(f"\n{'─'*50}")
                print(f"🔄 Iteration {iteration + 1}/{self._max_iterations}")
                print(f"💰 Cost so far: ${summary['total_cost_usd']:.4f} | "
                      f"Calls: {summary['total_calls']} (root: {summary['root_calls']}, sub: {summary['sub_calls']})")
                print(f"{'─'*50}")
            
            # Query root LM to interact with REPL environment
            response = self.llm.completion(self.messages + [next_action_prompt(query, iteration)])
            
            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(response, has_tool_calls=code_blocks is not None)
            
            # Process code execution or add assistant message
            if code_blocks is not None:
                self.messages = utils.process_code_execution(
                    response, self.messages, self.repl_env, 
                    self.repl_env_logger, self.logger
                )
            else:
                # Add assistant message when there are no code blocks
                assistant_message = {"role": "assistant", "content": "You responded with:\n" + response}
                self.messages.append(assistant_message)
            
            # Check that model produced a final answer
            final_answer = utils.check_for_final_answer(
                response, self.repl_env, self.logger,
            )

            # In practice, you may need some guardrails here.
            if final_answer:
                self.logger.log_final_response(final_answer)
                return final_answer

            
        # If we reach here, no final answer was found in any iteration
        print("No final answer found in any iteration")
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.logger.log_final_response(final_answer)

        return final_answer
    
    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        return self.cost_tracker.get_summary()
    
    def print_cost_summary(self):
        """Print a formatted cost summary."""
        self.cost_tracker.print_summary()
    
    def print_call_log(self):
        """Print detailed log of all API calls."""
        self.cost_tracker.print_call_log()
    
    def get_iteration_count(self) -> int:
        """Get the number of iterations used in the last completion."""
        return self._current_iteration

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv()
        self.messages = []
        self.query = None


if __name__ == "__main__":
    pass

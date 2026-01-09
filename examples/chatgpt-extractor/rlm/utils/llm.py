"""
LLM Client wrappers for OpenAI and Anthropic models.
Includes cost tracking and usage monitoring.
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# COST TRACKING
# ============================================================================

# ============================================================================
# ANTHROPIC PRICING - Updated Jan 2026
# Source: https://docs.anthropic.com/en/docs/about-claude/models
# ============================================================================
#
# Claude 4.5 Family (Latest - Sep-Nov 2025)
#   Sonnet 4.5: $3/1M input, $15/1M output (200K context, 1M beta)
#   Haiku 4.5:  $1/1M input, $5/1M output (200K context)
#   Opus 4.5:   $5/1M input, $25/1M output (200K context)
#
# Long context pricing (>200K tokens): 2x input cost
# ============================================================================

ANTHROPIC_PRICING = {
    # Claude 4.5 family (Latest - Sep/Oct/Nov 2025)
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},  # Alias
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},  # Alias
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},  # Alias
    
    # Claude 4.1 family (Aug 2025)
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 75.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},
    
    # Claude 4 family (May 2025)
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-0": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-opus-4-0": {"input": 15.00, "output": 75.00},
    
    # Claude 3.7 family (Feb 2025)
    "claude-3-7-sonnet-20250219": {"input": 3.00, "output": 15.00},
    
    # Claude 3.5 family (2024)
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    
    # Claude 3 family (2024)
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    
    # Default fallback (Sonnet 4.5 pricing)
    "default": {"input": 3.00, "output": 15.00},
}

OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-5": {"input": 5.00, "output": 20.00},
    "gpt-5-mini": {"input": 1.00, "output": 4.00},
    "gpt-5-nano": {"input": 0.50, "output": 2.00},
    "default": {"input": 5.00, "output": 20.00},
}


@dataclass
class APICallRecord:
    """Record of a single API call for tracking."""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    call_type: str  # "root" or "sub"
    
    
@dataclass
class CostTracker:
    """
    Tracks all API calls and their costs.
    Provides visibility into token usage and spending.
    """
    calls: list = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    budget_limit: Optional[float] = None  # Max spending allowed
    
    def record_call(
        self, 
        model: str, 
        input_tokens: int, 
        output_tokens: int,
        pricing: dict,
        call_type: str = "root"
    ) -> APICallRecord:
        """Record an API call and calculate costs."""
        model_pricing = pricing.get(model, pricing.get("default"))
        
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        record = APICallRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            call_type=call_type
        )
        
        self.calls.append(record)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        
        return record
    
    def check_budget(self) -> bool:
        """Check if we're within budget. Returns False if over budget."""
        if self.budget_limit is None:
            return True
        return self.total_cost < self.budget_limit
    
    def get_summary(self) -> dict:
        """Get a summary of all costs."""
        return {
            "total_calls": len(self.calls),
            "root_calls": len([c for c in self.calls if c.call_type == "root"]),
            "sub_calls": len([c for c in self.calls if c.call_type == "sub"]),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": self.total_cost,
            "budget_limit": self.budget_limit,
            "budget_remaining": (self.budget_limit - self.total_cost) if self.budget_limit else None,
        }
    
    def print_summary(self):
        """Print a formatted cost summary."""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("💰 COST SUMMARY")
        print("=" * 50)
        print(f"Total API Calls:     {summary['total_calls']}")
        print(f"  - Root LM calls:   {summary['root_calls']}")
        print(f"  - Sub-LM calls:    {summary['sub_calls']}")
        print(f"Input Tokens:        {summary['total_input_tokens']:,}")
        print(f"Output Tokens:       {summary['total_output_tokens']:,}")
        print(f"Total Tokens:        {summary['total_tokens']:,}")
        print(f"Total Cost:          ${summary['total_cost_usd']:.4f}")
        if summary['budget_limit']:
            print(f"Budget Limit:        ${summary['budget_limit']:.2f}")
            print(f"Budget Remaining:    ${summary['budget_remaining']:.4f}")
        print("=" * 50)
    
    def print_call_log(self):
        """Print detailed log of all calls."""
        print("\n📋 CALL LOG")
        print("-" * 80)
        for i, call in enumerate(self.calls, 1):
            print(f"{i:3}. [{call.call_type:4}] {call.model}")
            print(f"     Tokens: {call.input_tokens:,} in / {call.output_tokens:,} out")
            print(f"     Cost: ${call.total_cost:.6f}")
        print("-" * 80)


# Global cost tracker (can be replaced per session)
_global_cost_tracker: Optional[CostTracker] = None

def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker, creating one if needed."""
    global _global_cost_tracker
    if _global_cost_tracker is None:
        _global_cost_tracker = CostTracker()
    return _global_cost_tracker

def set_cost_tracker(tracker: CostTracker):
    """Set a new global cost tracker."""
    global _global_cost_tracker
    _global_cost_tracker = tracker

def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _global_cost_tracker
    _global_cost_tracker = CostTracker()


class OpenAIClient:
    """OpenAI Client wrapper for GPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5"):
        from openai import OpenAI
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")


class AnthropicClient:
    """
    Anthropic Client wrapper for Claude models.
    Includes automatic cost tracking.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "claude-sonnet-4-20250514",
        call_type: str = "root",  # "root" or "sub" for tracking
        cost_tracker: Optional[CostTracker] = None,
    ):
        import anthropic
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.call_type = call_type
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Use provided tracker or global tracker
        self.cost_tracker = cost_tracker if cost_tracker else get_cost_tracker()
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = 4096,
        **kwargs
    ) -> str:
        """
        Generate a completion using Anthropic's Claude API.
        
        Converts OpenAI-style messages to Anthropic format:
        - Extracts 'system' messages into the system parameter
        - Passes remaining messages as user/assistant turns
        
        Automatically tracks costs.
        """
        try:
            # Check budget before making call
            if not self.cost_tracker.check_budget():
                raise RuntimeError(
                    f"Budget limit exceeded! "
                    f"Spent ${self.cost_tracker.total_cost:.4f} of "
                    f"${self.cost_tracker.budget_limit:.2f} budget."
                )
            
            # Convert string input to message format
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]
            
            # Anthropic requires system messages to be passed separately
            system_content = ""
            chat_messages = []
            
            for msg in messages:
                if msg.get("role") == "system":
                    # Concatenate system messages
                    if system_content:
                        system_content += "\n\n"
                    system_content += msg.get("content", "")
                else:
                    chat_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Ensure we have at least one user message
            if not chat_messages:
                chat_messages = [{"role": "user", "content": "Hello"}]
            
            # Anthropic requires alternating user/assistant messages
            # and must start with user. Merge consecutive same-role messages.
            merged_messages = []
            for msg in chat_messages:
                if merged_messages and merged_messages[-1]["role"] == msg["role"]:
                    merged_messages[-1]["content"] += "\n\n" + msg["content"]
                else:
                    merged_messages.append(msg.copy())
            
            # Ensure first message is from user
            if merged_messages and merged_messages[0]["role"] != "user":
                merged_messages.insert(0, {"role": "user", "content": "Please continue."})
            
            # Remove timeout from kwargs if present (Anthropic doesn't use it the same way)
            kwargs.pop("timeout", None)
            
            # Build API call parameters - only include system if we have content
            api_params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": merged_messages,
                **kwargs
            }
            
            # Only add system parameter if we have system content
            # Anthropic API errors if system=None is passed
            if system_content:
                api_params["system"] = system_content
            
            response = self.client.messages.create(**api_params)
            
            # Track costs from response usage
            if hasattr(response, 'usage'):
                self.cost_tracker.record_call(
                    model=self.model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    pricing=ANTHROPIC_PRICING,
                    call_type=self.call_type
                )
            
            # Extract text from response
            return response.content[0].text

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
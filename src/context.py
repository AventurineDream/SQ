import tiktoken
from typing import List, Dict, Any

from schema import Event, EventType
from db import LedgerDB


class ContextAssembler:
    """Dynamically builds token-aware context windows for the Orchestrator and Actors.
    
    Replaces static summaries with literal, chronological transcripts of the
    shared canon, constrained only by the model's token limits.
    """

    def __init__(self, db: LedgerDB, model_name: str = "gpt-oss:20b", max_tokens: int = 4096):
        self.db = db
        self.model_name = model_name
        
        # We use a base token budget for the context window. 
        # A safety buffer is subtracted to leave room for system prompts and the model's output generation.
        self.max_context_tokens = int(max_tokens * 0.75) 
        
        # In a real setup, you would load the specific tokenizer for Qwen or Mistral.
        # For this prototype, we use standard tiktoken as a reliable proxy for token counts.
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4") # Proxy tokenizer
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Returns the token count for a given string."""
        return len(self.tokenizer.encode(text))

    def _format_event_for_prompt(self, event: Event) -> str:
        """Formats a single event into a readable transcript line."""
        timestamp_str = event.created_at.strftime("%H:%M:%S")
        
        if event.event_type == EventType.DIALOGUE:
            return f"[{timestamp_str}] {event.actor_id} says: \"{event.content}\""
        elif event.event_type == EventType.ACTION:
            return f"[{timestamp_str}] {event.actor_id} acts: {event.content}"
        elif event.event_type == EventType.NARRATION:
            return f"[{timestamp_str}] ORCHESTRATOR NARRATES: {event.content}"
        elif event.event_type == EventType.SYSTEM:
            return f"[{timestamp_str}] SYSTEM: {event.content}"
        else:
            return f"[{timestamp_str}] {event.actor_id}: {event.content}"

    def build_actor_prompt(
        self, season_id: str, actor_id: str, day: int, slot: int, system_prompt: str
    ) -> List[Dict[str, str]]:
        """Assembles the full prompt array for an actor to take their next turn.
        
        Pulls all visible events from the ledger, formats them chronologically, 
        and packs them backwards from most-recent to oldest until the token limit is hit.
        """
        
        # 1. Fetch the raw canonical events this specific actor is allowed to see
        visible_events: List[Event] = self.db.get_visible_events_for_actor(
            season_id=season_id, actor_id=actor_id, up_to_day=day, up_to_slot=slot
        )

        # 2. Format the events into text strings
        formatted_lines = [self._format_event_for_prompt(e) for e in visible_events]
        
        # 3. Token-aware packing (work backwards from the most recent event)
        context_buffer = []
        current_tokens = 0
        
        # Reverse the list so we prioritize the most recent context first
        for line in reversed(formatted_lines):
            line_tokens = self._count_tokens(line + "\n")
            
            if current_tokens + line_tokens > self.max_context_tokens:
                # If adding this line exceeds the limit, stop packing
                break
                
            context_buffer.insert(0, line) # Insert at the front to restore chronological order
            current_tokens += line_tokens

        # 4. Assemble the final context block
        compiled_transcript = "\n".join(context_buffer)
        
        if not compiled_transcript:
            compiled_transcript = "[No previous events. The scene is beginning.]"
            
        # 5. Structure the final prompt payload expected by standard LLM APIs
        prompt_payload = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    "Below is the literal chronological transcript of the scene so far. "
                    "Read it carefully and provide your next action.\n\n"
                    f"--- SCENE TRANSCRIPT ---\n{compiled_transcript}\n--- END TRANSCRIPT ---\n\n"
                    "What do you do next?"
                )
            }
        ]
        
        return prompt_payload

    def build_diary_prompt(
        self, season_id: str, actor_id: str, day: int, slot: int, character_card: str
    ) -> List[Dict[str, str]]:
        """Assembles the prompt for the tweet-style stochastic diary generation."""
        
        # For a diary, the actor still needs the transcript to know what to write about
        visible_events: List[Event] = self.db.get_visible_events_for_actor(
            season_id=season_id, actor_id=actor_id, up_to_day=day, up_to_slot=slot
        )
        
        formatted_lines = [self._format_event_for_prompt(e) for e in visible_events]
        compiled_transcript = "\n".join(formatted_lines[-50:]) # Only pass the last ~50 events for a quick diary
        
        prompt_payload = [
            {
                "role": "system",
                "content": (
                    f"{character_card}\n\n"
                    "You are generating a short, in-character micro-blog post (tweet-style) "
                    "reflecting on recent events. It MUST be under 280 characters."
                )
            },
            {
                "role": "user",
                "content": (
                    "Review the recent events you just experienced:\n\n"
                    f"{compiled_transcript}\n\n"
                    "Write your diary entry now. Ensure it reflects your specific personality and perspective on these events."
                )
            }
        ]
        
        return prompt_payload
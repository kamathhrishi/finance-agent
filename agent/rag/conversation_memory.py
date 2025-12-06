#!/usr/bin/env python3
"""
Conversation Memory Module for RAG System

This module handles conversation memory functionality for maintaining context
across multiple interactions in the RAG system.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncpg
from contextvars import ContextVar

# Configure logging
logger = logging.getLogger(__name__)

# Context variable for request-scoped database connection (thread-safe for async)
_db_connection_context: ContextVar[Optional[asyncpg.Connection]] = ContextVar('db_connection', default=None)


# ═════════════════════════════════════════════════════════════════════
# STAGE 1: INITIALIZATION & SETUP
# ═════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """Enhanced conversation memory for question analysis context with balanced user/AI messages."""
    
    def __init__(self, max_messages: int = 4, max_chars_per_message: int = 100):
        self.max_messages = max_messages  # Total messages to keep (4 = 2 user + 2 AI)
        self.max_chars_per_message = max_chars_per_message  # 100 chars max per message
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}  # conversation_id -> messages
    
    def set_database_connection(self, db_connection):
        """Set the database connection for retrieving conversation history (request-scoped)."""
        _db_connection_context.set(db_connection)
        logger.debug(f"Set database connection in context for async task")
    
    @property
    def db_connection(self) -> Optional[asyncpg.Connection]:
        """Get the database connection for the current async context."""
        return _db_connection_context.get()

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 2: MESSAGE MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════

    def add_message(self, conversation_id: str, message: str, role: str = "user"):
        """Add a message to conversation history with intelligent truncation."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            logger.info(f"💬 Created new in-memory conversation for {conversation_id}")
        
        # Smart truncation - preserve important parts
        truncated_message = self._smart_truncate(message, self.max_chars_per_message)
        logger.info(f"💬 Adding {role} message to conversation {conversation_id}: '{truncated_message[:50]}...'")
        
        # Store original length for context
        message_data = {
            "role": role,
            "content": truncated_message,
            "original_length": len(message)
        }
        
        self.conversations[conversation_id].append(message_data)
        
        # Keep only the most recent messages
        if len(self.conversations[conversation_id]) > self.max_messages:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_messages:]
    
    def _smart_truncate(self, message: str, max_chars: int) -> str:
        """Smart truncation that tries to preserve sentence boundaries."""
        if len(message) <= max_chars:
            return message
        
        # Try to find a good cut point near the end
        truncated = message[:max_chars-3]  # Leave room for "..."
        
        # Look for sentence endings in the last 50 characters
        best_cut = 0
        for i in range(len(truncated) - 50, len(truncated)):
            if truncated[i] in '.!?':
                best_cut = i + 1
                break
        
        # Look for word boundaries in the last 20 characters
        if best_cut == 0:
            for i in range(len(truncated) - 20, len(truncated)):
                if truncated[i] == ' ':
                    best_cut = i
                    break
        
        if best_cut > 0:
            return message[:best_cut] + "..."
        else:
            return message[:max_chars-3] + "..."
    
    def get_recent_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get recent messages for a conversation thread."""
        return self.conversations.get(conversation_id, [])

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 3: CONTEXT FORMATTING & DATABASE RETRIEVAL
    # ═════════════════════════════════════════════════════════════════════

    async def format_context(self, conversation_id: str) -> str:
        """Format recent messages as context string with clear role indicators."""
        # Try to get conversation context from database first
        if self.db_connection:
            try:
                db_context = await self._get_database_conversation_context(conversation_id)
                if db_context:
                    logger.info(f"💬 Retrieved conversation context from DATABASE for {conversation_id}")
                    return db_context
            except Exception as e:
                logger.warning(f"⚠️ Failed to get database conversation context: {e}")
                # Fall back to in-memory context
        
        # Fall back to in-memory context
        messages = self.get_recent_messages(conversation_id)
        if not messages:
            logger.info(f"💬 No conversation context found for {conversation_id} (fresh conversation)")
            return ""
        
        logger.info(f"💬 Retrieved {len(messages)} messages from IN-MEMORY for {conversation_id}")
        
        context_parts = []
        for i, msg in enumerate(messages):
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            role_label = "User" if msg["role"] == "user" else "Assistant"
            
            # Add truncation indicator if message was truncated
            truncation_note = f" (truncated from {msg['original_length']} chars)" if msg.get('original_length', 0) > len(msg['content']) else ""
            
            context_parts.append(f"{role_emoji} {role_label}: {msg['content']}{truncation_note}")
        
        return f"Recent conversation context ({len(messages)} messages):\n" + "\n".join(context_parts)

    async def _get_database_conversation_context(self, conversation_id: str) -> str:
        """Retrieve conversation context from database for the last 8 messages (4 exchanges) from this specific conversation thread."""
        try:
            # Import uuid module
            import uuid
            
            # Convert conversation_id to UUID for database query
            try:
                conv_uuid = uuid.UUID(conversation_id)
            except (ValueError, AttributeError):
                logger.warning(f"⚠️ Invalid conversation_id format: {conversation_id}, cannot retrieve conversation context")
                return ""
            
            # Get the last 8 messages from chat_messages table for THIS conversation thread
            logger.info(f"📚 Fetching conversation context from database for conversation: {conversation_id}")
            messages = await self.db_connection.fetch('''
                SELECT role, content, created_at
                FROM chat_messages 
                WHERE conversation_id = $1
                ORDER BY created_at DESC 
                LIMIT 8
            ''', conv_uuid)
            
            if not messages:
                logger.info(f"📚 No conversation history found in database for conversation: {conversation_id}")
                return ""
            
            logger.info(f"📚 Found {len(messages)} messages in database for conversation: {conversation_id}")
            
            # Format messages as context (reverse order to show chronological)
            context_parts = []
            for i, msg in enumerate(reversed(messages)):
                role_emoji = "👤" if msg['role'] == "user" else "🤖"
                role_label = "User" if msg['role'] == "user" else "Assistant"
                
                # Truncate to 100 chars
                content = msg['content'][:100]
                if len(msg['content']) > 100:
                    content += "..."
                
                context_parts.append(f"{role_emoji} {role_label}: {content}")
                
                # Log the first message for debugging
                if i == 0:
                    logger.info(f"📚 First message - {role_label}: {content[:50]}...")
            
            formatted_context = f"Recent conversation context ({len(messages)} messages):\n" + "\n".join(context_parts)
            logger.info(f"📚 Formatted conversation context length: {len(formatted_context)} chars")
            return formatted_context
            
        except Exception as e:
            logger.error(f"❌ Error retrieving database conversation context: {e}")
            return ""

import { useState, useCallback, useEffect } from 'react'
import { useAuthStatus } from './useAuthStatus'
import {
  streamChat,
  generateMessageId,
  fetchConversations,
  fetchConversation,
  type ChatMessage,
  type ReasoningStep,
  type Source,
  type SSEEvent,
  type Conversation,
} from '../lib/api'
import { _scopeStoreInternal } from '../lib/scopeStore'
import { getStoredModel } from '../lib/models'
import { track } from '../lib/analytics'

interface UseChatReturn {
  messages: ChatMessage[]
  isLoading: boolean
  error: string | null
  sendMessage: (content: string) => Promise<void>
  clearMessages: () => void
  currentReasoning: ReasoningStep[]
  // Conversation management
  conversations: Conversation[]
  currentConversationId: string | null
  loadConversation: (conversationId: string) => Promise<void>
  startNewConversation: () => void
  refreshConversations: () => Promise<void>
}

export function useChat(): UseChatReturn {
  const { authEnabled, isSignedIn, getOptionalToken } = useAuthStatus()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentReasoning, setCurrentReasoning] = useState<ReasoningStep[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)

  const refreshConversations = useCallback(async () => {
    if (!authEnabled || !isSignedIn) return
    try {
      const token = await getOptionalToken()
      if (token) {
        const convs = await fetchConversations(token)
        setConversations(convs)
      }
    } catch (err) {
      console.error('Failed to fetch conversations:', err)
    }
  }, [authEnabled, isSignedIn, getOptionalToken])

  // Fetch conversations on mount if signed in
  useEffect(() => {
    if (authEnabled && isSignedIn) {
      refreshConversations()
    }
  }, [authEnabled, isSignedIn, refreshConversations])

  const loadConversation = useCallback(async (conversationId: string) => {
    if (!authEnabled || !isSignedIn) return
    setIsLoading(true)
    setError(null)
    setMessages([])  // Clear immediately for clean transition
    try {
      const token = await getOptionalToken()
      if (token) {
        const conversation = await fetchConversation(conversationId, token)
        setCurrentConversationId(conversationId)
        // Convert backend messages to frontend format
        const messages = conversation.messages || []
        const loadedMessages: ChatMessage[] = messages.map((msg) => {
          return {
            id: msg.id,
            role: msg.role as 'user' | 'assistant',
            content: msg.content || '',
            sources: msg.citations || [],
            reasoning: msg.reasoning || [],
            timestamp: new Date(msg.created_at),
            isStreaming: false,
          }
        })
        setMessages(loadedMessages)
        setCurrentReasoning([]) // Clear global reasoning when loading saved conversation
      }
    } catch (err) {
      console.error('Failed to load conversation:', err)
      setError(err instanceof Error ? err.message : 'Failed to load conversation')
    } finally {
      setIsLoading(false)
    }
  }, [authEnabled, isSignedIn, getOptionalToken])

  const startNewConversation = useCallback(() => {
    setCurrentConversationId(null)
    setMessages([])
    setError(null)
    setCurrentReasoning([])
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    setError(null)
    setIsLoading(true)
    setCurrentReasoning([])

    // Get auth token if signed in
    let authToken: string | null = null
    authToken = await getOptionalToken()

    // Add user message
    const userMessage: ChatMessage = {
      id: generateMessageId(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }

    // Add placeholder assistant message
    const assistantMessageId = generateMessageId()
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      reasoning: [],
      sources: [],
      timestamp: new Date(),
      isStreaming: true,
    }

    setMessages((prev) => [...prev, userMessage, assistantMessage])

    // Hoisted out of the try block so the catch can reference them too.
    const _model = getStoredModel()
    const _sentAt = performance.now()

    try {
      let accumulatedContent = ''
      const accumulatedReasoning: ReasoningStep[] = []
      let accumulatedSources: Source[] = []
      let newConversationId = currentConversationId

      // Snapshot the current user's scope at send time. Read directly from the
      // singleton so the hook doesn't have to subscribe (avoids re-renders on
      // every chip add/remove).
      const scopedFilings = _scopeStoreInternal.getSnapshot()

      track({
        name: 'chat_message_sent',
        props: {
          model: _model,
          chars: content.trim().length,
          pinned_count: scopedFilings.length,
          conversation_existing: Boolean(currentConversationId),
        },
      })

      for await (const event of streamChat(content, {
        conversationId: currentConversationId || undefined,
        authToken,
        scopedFilings: scopedFilings.length > 0 ? scopedFilings : undefined,
        model: getStoredModel(),
      })) {
        // Capture conversation_id from first response
        if (event.conversation_id && !newConversationId) {
          newConversationId = event.conversation_id
          setCurrentConversationId(newConversationId)
        }

        handleSSEEvent(
          event,
          assistantMessageId,
          accumulatedContent,
          accumulatedReasoning,
          accumulatedSources,
          (newContent) => {
            accumulatedContent = newContent
          },
          (newSources) => {
            accumulatedSources = newSources
          }
        )
      }

      // Mark streaming as complete
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, isStreaming: false }
            : msg
        )
      )

      track({
        name: 'chat_response_received',
        props: {
          model: _model,
          latency_ms: Math.round(performance.now() - _sentAt),
          citation_count: accumulatedSources.length,
          conversation_id: newConversationId || undefined,
        },
      })

      // Refresh conversations list after sending a message
      if (authEnabled && isSignedIn) {
        refreshConversations()
      }
    } catch (err) {
      console.error('Chat error:', err)
      const errorMessage = err instanceof Error ? err.message : 'An error occurred'
      setError(errorMessage)
      track({
        name: 'chat_response_error',
        props: { model: _model, reason: errorMessage.slice(0, 120) },
      })

      // Update assistant message with error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: 'Sorry, an error occurred while processing your request. Please try again.',
                isStreaming: false,
              }
            : msg
        )
      )
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, authEnabled, isSignedIn, getOptionalToken, currentConversationId, refreshConversations])

  const handleSSEEvent = useCallback(
    (
      event: SSEEvent,
      messageId: string,
      currentContent: string,
      currentReasoningSteps: ReasoningStep[],
      _currentSources: Source[],
      setContent: (content: string) => void,
      setSources: (sources: Source[]) => void
    ) => {
      switch (event.type) {
        case 'token':
          // Backend sends token in 'content' field, not 'token'
          const tokenContent = event.content || event.token
          if (tokenContent) {
            const newContent = currentContent + tokenContent
            setContent(newContent)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId ? { ...msg, content: newContent } : msg
              )
            )
          }
          break

        // All reasoning/progress event types
        case 'reasoning':
        case 'progress':
        case 'analysis':
        case 'search':
        case 'news_search':
        case '10k_search':
        case 'iteration_start':
        case 'iteration_search':
        case 'iteration_transcript_search':
        case 'iteration_news_search':
        case 'iteration_followup':
        case 'iteration_complete':
        case 'iteration_final':
        case 'agent_decision':
        case 'planning_start':
        case 'planning_complete':
        case 'retrieval_complete':
        case 'evaluation_complete':
        case 'search_complete':
        case '10k_planning':
        case '10k_retrieval':
        case '10k_evaluation':
        case 'api_retry':
          if (event.message) {
            const newStep: ReasoningStep = {
              message: event.message,
              step: event.step || event.type,
              data: event.data,
            }
            currentReasoningSteps.push(newStep)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId
                  ? { ...msg, reasoning: [...currentReasoningSteps] }
                  : msg
              )
            )
          }
          break

        case 'result':
        case '10k_answer': {
          // Handle answer - could be in various nested locations
          const data = event.data as Record<string, unknown> | undefined
          const response = data?.response as Record<string, unknown> | undefined

          // Try multiple paths: event.answer -> event.data.answer -> event.data.response.answer
          const answerContent = event.answer ||
            data?.answer as string ||
            response?.answer as string

          if (answerContent) {
            setContent(answerContent)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId ? { ...msg, content: answerContent } : msg
              )
            )
          }

          // Handle citations - could be in various nested locations
          // Backend sends citations with fields: company, ticker, quarter, chunk_text, chunk_id, etc.
          const citationsData = event.citations ||
            data?.citations ||
            response?.citations
          if (citationsData && Array.isArray(citationsData)) {
            // Pass through ALL fields from backend - cast to any to avoid type stripping
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const newSources: Source[] = citationsData.map((c: any) => ({
              // Spread first so any field the backend sends survives —
              // including ones we forget to enumerate. Without this, the
              // fs_research agent's source_backend / line_start / line_end /
              // filing_type were getting silently dropped, causing
              // handleViewSECFiling() to fall through to the legacy SQL/S3
              // path that 404'd → showed "Coming Soon" placeholder.
              ...c,

              // Explicit overrides where we want a specific shape:
              // - normalize type from either field name
              type: c.type || c.citation_type,
            }))
            setSources(newSources)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId ? { ...msg, sources: newSources } : msg
              )
            )
          }
          break
        }

        case 'error':
          setError(event.message || 'An error occurred')
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === messageId
                ? { ...msg, content: event.message || 'An error occurred', isStreaming: false }
                : msg
            )
          )
          break

        case 'done':
          // Stream complete - no action needed
          break

        default:
          // Handle any unknown event types that have a message
          if (event.message) {
            const newStep: ReasoningStep = {
              message: event.message,
              step: event.step || event.type,
              data: event.data,
            }
            currentReasoningSteps.push(newStep)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId
                  ? { ...msg, reasoning: [...currentReasoningSteps] }
                  : msg
              )
            )
          }
      }
    },
    []
  )

  const clearMessages = useCallback(() => {
    setMessages([])
    setError(null)
    setCurrentReasoning([])
    setCurrentConversationId(null)
  }, [])

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    currentReasoning,
    conversations,
    currentConversationId,
    loadConversation,
    startNewConversation,
    refreshConversations,
  }
}

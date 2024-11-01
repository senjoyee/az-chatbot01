'use client'

import { useState, useRef, useEffect } from 'react'
import { Send } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface Message {
  id: number
  text: string
  sender: 'user' | 'assistant'
}

export default function Component() {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, text: "Hello! How can I assist you today?", sender: 'assistant' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return
  
    const userMessage = { 
      id: messages.length + 1, 
      text: input.trim(), 
      sender: 'user' as const
    }
  
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setError(null)
  
    try {
      const response = await fetch('https://jscbbackend01.azurewebsites.net/conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.text,
          conversation: {
            conversation: messages.map(msg => ({
              role: msg.sender,
              content: msg.text,
              id: msg.id
            }))
          }
        }),
      })
  
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error: ${response.status} - ${errorText}`);
      }
  
      const data = await response.json()
  
      if (!data.answer) {
        throw new Error('Invalid response format')
      }
  
      const botResponse = { 
        id: messages.length + 2, 
        text: data.answer,
        sender: 'assistant' as const
      }
      
      setMessages(prev => [...prev, botResponse])
    } catch (error) {
      console.error('Chat error:', error)
      setError(error instanceof Error ? error.message : 'Failed to send message')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200 p-4">
      <div className="relative w-full max-w-6xl rounded-lg bg-white shadow-2xl overflow-hidden">
        <div className="flex flex-col h-[80vh] relative z-10">
          <div className="bg-primary text-primary-foreground p-6">
            <h2 className="text-2xl font-semibold font-mono">Document Assistant</h2>
          </div>

          {error && (
            <Alert variant="destructive" className="m-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <ScrollArea className="flex-grow px-8 py-6" ref={scrollAreaRef}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`mb-4 flex ${
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.sender === 'assistant' && (
                  <Avatar className="h-8 w-8 mr-2">
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                )}
                <div
                  className={`inline-block p-4 rounded-lg max-w-[85%] ${
                    message.sender === 'user'
                      ? 'bg-primary/80 text-primary-foreground rounded-br-none'
                      : 'bg-gray-100 text-gray-800 rounded-bl-none'
                  }`}
                >
                  <pre 
                    className="font-mono text-sm whitespace-pre-wrap break-words"
                    style={{ 
                      whiteSpace: 'pre-wrap',
                      wordWrap: 'break-word'
                    }}
                  >
                    {message.text}
                  </pre>
                </div>
                {message.sender === 'user' && (
                  <Avatar className="h-8 w-8 ml-2">
                    <AvatarFallback>U</AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex items-center space-x-2 text-gray-500 font-mono">
                <div className="animate-pulse">Thinking...</div>
              </div>
            )}
          </ScrollArea>

          <div className="border-t p-6 bg-gray-50">
            <div className="flex items-center space-x-4">
              <Input
                type="text"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                disabled={loading}
                className="flex-grow text-base rounded-full font-mono"
              />
              <Button 
                onClick={handleSend} 
                size="icon" 
                className="rounded-full"
                disabled={loading || !input.trim()}
              >
                <Send className="h-5 w-5" />
                <span className="sr-only">Send</span>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
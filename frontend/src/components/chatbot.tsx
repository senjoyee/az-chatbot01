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

// Helper functions
const isContactResponse = (text: string): boolean => {
  return text.toLowerCase().includes('contact') && 
         (text.toLowerCase().includes('email') || text.toLowerCase().includes('phone'));
};

const formatContactResponse = (text: string): string => {
  // Get introduction text
  const [intro, ...rest] = text.split(/(?=\d+\.)/);
  let result = intro.trim() + '\n\n';
  
  // Process each contact block
  let currentPerson = '';
  let details: string[] = [];
  
  rest.forEach(block => {
    const lines = block.split('\n').map(line => line.trim()).filter(Boolean);
    
    lines.forEach(line => {
      if (line.match(/^\d+\./)) {
        // If we have a previous person, add them to result
        if (currentPerson) {
          result += `${currentPerson}\n${details.join('\n')}\n\n`;
          details = [];
        }
        currentPerson = line;
      } else if (line.match(/^[a-z]\.|^-/)) {
        details.push(line.replace(/^[a-z]\./, '-'));
      } else if (!line.match(/^[A-Z]/)) {
        details.push(`- ${line}`);
      }
    });
  });
  
  // Add last person
  if (currentPerson) {
    result += `${currentPerson}\n${details.join('\n')}`;
  }
  
  return result.trim();
};

const formatGeneralResponse = (text: string): string => {
  return text
    .replace(/\*\*/g, '')
    .replace(/(.*?)(?=\d+\.|$)/s, '$1\n\n')
    .replace(/([a-z])\.\s+/g, '- ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

const formatBotResponse = (text: string): string => {
  // Remove any markdown formatting
  text = text.replace(/\*\*/g, '');
  
  // Choose formatter based on content
  return isContactResponse(text) 
    ? formatContactResponse(text)
    : formatGeneralResponse(text);
};

export default function Chatbot() {
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
      sender: 'user' 
    }
  
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setError(null)
  
    try {
      // Format conversation history to match backend expectations
      const conversationHistory = messages.map(msg => ({
        role: msg.sender,  // Map 'bot' to 'assistant'
        content: msg.text,
        id: msg.id
      }));
  
      const response = await fetch('https://jscbbackend01.azurewebsites.net/conversation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.text,
          conversation: {
            conversation: messages.map(msg => ({
              role: msg.sender,  // Keep as 'user' or 'assistant'
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
  
      // Update the botResponse creation in handleSend:
      const botResponse = { 
        id: messages.length + 2, 
        text: formatBotResponse(data.answer), 
        sender: 'assistant' 
      }
      
      setMessages(prev => [...prev, botResponse])
    } catch (error) {
      console.error('Chat error:', error)
      setError(error instanceof Error ? error.message : 'Failed to send message')
      // Optionally, remove the user message if the request failed
      // setMessages(prev => prev.slice(0, -1))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200 p-4">
      <div className="relative w-full max-w-6xl rounded-lg bg-white shadow-2xl overflow-hidden"> {/* Increased from max-w-4xl */}
        {/* Background Logo */}
        <div className="absolute inset-0 flex items-center justify-center opacity-5">
          <img 
            src="/SoftwareOne_Logo_Lrg_RGB_Blk.svg" 
            alt="SoftwareOne Logo" 
            className="max-w-full max-h-full" 
          />
        </div>

        {/* Chat Interface */}
        <div className="flex flex-col h-[80vh] relative z-10">
          {/* Chat Header */}
          <div className="bg-primary text-primary-foreground p-6">
            <h2 className="text-2xl font-semibold">Document Assistant</h2>
          </div>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive" className="m-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Messages Area */}
          <ScrollArea className="flex-grow px-8 py-6" ref={scrollAreaRef}> {/* Increased from p-6 */}
            {messages.map((message) => (
              <div
                key={message.id}
                className={`mb-4 flex ${
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}
              >
                {message.sender === 'bot' && (
                  <Avatar className="h-8 w-8 mr-2">
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                )}
                <div
                  className={`inline-block p-4 rounded-lg max-w-[85%] ${
                    message.sender === 'user'
                      ? 'bg-primary/80 text-primary-foreground rounded-br-none' // Added /80 for 80% opacity
                      : 'bg-gray-100 text-gray-800 rounded-bl-none whitespace-pre-line font-normal leading-[1.2]'
                  }`}
                >
                  {message.text}
                </div>
                {message.sender === 'user' && (
                  <Avatar className="h-8 w-8 ml-2">
                    <AvatarFallback>U</AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex items-center space-x-2 text-gray-500">
                <div className="animate-pulse">Thinking...</div>
              </div>
            )}
          </ScrollArea>

          {/* Input Area */}
          <div className="border-t p-6 bg-gray-50">
            <div className="flex items-center space-x-4">
              <Input
                type="text"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                disabled={loading}
                className="flex-grow text-base rounded-full"
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
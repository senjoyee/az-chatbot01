'use client'

import { useState, useRef, useEffect } from 'react'
import { Send } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

interface Message {
  id: number
  text: string
  sender: 'user' | 'bot'
}

export default function Component() {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, text: "Hello! How can I assist you today?", sender: 'bot' }
  ])
  const [input, setInput] = useState('')
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }, [messages])

  const handleSend = async () => {
    if (input.trim()) {
      const newMessage = { id: messages.length + 1, text: input, sender: 'user' };
      setMessages([...messages, newMessage]);
      setInput('');
    
      try {
        // Update the fetch URL to use the proxy
        const response = await fetch('https://jscb-proxy-nginx.azurewebsites.net/api/conversation', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: input,
            conversation: messages,
          }),
        });
        const data = await response.json();
        const botResponse = { id: messages.length + 2, text: data.answer, sender: 'bot' };
        setMessages(prevMessages => [...prevMessages, botResponse]);
      } catch (error) {
        console.error('Error:', error);
      }
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200 p-4">
      <div className="relative w-full max-w-4xl rounded-lg bg-white shadow-2xl overflow-hidden">
        {/* Background Logo */}
        <div className="absolute inset-0 flex items-center justify-center opacity-5">
          <img src="/SoftwareOne_Logo_Lrg_RGB_Blk.svg" alt="SoftwareOne Logo" className="max-w-full max-h-full" />
        </div>
        
        {/* Chat Interface */}
        <div className="flex flex-col h-[80vh] relative z-10">
          {/* Chat Header */}
          <div className="bg-primary text-primary-foreground p-6">
            <h2 className="text-2xl font-semibold">Document Assistant</h2>
          </div>
          
          {/* Messages Area */}
          <ScrollArea className="flex-grow p-6" ref={scrollAreaRef}>
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
                  className={`inline-block p-4 rounded-lg max-w-[70%] ${
                    message.sender === 'user'
                      ? 'bg-primary text-primary-foreground rounded-br-none'
                      : 'bg-gray-100 text-gray-800 rounded-bl-none'
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
                className="flex-grow text-base rounded-full"
              />
              <Button onClick={handleSend} size="icon" className="rounded-full">
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
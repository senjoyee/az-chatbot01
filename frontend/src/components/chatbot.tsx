'use client';

import { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Alert, AlertDescription } from "@/components/ui/alert";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { fetchEventSource } from '@microsoft/fetch-event-source';

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'assistant';
  isStreaming?: boolean;
  error?: boolean;
}

const API_BASE_URL = 'https://jscbbackend01.azurewebsites.net';

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, text: "Hello! How can I assist you today?", sender: 'assistant' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const askQuestionStream = async (
    message: string,
    conversation: Message[],
    onMessage: (data: string) => void,
    onError: (err: any) => void
  ) => {
    await fetchEventSource(`${API_BASE_URL}/conversation/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question: message,
        conversation: {
          conversation: conversation.map(msg => ({
            role: msg.sender,
            content: msg.text,
            id: msg.id,
          }))
        }
      }),
      onmessage(event) {
        if (event.data) {
          onMessage(event.data);
        }
      },
      onerror(err) {
        onError(err);
      },
      withCredentials: true,
    });
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);

    const userMessage: Message = {
      id: messages.length + 1,
      text: input.trim(),
      sender: 'user'
    };

    const assistantMessage: Message = {
      id: messages.length + 2,
      text: "",
      sender: 'assistant',
      isStreaming: true,
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);

    try {
      await askQuestionStream(
        input.trim(),
        [...messages, userMessage],
        (data: string) => {
          setMessages(prev => {
            const updated = [...prev];
            const lastMessage = updated[updated.length - 1];
            updated[updated.length - 1] = {
              ...lastMessage,
              text: lastMessage.text + data
            };
            return updated;
          });
        },
        (err: any) => {
          setMessages(prev => {
            const updated = [...prev];
            const lastMessage = updated[updated.length - 1];
            updated[updated.length - 1] = {
              ...lastMessage,
              isStreaming: false,
              error: true
            };
            return updated;
          });
          setError('An error occurred while streaming the response.');
        }
      );
    } catch (err: any) {
      setMessages(prev => {
        const updated = [...prev];
        const lastMessage = updated[updated.length - 1];
        updated[updated.length - 1] = {
          ...lastMessage,
          text: 'Error: ' + (err as Error).message,
          isStreaming: false,
          error: true
        };
        return updated;
      });
    } finally {
      setLoading(false);
      setInput('');
    }
  };

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

          <ScrollArea className="flex-grow px-6 py-4" ref={scrollAreaRef}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`mb-2 flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.sender === 'assistant' && (
                  <Avatar className="h-8 w-8 mr-2">
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
                )}
                <div
                  className={`inline-block p-3 rounded-lg max-w-[85%] ${
                    message.sender === 'user'
                      ? 'bg-primary/80 text-primary-foreground rounded-br-none'
                      : 'bg-gray-100 text-gray-800 rounded-bl-none'
                  }`}
                >
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkBreaks]}
                    className="font-mono text-sm whitespace-pre-wrap break-words [&>*]:leading-tight [&>p]:my-0.5 last:[&>p]:mb-0 first:[&>p]:mt-0"
                    components={{
                      code: ({ node, inline, className, children, ...props }) => (
                        <code
                          className={`${
                            inline
                              ? 'bg-gray-200 px-1 rounded'
                              : 'block bg-gray-800 text-white p-2 rounded my-1'
                          } ${className}`}
                          {...props}
                        >
                          {children}
                        </code>
                      ),
                      p: ({ children }) => (
                        <p className="whitespace-pre-wrap break-words leading-5">
                          {children}
                        </p>
                      ),
                      h1: ({ children }) => (
                        <h1 className="text-base font-bold my-1">{children}</h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="text-base font-semibold my-1">{children}</h2>
                      ),                      
                    }}
                  >
                    {message.text}
                  </ReactMarkdown>
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
  );
}
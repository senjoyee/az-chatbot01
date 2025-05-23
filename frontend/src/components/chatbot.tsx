'use client'  

import { useState, useRef, useEffect } from 'react'  
import { Send, FileQuestion, Upload, ChevronLeft, ChevronRight, Copy } from 'lucide-react'  
import { Button } from "@/components/ui/button"  
import { Input } from "@/components/ui/input"  
import { ScrollArea } from "@/components/ui/scroll-area"  
import { Avatar, AvatarFallback } from "@/components/ui/avatar"  
import { Alert, AlertDescription } from "@/components/ui/alert"  
import { FileSidebar } from "@/components/ui/file-sidebar"
import { toast } from 'sonner'
import Link from 'next/link'
import ReactMarkdown from 'react-markdown'  
import remarkGfm from 'remark-gfm'
import { sendMessage } from "./api";  

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
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [sidebarVisible, setSidebarVisible] = useState(true)
  const scrollAreaRef = useRef<HTMLDivElement>(null)  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const lastUserMessageRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {  
    // Scroll to position the last user message at the top with the AI response below
    if (messages.length > 1 && lastUserMessageRef.current) {
      // If the last message is from the assistant and there's a preceding user message
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.sender === 'assistant') {
        // Scroll to position the user's question at the top
        lastUserMessageRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, [messages])  

  const handleSend = async () => {  
    if (!input.trim() || loading || selectedFiles.length === 0) return  
    
    const userMessage = {  
      id: messages.length + 1,  
      text: input,  
      sender: 'user' as const  
    }  
    
    setMessages(prev => [...prev, userMessage])  
    setInput('')  
    setError(null)  
    setLoading(true)  
    
    try {  
      // Use the sendMessage API function instead of direct fetch
      const data = await sendMessage(userMessage.text, messages, selectedFiles);

      if (!data || !data.answer) {  
        console.error("Invalid response format:", data);
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

  const handleFileSelectionChange = (files: string[]) => {
    setSelectedFiles(files)
    // You can add additional logic here if needed when files are selected
    console.log('Selected files:', files)
  }

  return (  
    <div className="flex h-screen bg-gray-50 p-4 gap-4 overflow-hidden font-sans">  
      {/* File Sidebar */}
      <div className={`${sidebarVisible ? 'w-1/3' : 'w-0'} bg-[#2F3336] rounded-xl overflow-hidden shadow-lg flex flex-col transition-all duration-300 ease-in-out`}>
        <div className="bg-[#2F3336] p-4 text-white font-medium">
          Sources
        </div>
        <div className="flex-1 overflow-hidden bg-white">
          <FileSidebar onFileSelectionChange={handleFileSelectionChange} />
        </div>
      </div>
      
      {/* Sidebar Toggle Button */}
      <div className="flex items-center">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={() => setSidebarVisible(!sidebarVisible)}
          className="h-8 w-8 rounded-full bg-gray-700 text-white hover:bg-gray-600"
        >
          {sidebarVisible ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </Button>
      </div>
      
      {/* Chat Area */}
      <div className="flex-1 flex flex-col bg-[#2F3336] rounded-xl overflow-hidden shadow-lg">
        <div className="flex flex-col h-full">  
          <div className="bg-[#2F3336] p-4 flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-white font-medium">Document Assistant</h2>
              <Link href="/uploadservice" className="flex items-center px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors">
                <Upload className="h-4 w-4 mr-1.5" />
                Upload Documents
              </Link>
            </div>
            <div className="text-gray-200">  
              {selectedFiles.length > 0 ? (
                <p className="text-sm text-gray-200">
                  {selectedFiles.length} file(s) selected for context
                </p>
              ) : (
                <p className="text-sm text-gray-200">
                  No files selected. Please select at least one file to start chatting.
                </p>
              )}
            </div>  
          </div>

          {error && (  
            <Alert variant="destructive" className="m-4">  
              <AlertDescription>{error}</AlertDescription>  
            </Alert>  
          )}  

          <ScrollArea className="flex-grow px-6 py-4 bg-white" ref={scrollAreaRef}>  
            {selectedFiles.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center p-6">
                <FileQuestion className="h-16 w-16 text-gray-300 mb-4" />
                <h3 className="text-lg font-medium text-gray-700 mb-2">Select files to start chatting</h3>
                <p className="text-gray-500 max-w-md">
                  Please select one or more files from the sidebar to provide context for your questions.
                  If you're unsure which files to select, you can select all files.
                </p>
              </div>
            ) : (
              <>
                {messages.map((message) => (  
                  <div  
                    key={message.id}  
                    ref={message.sender === 'user' ? (el) => { lastUserMessageRef.current = el } : undefined}
                    className={`mb-6 flex ${  
                      message.sender === 'user' ? 'justify-end' : 'justify-start'  
                    }`}  
                  >  
                    {message.sender === 'assistant' && (  
                      <Avatar className="h-8 w-8 mr-2 mt-1">  
                        <AvatarFallback>AI</AvatarFallback>  
                      </Avatar>  
                    )}  
                    <div  
                      className={`inline-block p-4 rounded-lg max-w-[85%] ${  
                        message.sender === 'user'  
                          ? 'bg-blue-600 text-white rounded-br-none'  
                          : 'bg-gray-100 text-gray-800 rounded-bl-none'  
                      } ${message.sender === 'assistant' ? 'group' : ''}`}  
                    >  
                      <div className="relative">
                        {message.sender === 'assistant' && (
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="absolute top-0 right-0 h-6 w-6 opacity-0 group-hover:opacity-100 hover:opacity-100 transition-opacity"
                            onClick={() => {
                              navigator.clipboard.writeText(message.text);
                              toast.success("Copied to clipboard");
                            }}
                            title="Copy to clipboard"
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                        )}
                        <ReactMarkdown   
                          className="text-sm font-medium"
                          remarkPlugins={[remarkGfm]}
                          components={{  
                            code: ({node, inline, className, children, ...props}) => (  
                              <code  
                                className={`${
                                  inline   
                                    ? 'bg-gray-200 px-1 rounded'   
                                    : 'block bg-gray-800 text-white p-2 rounded font-mono'  
                                } ${className}`}  
                                {...props}  
                              >  
                                {children}  
                              </code>  
                            ),  
                            p: ({children}) => (  
                              <p className="mb-2 whitespace-pre-line">
                                {children}
                              </p>  
                            ),  
                            h1: ({children}) => (  
                              <h1 className="text-lg font-semibold mb-2">{children}</h1>  
                            ),  
                            h2: ({children}) => (  
                              <h2 className="text-base font-medium mb-2">{children}</h2>  
                            ),
                            ul: ({children}) => (
                              <ul className="list-disc pl-6 mb-2 space-y-1">{children}</ul>
                            ),
                            ol: ({children}) => (
                              <ol className="list-decimal pl-6 mb-2 space-y-1">{children}</ol>
                            ),
                            li: ({children}) => (
                              <li className="mb-1">{children}</li>
                            ),
                            br: () => <br />
                          }}  
                        >  
                          {message.text}  
                        </ReactMarkdown>  
                      </div>
                    </div>  
                    {message.sender === 'user' && (  
                      <Avatar className="h-8 w-8 ml-2 mt-1">  
                        <AvatarFallback>U</AvatarFallback>  
                      </Avatar>  
                    )}  
                  </div>  
                ))}
                {/* This empty div is used as a target for auto-scrolling */}
                <div ref={messagesEndRef} />
                {loading && (  
                  <div className="flex items-center space-x-2 text-gray-500 mt-2">  
                    <div className="animate-pulse">Thinking...</div>  
                  </div>  
                )}  
              </>
            )}
          </ScrollArea>  

          <div className="border-t p-4 bg-[#2F3336] rounded-b-xl">  
            <div className="flex items-center space-x-4">  
              <Input  
                type="text"  
                placeholder={selectedFiles.length === 0 ? "Select files to enable chat..." : "Type your message..."}  
                value={input}  
                onChange={(e) => setInput(e.target.value)}  
                onKeyDown={(e) => {  
                  if (e.key === 'Enter' && !e.shiftKey && selectedFiles.length > 0) {  
                    e.preventDefault()  
                    handleSend()  
                  }  
                }}  
                className="flex-1 bg-white border-gray-300"  
                disabled={loading || selectedFiles.length === 0}  
              />  
              <Button 
                onClick={handleSend} 
                disabled={loading || !input.trim() || selectedFiles.length === 0}
                title={selectedFiles.length === 0 ? "Select at least one file to enable chat" : "Send message"}
                className="font-medium bg-blue-600 hover:bg-blue-700"
              >  
                <Send className="h-4 w-4 mr-2" />  
                Send  
              </Button>  
            </div>  
          </div>  
        </div>  
      </div>  
    </div>  
  )  
}
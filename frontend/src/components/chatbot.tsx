'use client'  

import { useState, useRef, useEffect } from 'react'  
import { Send, FileQuestion } from 'lucide-react'  
import { Button } from "@/components/ui/button"  
import { Input } from "@/components/ui/input"  
import { ScrollArea } from "@/components/ui/scroll-area"  
import { Avatar, AvatarFallback } from "@/components/ui/avatar"  
import { Alert, AlertDescription } from "@/components/ui/alert"  
import { FileSidebar } from "@/components/ui/file-sidebar"
import ReactMarkdown from 'react-markdown'  
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
  const scrollAreaRef = useRef<HTMLDivElement>(null)  

  useEffect(() => {  
    if (scrollAreaRef.current) {  
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight  
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
    <div className="flex h-screen bg-gradient-to-br from-gray-100 to-gray-200 overflow-hidden">  
      {/* File Sidebar */}
      <FileSidebar onFileSelectionChange={handleFileSelectionChange} />
      
      {/* Chat Area */}
      <div className="flex-1 flex items-center justify-center p-4 overflow-hidden">
        <div className="relative w-full h-full max-w-6xl rounded-lg bg-white shadow-2xl overflow-hidden">  
          <div className="flex flex-col h-full relative z-10">  
            <div className="bg-primary text-primary-foreground p-6">  
              <h2 className="text-2xl font-semibold font-mono">Document Assistant</h2>  
              {selectedFiles.length > 0 ? (
                <p className="text-sm mt-1 text-primary-foreground/80 font-mono">
                  {selectedFiles.length} file(s) selected for context
                </p>
              ) : (
                <p className="text-sm mt-1 text-primary-foreground/80 font-mono">
                  No files selected. Please select at least one file to start chatting.
                </p>
              )}
            </div>  

            {error && (  
              <Alert variant="destructive" className="m-4">  
                <AlertDescription>{error}</AlertDescription>  
              </Alert>  
            )}  

            <ScrollArea className="flex-grow px-6 py-4" ref={scrollAreaRef}>  
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
                      className={`mb-2 flex ${  
                        message.sender === 'user' ? 'justify-end' : 'justify-start'  
                      }`}  
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
                          className="font-mono text-sm whitespace-pre-wrap break-words [&>*]:leading-tight [&>p]:my-0.5 last:[&>p]:mb-0 first:[&>p]:mt-0"  
                          components={{  
                            code: ({node, inline, className, children, ...props}) => (  
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
                            p: ({children}) => (  
                              <p className="whitespace-pre-wrap break-words leading-5">  
                                {children}  
                              </p>  
                            ),  
                            h1: ({children}) => (  
                              <h1 className="text-base font-bold my-1">{children}</h1>  
                            ),  
                            h2: ({children}) => (  
                              <h2 className="text-base font-semibold my-1">{children}</h2>  
                            )  
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
                </>
              )}
            </ScrollArea>  

            <div className="border-t p-6 bg-gray-50">  
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
                  className="flex-1 font-mono"  
                  disabled={loading || selectedFiles.length === 0}  
                />  
                <Button 
                  onClick={handleSend} 
                  disabled={loading || !input.trim() || selectedFiles.length === 0}
                  title={selectedFiles.length === 0 ? "Select at least one file to enable chat" : "Send message"}
                >  
                  <Send className="h-4 w-4 mr-2" />  
                  Send  
                </Button>  
              </div>  
            </div>  
          </div>  
        </div>  
      </div>  
    </div>  
  )  
}
'use client'

import React, { useState, useRef, useEffect } from 'react'
import { ChevronLeft, ChevronRight, GripVertical } from 'lucide-react'
import { Button } from './button'

interface ResizableSidebarProps {
  children: React.ReactNode
  defaultWidth?: number
  minWidth?: number
  maxWidthPercent?: number
  onCollapsedChange?: (collapsed: boolean) => void
}

export function ResizableSidebar({
  children,
  defaultWidth = 280,
  minWidth = 200,
  maxWidthPercent = 33, // 1/3 of screen width
  onCollapsedChange
}: ResizableSidebarProps) {
  const [width, setWidth] = useState(defaultWidth)
  const [collapsed, setCollapsed] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const sidebarRef = useRef<HTMLDivElement>(null)
  const startXRef = useRef<number>(0)
  const startWidthRef = useRef<number>(defaultWidth)

  // Calculate max width based on screen width
  const getMaxWidth = () => {
    return window.innerWidth * (maxWidthPercent / 100)
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    setIsDragging(true)
    startXRef.current = e.clientX
    startWidthRef.current = width
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    
    // Add a class to the body to indicate dragging state
    document.body.classList.add('resizing')
  }

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return
    
    const deltaX = e.clientX - startXRef.current
    const newWidth = startWidthRef.current + deltaX
    const maxWidth = getMaxWidth()
    
    // Constrain width between min and max values
    if (newWidth >= minWidth && newWidth <= maxWidth) {
      setWidth(newWidth)
    } else if (newWidth < minWidth) {
      setWidth(minWidth)
    } else if (newWidth > maxWidth) {
      setWidth(maxWidth)
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
    document.removeEventListener('mousemove', handleMouseMove)
    document.removeEventListener('mouseup', handleMouseUp)
    
    // Remove the dragging class
    document.body.classList.remove('resizing')
  }

  const toggleCollapse = () => {
    const newCollapsedState = !collapsed
    setCollapsed(newCollapsedState)
    if (onCollapsedChange) {
      onCollapsedChange(newCollapsedState)
    }
  }

  // Clean up event listeners
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.classList.remove('resizing')
    }
  }, [])

  // Handle window resize to ensure sidebar doesn't exceed max width
  useEffect(() => {
    const handleResize = () => {
      const maxWidth = getMaxWidth()
      if (width > maxWidth) {
        setWidth(maxWidth)
      }
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [width])

  // Add a style element for the cursor
  useEffect(() => {
    const styleElement = document.createElement('style')
    styleElement.innerHTML = `
      body.resizing {
        cursor: ew-resize !important;
        user-select: none;
      }
      body.resizing * {
        cursor: ew-resize !important;
        user-select: none;
      }
    `
    document.head.appendChild(styleElement)
    
    return () => {
      document.head.removeChild(styleElement)
    }
  }, [])

  if (collapsed) {
    return (
      <div className="h-full flex flex-col border-r bg-gray-50">
        <Button 
          variant="ghost" 
          size="icon" 
          onClick={toggleCollapse}
          className="m-2"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <div 
      ref={sidebarRef}
      className={`h-full flex flex-col border-r bg-gray-50 relative ${isDragging ? 'select-none' : ''}`}
      style={{ width: `${width}px` }}
    >
      {children}
      
      {/* Resize handle */}
      <div 
        className="absolute right-0 top-0 bottom-0 w-6 cursor-ew-resize flex items-center justify-center z-10"
        onMouseDown={handleMouseDown}
        title="Drag to resize"
      >
        <div className={`h-full w-1 ${isDragging ? 'bg-primary' : 'bg-gray-300'} hover:bg-primary transition-colors`}></div>
        <GripVertical className="h-6 w-6 absolute opacity-50 text-gray-500 hover:text-primary transition-colors" />
      </div>
      
      {/* Collapse button - positioned in the top right corner */}
      <Button 
        variant="ghost" 
        size="icon" 
        onClick={toggleCollapse}
        className="absolute top-2 right-2 h-8 w-8 z-20"
        title="Collapse sidebar"
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
    </div>
  )
}

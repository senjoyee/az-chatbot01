'use client'

import React, { useState, useRef, useEffect } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
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
  const dragHandleRef = useRef<HTMLDivElement>(null)

  // Calculate max width based on screen width
  const getMaxWidth = () => {
    return window.innerWidth * (maxWidthPercent / 100)
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    setIsDragging(true)
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return
    
    const newWidth = e.clientX
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
      className="h-full flex flex-col border-r bg-gray-50 relative"
      style={{ width: `${width}px` }}
    >
      {children}
      
      {/* Resize handle */}
      <div 
        ref={dragHandleRef}
        className="absolute right-0 top-0 bottom-0 w-1 cursor-ew-resize hover:bg-primary/50 transition-colors"
        onMouseDown={handleMouseDown}
      />
      
      {/* Collapse button - positioned in the top right corner */}
      <Button 
        variant="ghost" 
        size="icon" 
        onClick={toggleCollapse}
        className="absolute top-2 right-2 h-8 w-8 z-10"
        title="Collapse sidebar"
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>
    </div>
  )
}

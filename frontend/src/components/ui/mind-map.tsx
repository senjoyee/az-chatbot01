'use client'

import React, { useEffect, useRef } from 'react'
import ReactECharts from 'echarts-for-react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './dialog'
import { Button } from './button'
import { Copy, Check } from 'lucide-react'
import { MindMapNode } from '../api'
import { copyToClipboard } from '../../utils/copy-utils'
import { toast } from 'sonner'

interface MindMapProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  data: MindMapNode | null
  filename: string
  loading: boolean
}

interface CopyButtonProps {
  text: string
}

function CopyButton({ text }: CopyButtonProps) {
  const [copied, setCopied] = React.useState(false)

  const handleCopy = async () => {
    try {
      await copyToClipboard(text, true) // true to remove markdown formatting
      setCopied(true)
      toast.success('Copied to clipboard')
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      toast.error('Failed to copy')
    }
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-8 w-8 p-0 text-gray-500 hover:text-gray-700"
      onClick={handleCopy}
      title="Copy to clipboard"
    >
      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
    </Button>
  )
}

export function MindMap({ open, onOpenChange, data, filename, loading }: MindMapProps) {
  // Convert the mind map data to ECharts format
  const getOption = () => {
    if (!data) return {}

    return {
      tooltip: {
        trigger: 'item',
        triggerOn: 'mousemove'
      },
      series: [
        {
          type: 'tree',
          data: [data],
          top: '10%',
          left: '5%',
          bottom: '10%',
          right: '20%',
          symbolSize: 12,
          initialTreeDepth: 3,
          label: {
            position: 'left',
            verticalAlign: 'middle',
            align: 'right',
            fontSize: 14,
            color: '#333'
          },
          leaves: {
            label: {
              position: 'right',
              verticalAlign: 'middle',
              align: 'left'
            }
          },
          emphasis: {
            focus: 'descendant'
          },
          expandAndCollapse: true,
          animationDuration: 550,
          animationDurationUpdate: 750,
          lineStyle: {
            width: 1.5,
            curveness: 0.5,
            color: '#aaa'
          }
        }
      ]
    }
  }

  // Get a flat string representation of the mind map for copying
  const getMindMapText = () => {
    if (!data) return ''
    
    const formatNode = (node: MindMapNode, level = 0): string => {
      const indent = '  '.repeat(level)
      let result = `${indent}${node.name}\n`
      
      if (node.children && node.children.length > 0) {
        node.children.forEach(child => {
          result += formatNode(child, level + 1)
        })
      }
      
      return result
    }
    
    return formatNode(data)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto font-noto-sans">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div className="font-noto-sans">
              <DialogTitle className="font-noto-sans">Document Mind Map</DialogTitle>
              <DialogDescription className="text-sm text-gray-500 font-noto-sans">
                {filename}
              </DialogDescription>
            </div>
            {!loading && data && (
              <CopyButton text={getMindMapText()} />
            )}
          </div>
        </DialogHeader>
        <div className="py-4">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-8 font-noto-sans">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-700 mb-4"></div>
              <p className="text-gray-600">Generating mind map...</p>
            </div>
          ) : data ? (
            <div className="w-full h-[60vh]">
              <ReactECharts
                option={getOption()}
                style={{ height: '100%', width: '100%' }}
                opts={{ renderer: 'canvas' }}
              />
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No mind map data available
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

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

    const option = {
      backgroundColor: '#fff',
      tooltip: {
        trigger: 'item',
        triggerOn: 'mousemove',
        formatter: '{b}',
        backgroundColor: 'rgba(50,50,50,0.9)',
        borderColor: '#777',
        textStyle: {
          color: '#fff'
        }
      },
      toolbox: {
        show: true,
        feature: {
          restore: { show: true, title: 'Reset' },
          saveAsImage: { show: true, title: 'Save Image' },
          dataZoom: { show: true, title: 'Zoom' }
        },
        right: 20,
        top: 20
      },
      series: [
        {
          type: 'tree',
          data: [data],
          top: 5,
          left: 5,
          bottom: 5,
          right: 5,
          symbolSize: 12,
          initialTreeDepth: 1, // Expand only first two levels (root=0, children=1)
          layout: 'orthogonal', // Use orthogonal layout for a cleaner look
          orient: 'LR', // Left to right orientation
          roam: true, // Enable panning and zooming
          nodeGap: 20, // Reduce gap between nodes
          layerGap: 40, // Reduce horizontal gap between layers
          zoom: 1, // Default zoom level
          center: ['50%', '50%'], // Center the chart
          label: {
            position: 'right',
            verticalAlign: 'middle',
            align: 'left',
            fontSize: 14,
            color: '#333',
            backgroundColor: 'rgba(245,245,245,0.8)',
            padding: [4, 8, 4, 8],
            borderRadius: 4
          },
          leaves: {
            label: {
              position: 'right',
              verticalAlign: 'middle',
              align: 'left'
            }
          },
          emphasis: {
            focus: 'descendant',
            itemStyle: {
              borderWidth: 2,
              borderColor: '#3771c8'
            },
            lineStyle: {
              width: 2,
              color: '#3771c8'
            },
            label: {
              fontWeight: 'bold',
              color: '#3771c8'
            }
          },
          expandAndCollapse: true,
          animationDuration: 550,
          animationDurationUpdate: 750,
          lineStyle: {
            width: 1.5,
            curveness: 0.3,
            color: '#aaa'
          },
          itemStyle: {
            color: '#6b9ac4',
            borderColor: '#4d7ca8'
          }
        }
      ]
    };
    console.log('MindMap option JSON:', JSON.stringify(option, null, 2));
    console.log('MindMap layerGap:', option.series[0]?.layerGap);
    console.log('MindMap nodeGap:', option.series[0]?.nodeGap);
    return option;
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
      <DialogContent className="fixed inset-0 w-screen h-screen max-w-none max-h-none translate-x-0 translate-y-0 left-0 top-0 overflow-hidden p-0 font-noto-sans rounded-none border-0">
        <DialogHeader className="p-4 border-b">
          <div className="flex items-center justify-between">
            <div className="font-noto-sans">
              <DialogTitle className="font-noto-sans text-xl">Document Mind Map</DialogTitle>
              <DialogDescription className="text-sm text-gray-500 font-noto-sans">
                {filename}
              </DialogDescription>
            </div>
            {!loading && data && (
              <CopyButton text={getMindMapText()} />
            )}
          </div>
        </DialogHeader>
        <div className="absolute inset-0 top-[60px] overflow-hidden">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-full font-noto-sans">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-700 mb-4"></div>
              <p className="text-gray-600 text-lg">Generating mind map...</p>
            </div>
          ) : data ? (
            <div className="w-full h-full">
              <ReactECharts
                option={getOption()}
                style={{ height: '100%', width: '100%' }}
                opts={{ renderer: 'canvas', devicePixelRatio: window.devicePixelRatio }}
                notMerge={true}
                lazyUpdate={true}
                className="w-full h-full"
                onEvents={{
                  'rendered': (chart: any) => {
                    // Force resize after rendering to ensure proper layout
                    setTimeout(() => {
                      window.dispatchEvent(new Event('resize'))
                      // Auto-fit content to view
                      if (chart && chart.resize) {
                        chart.resize()
                      }
                    }, 100)
                  }
                }}
              />
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500 text-lg">
              No mind map data available
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

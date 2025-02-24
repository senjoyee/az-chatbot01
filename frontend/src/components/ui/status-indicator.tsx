'use client'

import React from 'react'
import { Clock, Loader2, CheckCircle2, AlertCircle } from 'lucide-react'
import { FileProcessingStatus } from '../types'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './tooltip'

interface StatusIndicatorProps {
  status: FileProcessingStatus
  errorMessage?: string
  className?: string
}

export function StatusIndicator({ status, errorMessage, className = '' }: StatusIndicatorProps) {
  const getStatusConfig = () => {
    switch (status) {
      case FileProcessingStatus.NOT_STARTED:
        return {
          icon: Clock,
          color: 'text-gray-500',
          label: 'Waiting to process'
        }
      case FileProcessingStatus.IN_PROGRESS:
        return {
          icon: Loader2,
          color: 'text-blue-500 animate-spin',
          label: 'Processing'
        }
      case FileProcessingStatus.COMPLETED:
        return {
          icon: CheckCircle2,
          color: 'text-green-500',
          label: 'Completed'
        }
      case FileProcessingStatus.FAILED:
        return {
          icon: AlertCircle,
          color: 'text-red-500',
          label: 'Failed'
        }
      default:
        return {
          icon: Clock,
          color: 'text-gray-500',
          label: 'Unknown'
        }
    }
  }

  const config = getStatusConfig()
  const Icon = config.icon

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className={`flex items-center gap-2 ${className}`}>
            <Icon className={`h-4 w-4 ${config.color}`} />
            <span className="text-sm">{config.label}</span>
            {errorMessage && status === FileProcessingStatus.FAILED && (
              <AlertCircle className="h-4 w-4 text-red-500" />
            )}
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {status === FileProcessingStatus.FAILED && errorMessage
              ? `Error: ${errorMessage}`
              : config.label}
          </p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}
